""" Ref: https://github.com/vchoutas/torch-trust-ncg/blob/main/torchtrustncg/trust_region.py """

# -*- coding: utf-8 -*-

# @author Vasileios Choutas
# Contact: vassilis.choutas@tuebingen.mpg.de


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from typing import NewType, List, Tuple

import torch
from torch import norm
import torch.optim as optim
import torch.autograd as autograd

import math

Tensor = NewType('Tensor', torch.Tensor)


def eye_like(tensor, device):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor, device=device), device=device)


class TrustRegion(optim.Optimizer):

    def __init__(
        self,
        params: List[Tensor],
        max_trust_radius: float = 1000,
        initial_trust_radius: float = 0.5,
        eta: float = 0.15,
        gtol: float = 1e-05,
        kappa_easy: float = 0.1,
        max_newton_iter: int = 50,
        max_krylov_dim: int = 15,
        lanczos_tol: float = 1e-4,
        opt_method: str = 'cg',
        epsilon: float = 1.0e-09,
        **kwargs
    ) -> None:
        """ Trust Region
                Newton Conjugate Gradient
                    Uses the Conjugate Gradient Algorithm to find the solution of the
                    trust region sub-problem. For more details see Algorithm 7.2 of
                    "Numerical Optimization, Nocedal and Wright"
                Generalized Lanczos Method
                    Uses the GEneralized Lanczos Algorithm to find the solution of the
                    trust region sub-problem. For more details see Algorithm7.5.2 of
                    "Trust Region Methods, Conn et al."
                    Arguments:
                        params (iterable): A list or iterable of tensors that will be
                            optimized
                        max_trust_radius: float
                            The maximum value for the trust radius
                        initial_trust_radius: float
                            The initial value for the trust region
                        eta: float
                            Minimum improvement ration for accepting a step
                        kappa_easy: float
                            Parameter related to the convergence of Krylov method, see Lemma 7.3.5 Conn et al.
                        max_newton_iter: int
                            Maximum Newton iterations for root finding
                        max_krylov_dim: int
                            Maximum Krylov dimension
                        lanczos_tol: float
                            Approximation error of the optimizer in Krylov subspace, see Theorem 7.5.10 Conn et al.
                        opt_method: string
                            The method to solve the subproblem.
                        gtol: float
                            Gradient tolerance for stopping the optimization
        """
        defaults = dict()

        super(TrustRegion, self).__init__(params, defaults)

        self.steps = 0
        self.max_trust_radius = max_trust_radius
        self.initial_trust_radius = initial_trust_radius
        self.eta = eta
        self.gtol = gtol
        self._params = self.param_groups[0]['params']

        self.kappa_easy = kappa_easy
        self.opt_method = opt_method
        self.lanczos_tol = lanczos_tol
        self.max_krylov_dim = max_krylov_dim
        self.max_newton_iter = max_newton_iter
        self.kwargs = kwargs

        self.epsilon = epsilon

        self.T_lambda = lambda _lambda, T_x, device: T_x.to(
            device) + _lambda * eye_like(T_x, device)
        self.lambda_const = lambda lambda_k: (
            1 + lambda_k) * torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps))

        if not (opt_method == 'cg' or opt_method == 'krylov'):
            raise ValueError('opt_method should be "cg" or "krylov"')

    @torch.enable_grad()
    def _compute_hessian_vector_product(
            self,
            gradient: Tensor,
            p: Tensor) -> Tensor:

        hess_vp = autograd.grad(
            torch.sum(gradient * p, dim=-1), self._params,
            only_inputs=True, retain_graph=True, allow_unused=True)
        return torch.cat([torch.flatten(vp) for vp in hess_vp], dim=-1)
        #  hess_vp = torch.cat(
        #  [torch.flatten(vp) for vp in hess_vp], dim=-1)
        #  return torch.flatten(hess_vp)

    def _gather_flat_grad(self) -> Tensor:
        """ Concatenates all gradients into a single gradient vector
        """
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        output = torch.cat(views, 0)
        return output

    @torch.no_grad()
    def _improvement_ratio(self, p, start_loss, gradient, closure):
        """ Calculates the ratio of the actual to the expected improvement

            Arguments:
                p (torch.tensor): The update vector for the parameters
                start_loss (torch.tensor): The value of the loss function
                    before applying the optimization step
                gradient (torch.tensor): The flattened gradient vector of the
                    parameters
                closure (callable): The function that evaluates the loss for
                    the current values of the parameters
            Returns:
                The ratio of the actual improvement of the loss to the expected
                improvement, as predicted by the local quadratic model
        """

        # Apply the update on the parameter to calculate the loss on the new
        # point
        hess_vp = self._compute_hessian_vector_product(gradient, p)

        # Apply the update of the parameter vectors.
        # Use a torch.no_grad() context since we are updating the parameters in
        # place
        with torch.no_grad():
            start_idx = 0
            for param in self._params:
                num_els = param.numel()
                curr_upd = p[start_idx:start_idx + num_els]
                param.data.add_(curr_upd.view_as(param))
                start_idx += num_els

        # No need to backpropagate since we only need the value of the loss at
        # the new point to find the ratio of the actual and the expected
        # improvement
        new_loss = closure(backward=False)
        # The numerator represents the actual loss decrease
        numerator = start_loss - new_loss

        new_quad_val = self._quad_model(p, start_loss, gradient, hess_vp)

        # The denominator
        denominator = start_loss - new_quad_val

        # TODO: Convert to epsilon, print warning
        ratio = numerator / (denominator + 1e-20)
        return ratio

    @torch.no_grad()
    def _quad_model(
            self,
            p: Tensor,
            loss: float,
            gradient: Tensor,
            hess_vp: Tensor) -> float:
        """ Returns the value of the local quadratic approximation
        """
        return (loss + torch.flatten(gradient * p).sum(dim=-1) +
                0.5 * torch.flatten(hess_vp * p).sum(dim=-1))

    @torch.no_grad()
    def calc_boundaries(
            self,
            iterate: Tensor,
            direction: Tensor,
            trust_radius: float) -> Tuple[Tensor, Tensor]:
        """ Calculates the offset to the boundaries of the trust region
        """

        a = torch.sum(direction ** 2)
        b = 2 * torch.sum(direction * iterate)
        c = torch.sum(iterate ** 2) - trust_radius ** 2
        sqrt_discriminant = torch.sqrt(b * b - 4 * a * c)
        ta = (-b + sqrt_discriminant) / (2 * a)
        tb = (-b - sqrt_discriminant) / (2 * a)
        if ta.item() < tb.item():
            return [ta, tb]
        else:
            return [tb, ta]

    @torch.no_grad()
    def _solve_subproblem_cg(
            self,
            loss: float,
            flat_grad: Tensor,
            trust_radius: float) -> Tuple[Tensor, bool]:
        ''' Solves the quadratic subproblem in the trust region
        '''

        # The iterate vector that contains the increment from the starting
        # point
        iterate = torch.zeros_like(flat_grad, requires_grad=False)

        # The residual of the CG algorithm
        residual = flat_grad.detach()
        # The first direction of descent
        direction = -residual

        jac_mag = torch.norm(flat_grad).item()
        # Tolerance define in Nocedal & Wright in chapter 7.1
        tolerance = min(0.5, math.sqrt(jac_mag)) * jac_mag

        # If the magnitude of the gradients is smaller than the tolerance then
        # exit
        if jac_mag <= tolerance:
            return iterate, False

        # Iterate to solve the subproblem
        while True:
            # Calculate the Hessian-Vector product
            #  start = time.time()
            hessian_vec_prod = self._compute_hessian_vector_product(
                flat_grad, direction
            )
            #  torch.cuda.synchronize()
            #  print('Hessian Vector Product', time.time() - start)

            # This term is equal to p^T * H * p
            #  start = time.time()
            hevp_dot_prod = torch.sum(hessian_vec_prod * direction)
            #  print('p^T H p', time.time() - start)

            # If non-positive curvature
            if hevp_dot_prod.item() <= 0:
                # Find boundaries and select minimum
                #  start = time.time()
                ta, tb = self.calc_boundaries(iterate, direction, trust_radius)
                pa = iterate + ta * direction
                pb = iterate + tb * direction

                # Calculate the point on the boundary with the smallest value
                bound1_val = self._quad_model(pa, loss, flat_grad,
                                              hessian_vec_prod)
                bound2_val = self._quad_model(pb, loss, flat_grad,
                                              hessian_vec_prod)
                #  torch.cuda.synchronize()
                #  print('First if', time.time() - start)
                #  print()
                if bound1_val.item() < bound2_val.item():
                    return pa, True
                else:
                    return pb, True

            # The squared euclidean norm of the residual needed for the CG
            # update
            #  start = time.time()
            residual_sq_norm = torch.sum(residual * residual, dim=-1)

            # Compute the step size for the CG algorithm
            cg_step_size = residual_sq_norm / hevp_dot_prod

            # Update the point
            next_iterate = iterate + cg_step_size * direction

            iterate_norm = torch.norm(next_iterate, dim=-1)
            #  torch.cuda.synchronize()
            #  print('CG Updates', time.time() - start)

            # If the point is outside of the trust region project it on the
            # border and return
            if iterate_norm.item() >= trust_radius:
                #  start = time.time()
                ta, tb = self.calc_boundaries(iterate, direction, trust_radius)
                p_boundary = iterate + tb * direction

                #  torch.cuda.synchronize()
                #  print('Second if', time.time() - start)
                #  print()
                return p_boundary, True

            #  start = time.time()
            # Update the residual
            next_residual = residual + cg_step_size * hessian_vec_prod
            #  torch.cuda.synchronize()
            #  print('Residual update', time.time() - start)
            # If the residual is small enough, exit
            if torch.norm(next_residual, dim=-1).item() < tolerance:
                #  print()
                return next_iterate, False

            #  start = time.time()
            beta = torch.sum(next_residual ** 2, dim=-1) / residual_sq_norm
            # Compute the new search direction
            direction = (-next_residual + beta * direction).squeeze()
            if torch.isnan(direction).sum() > 0:
                raise RuntimeError

            iterate = next_iterate
            residual = next_residual
            #  torch.cuda.synchronize()
            #  print('Replacing vectors', time.time() - start)
            #  print(trust_radius)
            #  print()

    @torch.no_grad()
    def _converged(self, s, trust_radius):

        if abs(norm(s) - trust_radius) <= self.kappa_easy * trust_radius:
            return True
        else:
            return False

    @torch.no_grad()
    def _lambda_one_plus(self, T, device):

        eigen_pairs = torch.linalg.eigh(T)

        Lambda, U = eigen_pairs.eigenvalues, eigen_pairs.eigenvectors
        lambda_n, u_n = Lambda[0].to(device=device), U[:, 0].to(device=device)

        return torch.maximum(-lambda_n, torch.tensor([0], device=device)), lambda_n, u_n[:, None]

    @torch.no_grad()
    def _quad_model_krylov(
            self,
            lanczos_g: Tensor,
            loss: float,
            s_x: Tensor,
            T_x: Tensor) -> float:
        """
         Returns the value of the local quadratic approximation
        """

        return (loss + torch.sum(lanczos_g * s_x) + 1 / 2 * torch.sum(T_x.mm(s_x) * s_x)).item()

    def _root_finder(self, trust_radius, T_x, lanczos_g, loss, device):

        n_iter_nu, n_iter_r = 0, 0
        lambda_k, lambda_n, u_n = self._lambda_one_plus(T_x, device)
        lambda_const = self.lambda_const(lambda_k).to(device=device)
        if lambda_k == 0:  # T_x is positive definite
            _lambda = torch.tensor(
                [0], dtype=torch.float32, device=device)  # + lambda_const
        else:
            _lambda = lambda_k + lambda_const

        s, L = self._compute_s(_lambda=_lambda, lambda_const=lambda_const,
                               lanczos_g=lanczos_g, T_x=T_x, device=device)

        if norm(s) <= trust_radius:

            if _lambda == 0 or norm(s) == trust_radius:
                return s
            else:
                ta, tb = self.calc_boundaries(
                    iterate=s, direction=u_n, trust_radius=trust_radius)
                pa = s + ta * u_n
                pb = s + tb * u_n

                # Calculate the point on the boundary with the smallest value
                bound1_val = self._quad_model_krylov(lanczos_g, loss, pa, T_x)
                bound2_val = self._quad_model_krylov(lanczos_g, loss, pb, T_x)

                if bound1_val < bound2_val:
                    return pa
                else:
                    return pb

        while True:
            if self._converged(s, trust_radius) or norm(s) < torch.finfo(float).eps:
                break

            # w = torch.triangular_solve(
            #     s, L.T.to(device=device), upper=False).solution
            w = torch.linalg.solve_triangular(
                L.T.to(device=device), s, upper=False)
            _lambda = self._nu_next(_lambda, trust_radius, s, w)

            s, L = self._compute_s(_lambda, lambda_const,
                                   lanczos_g, T_x, device)

            n_iter_nu += 1
            if n_iter_nu > self.max_newton_iter - 1:  # self.max_krylov_dim:
                print(RuntimeWarning(
                    'Maximum number of newton iterations exceeded for _lambda: {}'.format(_lambda)))
                break

        return s

    @torch.no_grad()
    def _nu_next(self, _lambda, trust_radius, s, w):

        norm_s = norm(s)
        norm_w = norm(w)

        phi = 1 / norm_s - 1 / trust_radius

        phi_prime = norm_w ** 2 / norm_s ** 3

        return _lambda - phi / phi_prime

    @torch.no_grad()
    def _compute_s(self, _lambda, lambda_const, lanczos_g, T_x, device):
        try:
            L = torch.linalg.cholesky(self.T_lambda(_lambda, T_x, device))
        except RuntimeError:
            # print('Recursion')
            lambda_const *= 2
            # RecursionError: maximum recursion depth exceeded while calling a Python object
            s, L = self._compute_s(
                _lambda + lambda_const, lambda_const, lanczos_g, T_x, device)

        s = torch.cholesky_solve(-lanczos_g[:, None],
                                 L.to(device=device), upper=True)
        return s, L

    @torch.no_grad()
    def _solve_subproblem_krylov(
            self,
            loss: float,
            flat_grad: Tensor,
            trust_radius: float) -> Tuple[Tensor, bool]:
        """
            Solves the quadratic subproblem in the trust region using Generalized Lanczos Method,
            see Algorithm 7.5.2 Conn et al.
        """
        INTERIOR_FLAG = True
        Q, diagonals, off_diagonals = [], [], []

        flat_grads_detached = flat_grad.detach()
        n_features = len(flat_grads_detached)
        h = torch.zeros_like(flat_grads_detached, requires_grad=False)
        q, p = flat_grads_detached, -flat_grads_detached

        gamma0 = torch.norm(q)

        krylov_dim, sigma = 0, 1

        device = flat_grad.device
        targs = {'device': device, 'dtype': flat_grad.dtype}

        while True:
            Hp = self._compute_hessian_vector_product(flat_grad, p)
            ptHp = torch.sum(Hp * p)
            alpha = torch.norm(q) ** 2 / ptHp
            # if alpha == 0:
            #     print('hard case')
            if krylov_dim == 0:
                diagonals.append(1. / alpha.clamp_(min=self.epsilon).item())
                off_diagonals.append(float('inf'))  # dummy value
                Q.append(sigma * q / norm(q))
                T_x = torch.tensor([diagonals], **targs)
                alpha_prev = alpha
            else:
                diagonals.append(1. / alpha.item() +
                                 beta.item() / alpha_prev.item())
                sigma = - torch.sign(alpha_prev) * sigma
                Q.append(sigma * q / norm(q))
                T_x = (torch.diag(torch.tensor(diagonals, **targs), 0)
                       + torch.diag(torch.tensor(off_diagonals[1:], **targs), -1)
                       + torch.diag(torch.tensor(off_diagonals[1:], **targs), 1))
                alpha_prev = alpha

            if INTERIOR_FLAG and alpha < 0 or torch.norm(h + alpha * p) >= trust_radius:
                INTERIOR_FLAG = False

            if INTERIOR_FLAG:
                h = h + alpha * p
            else:
                # Lanczos Step 2: solve problem in subspace
                e_1 = torch.eye(1, krylov_dim + 1,
                                device=flat_grad.device).flatten()
                lanczos_g = gamma0 * e_1
                s = self._root_finder(trust_radius=trust_radius,
                                      T_x=T_x, lanczos_g=lanczos_g,
                                      loss=loss, device=flat_grad.device)
                s = s.to(flat_grad.device)

            q_next = q + alpha * Hp

            # test for convergence
            if INTERIOR_FLAG and norm(q_next) ** 2 < self.lanczos_tol:
                break
            if not INTERIOR_FLAG and torch.norm(q_next) * abs(s[-1]) < self.lanczos_tol:
                break

            if krylov_dim == n_features:
                # print(RuntimeWarning(
                #     'Krylov dimensionality reach full space! Breaking out..'))
                break
                # return h

            if krylov_dim > self.max_krylov_dim:
                # print(RuntimeWarning('Max Krylov dimension reached! Breaking out..'))
                break

            beta = torch.dot(q_next, q_next) / torch.dot(q, q)
            off_diagonals.append(torch.sqrt(beta) / torch.abs(alpha_prev))
            p = -q_next + beta * p
            q = q_next
            krylov_dim = krylov_dim + 1

        if not INTERIOR_FLAG:
            # Return to the original space
            Q = torch.vstack(Q).T
            h = torch.sum(Q * torch.squeeze(s), dim=1)

        return h, not INTERIOR_FLAG  # INTERIOR_FLAG is False == hit_boundary is True

    def step(self, closure=None) -> float:
        starting_loss = closure(backward=True)

        flat_grad = self._gather_flat_grad()

        state = self.state
        if len(state) == 0:
            state['trust_radius'] = torch.full([1],
                                               self.initial_trust_radius,
                                               dtype=flat_grad.dtype,
                                               device=flat_grad.device)
        trust_radius = state['trust_radius']

        if self.opt_method == 'cg':
            param_step, hit_boundary = self._solve_subproblem_cg(
                starting_loss, flat_grad, trust_radius)
        else:
            param_step, hit_boundary = self._solve_subproblem_krylov(
                starting_loss, flat_grad, trust_radius)

        self.param_step = param_step

        if torch.norm(param_step).item() <= self.gtol:
            return starting_loss

        improvement_ratio = self._improvement_ratio(
            param_step, starting_loss, flat_grad, closure)

        if improvement_ratio.item() < 0.25:
            trust_radius.mul_(0.25)
        else:
            if improvement_ratio.item() > 0.75 and hit_boundary:
                trust_radius.mul_(2).clamp_(0.0, self.max_trust_radius)

        if improvement_ratio.item() <= self.eta:
            # If the improvement is not sufficient, then undo the update
            start_idx = 0
            for param in self._params:
                num_els = param.numel()
                curr_upd = param_step[start_idx:start_idx + num_els]
                param.data.add_(-curr_upd.view_as(param))
                start_idx += num_els

        self.steps += 1
        return starting_loss