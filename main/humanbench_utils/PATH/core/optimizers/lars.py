import torch
from torch.optim.optimizer import Optimizer, required


class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD, based on
    `"Large Batch Training of Convolutional Networks" <https://arxiv.org/abs/1708.03888>`_

    Arguments:
        - params (:obj:`iterable`): iterable of parameters to optimize or dicts defining parameter groups
        - lr (:obj:`float`): learning rate
        - momentum (:obj:`float`, optional): momentum factor (default: 0)
        - weight_decay (:obj:`float`, optional): weight decay (L2 penalty) (default: 0)
        - dampening (:obj:`float`, optional): dampening for momentum (default: 0)
        - eta(:obj:`float`): LARS coefficient (default 0.001)
        - nesterov (:obj:`bool`, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, eta=0.001, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, eta=eta, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            - closure (:obj:`callable`, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            eta = group['eta']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # compute local learning rate
                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p)

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                    grad_norm.add_(weight_decay, weight_norm)
                local_lr = eta * weight_norm / grad_norm

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(
                            p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr']*local_lr, d_p)

        return loss
