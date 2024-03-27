# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad
from mmengine.hooks import Hook
from torch.nn.modules.batchnorm import _BatchNorm

from ..utils.dist_utils import allreduce_grads
from .utils import cast_tensor_type


class OptimizerHook(Hook):
    """A hook contains custom operations for the optimizer.

    Args:
        grad_clip (dict, optional): A config dict to control the clip_grad.
            Default: None.
        detect_anomalous_params (bool): This option is only used for
            debugging which will slow down the training speed.
            Detect anomalous parameters that are not included in
            the computational graph with `loss` as the root.
            There are two cases

                - Parameters were not used during
                  forward pass.
                - Parameters were not used to produce
                  loss.
            Default: False.
    """

    def __init__(self,
                 grad_clip: Optional[dict] = None,
                 detect_anomalous_params: bool = False):
        self.grad_clip = grad_clip
        self.detect_anomalous_params = detect_anomalous_params

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        runner.outputs['loss'].backward()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()

    def detect_anomalous_parameters(self, loss: Tensor, runner) -> None:
        logger = runner.logger
        parameters_in_graph = set()
        visited = set()

        def traverse(grad_fn):
            if grad_fn is None:
                return
            if grad_fn not in visited:
                visited.add(grad_fn)
                if hasattr(grad_fn, 'variable'):
                    parameters_in_graph.add(grad_fn.variable)
                parents = grad_fn.next_functions
                if parents is not None:
                    for parent in parents:
                        grad_fn = parent[0]
                        traverse(grad_fn)

        traverse(loss.grad_fn)
        for n, p in runner.model.named_parameters():
            if p not in parameters_in_graph and p.requires_grad:
                logger.log(
                    level=logging.ERROR,
                    msg=f'{n} with shape {p.size()} is not '
                    f'in the computational graph \n')

class Fp16OptimizerHook(OptimizerHook):
    """FP16 optimizer hook.

    The steps of fp16 optimizer is as follows.
    1. Scale the loss value.
    2. BP in the fp16 model.
    2. Copy gradients from fp16 model to fp32 weights.
    3. Update fp32 weights.
    4. Copy updated parameters from fp32 weights to fp16 model.

    Refer to https://arxiv.org/abs/1710.03740 for more details.

    Args:
        loss_scale (float): Scale factor multiplied with loss.
    """

    def __init__(self,
                 grad_clip=None,
                 coalesce=True,
                 bucket_size_mb=-1,
                 loss_scale=512.,
                 distributed=True):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.loss_scale = loss_scale
        self.distributed = distributed

    def before_run(self, runner):
        """Preparing steps before Mixed Precision Training.

        1. Make a master copy of fp32 weights for optimization.
        2. Convert the main model from fp32 to fp16.

        Args:
            runner (:obj:`mmcv.Runner`): The underlines training runner.
        """
        # keep a copy of fp32 weights
        runner.optimizer.param_groups = copy.deepcopy(
            runner.optimizer.param_groups)
        # convert model to fp16
        wrap_fp16_model(runner.model)

    @staticmethod
    def copy_grads_to_fp32(fp16_net, fp32_weights):
        """Copy gradients from fp16 model to fp32 weight copy."""
        for fp32_param, fp16_param in zip(fp32_weights, fp16_net.parameters()):
            if fp16_param.grad is not None:
                if fp32_param.grad is None:
                    fp32_param.grad = fp32_param.data.new(fp32_param.size())
                fp32_param.grad.copy_(fp16_param.grad)

    @staticmethod
    def copy_params_to_fp16(fp16_net, fp32_weights):
        """Copy updated params from fp32 weight copy to fp16 model."""
        for fp16_param, fp32_param in zip(fp16_net.parameters(), fp32_weights):
            fp16_param.data.copy_(fp32_param.data)

    def after_train_iter(self, runner):
        """Backward optimization steps for Mixed Precision Training.

        1. Scale the loss by a scale factor.
        2. Backward the loss to obtain the gradients (fp16).
        3. Copy gradients from the model to the fp32 weight copy.
        4. Scale the gradients back and update the fp32 weight copy.
        5. Copy back the params from fp32 weight copy to the fp16 model.

        Args:
            runner (:obj:`mmcv.Runner`): The underlines training runner.
        """
        # clear grads of last iteration
        runner.model.zero_grad()
        runner.optimizer.zero_grad()
        # scale the loss value
        scaled_loss = runner.outputs['loss'] * self.loss_scale
        scaled_loss.backward()
        # copy fp16 grads in the model to fp32 params in the optimizer
        fp32_weights = []
        for param_group in runner.optimizer.param_groups:
            fp32_weights += param_group['params']
        self.copy_grads_to_fp32(runner.model, fp32_weights)
        # allreduce grads
        if self.distributed:
            allreduce_grads(fp32_weights, self.coalesce, self.bucket_size_mb)
        # scale the gradients back
        for param in fp32_weights:
            if param.grad is not None:
                param.grad.div_(self.loss_scale)
        if self.grad_clip is not None:
            self.clip_grads(fp32_weights)
        # update fp32 params
        runner.optimizer.step()
        # copy fp32 params to the fp16 model
        self.copy_params_to_fp16(runner.model, fp32_weights)


def wrap_fp16_model(model):
    """Wrap the FP32 model to FP16.

    1. Convert FP32 model to FP16.
    2. Remain some necessary layers to be FP32, e.g., normalization layers.

    Args:
        model (nn.Module): Model in FP32.
    """
    # convert model to fp16
    model.half()
    # patch the normalization layers to make it work in fp32 mode
    patch_norm_fp32(model)
    # set `fp16_enabled` flag
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True


def patch_norm_fp32(module):
    """Recursively convert normalization layers from FP16 to FP32.

    Args:
        module (nn.Module): The modules to be converted in FP16.

    Returns:
        nn.Module: The converted module, the normalization layers have been
            converted to FP32.
    """
    if isinstance(module, (_BatchNorm, nn.GroupNorm)):
        module.float()
        module.forward = patch_forward_method(module.forward, torch.half,
                                              torch.float)
    for child in module.children():
        patch_norm_fp32(child)
    return module


def patch_forward_method(func, src_type, dst_type, convert_output=True):
    """Patch the forward method of a module.

    Args:
        func (callable): The original forward method.
        src_type (torch.dtype): Type of input arguments to be converted from.
        dst_type (torch.dtype): Type of input arguments to be converted to.
        convert_output (bool): Whether to convert the output back to src_type.

    Returns:
        callable: The patched forward method.
    """

    def new_forward(*args, **kwargs):
        output = func(*cast_tensor_type(args, src_type, dst_type),
                      **cast_tensor_type(kwargs, src_type, dst_type))
        if convert_output:
            output = cast_tensor_type(output, dst_type, src_type)
        return output

    return new_forward
