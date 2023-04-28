import time

import numpy as np
import torch
from core import distributed_utils as dist

from easydict import EasyDict as edict

import warnings
from torch._six import inf
from core.utils import sync_print

# return if any inf/nan
# div norm by loss_scale, for 'real' norm
# if auto_clipper provided, compute max_norm using auto_clipper
# else, using give max_norm
def clip_grad_norm_(parameters, max_norm=1000000, norm_type=2, auto_clipper=None, loss_scale=1.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p[1].grad is not None, parameters))

    if len(parameters) == 0: return None

    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for name,p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type

        total_norm = total_norm ** (1. / norm_type)

    # check inf/nan
    overflow_num = torch.zeros(1)
    if np.isinf(total_norm) or np.isnan(total_norm):
        overflow_num[0] = 1
    dist.allreduce(overflow_num)

    if overflow_num > 0:
        for name,p in parameters:
            p.grad.data.fill_(float('nan'))
        sync_print('total_norm is inf({})/nan({}), skip clipping!!!'.format(np.isinf(total_norm), np.isnan(total_norm)))
        return total_norm

    # rescale the total_norm by loss_scale
    total_norm /= loss_scale

    # update auto_clipper, compute max_norm
    if auto_clipper is not None:
        max_norm = auto_clipper.update(total_norm)

    # do clipping
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        # sync_print('clip_coef: {}'.format(clip_coef))
        for _, p in parameters:
            p.grad.data.mul_(clip_coef)

    return total_norm

class ClipMeter(object):
    def __init__(self, mom=None, thresh=None, min_max=False, mean=False, init=False):
        self.thresh = thresh
        self.mom = mom
        self.min_max = min_max
        self.mean = mean
        self.val = 1.0
        self.init = init

    def get_mean(self):
        return self.val

    def get_clip_val(self):
        if self.mean:
            return self.get_mean()
        else:
            return self.get_mean() * (1+self.thresh)

    def update(self, x):
        if self.init:
            self.val = x
            self.init = False
        mean = self.get_mean()
        if self.min_max:
            x = max(min(x, mean*(1+self.thresh)), mean*(1-self.thresh))
        else:
            x = min(x, mean*(1+self.thresh))

        self.val = self.mom * self.val + (1-self.mom)*x
        return self.get_clip_val()
