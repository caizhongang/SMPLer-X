import os
import logging
import torch
import numpy as np
from easydict import EasyDict
from bisect import bisect_right
import math
import core.fp16 as fp16

class _LRScheduler(object):
    def __init__(self, optimizer, last_iter=-1):
        if not isinstance(optimizer, torch.optim.Optimizer) and not isinstance(optimizer, fp16.FP16_Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            self.has_base_lrs = True
            self._get_base_lrs_later()
        else:
            self.has_base_lrs = False
        #else:
        #    for i, group in enumerate(optimizer.param_groups):
        #        if 'initial_lr' not in group:
        #            raise KeyError("param 'initial_lr' is not specified "
        #                           "in param_groups[{}] when resuming an optimizer".format(i))
        self.last_iter = last_iter

    def _get_base_lrs_later(self):
        self.base_lrs = list(map(lambda group: group['initial_lr'], self.optimizer.param_groups))

    def _get_new_lr(self):
        raise NotImplementedError

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if not self.has_base_lrs:
            # called when optimizer is recovered after lr_scheduler
            # then at __init__ there is no 'initial_lr'
            self._get_base_lrs_later()

        if this_iter is None:
            this_iter = self.last_iter + 1
        self.last_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_new_lr()):
            param_group['lr'] = lr

class _WarmUpLRScheduler(_LRScheduler):

    def __init__(self, optimizer, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        if warmup_steps == 0:
            self.warmup_lr = base_lr
        else:
            self.warmup_lr = warmup_lr
        super(_WarmUpLRScheduler, self).__init__(optimizer, last_iter)

    def _get_warmup_lr(self):
        if self.warmup_steps > 0 and self.last_iter < self.warmup_steps:
            # first compute relative scale for self.base_lr, then multiply to base_lr
            scale = ((self.last_iter/self.warmup_steps)*(self.warmup_lr - self.base_lr) + self.base_lr)/self.base_lr
            #print('last_iter: {}, warmup_lr: {}, base_lr: {}, scale: {}'.format(self.last_iter, self.warmup_lr, self.base_lr, scale))
            return [scale * base_lr for base_lr in self.base_lrs]
        else:
            return None

class StepLRScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, lr_steps, lr_mults, base_lr, warmup_lr, warmup_steps, last_iter=-1, max_iter=None):
        super(StepLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_steps, last_iter)

        assert len(lr_steps) == len(lr_mults), "{} vs {}".format(milestone, lr_mults)
        for x in lr_steps:
            assert isinstance(x, int)
        if not list(lr_steps) == sorted(lr_steps):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', lr_steps)
        self.lr_steps = lr_steps
        self.lr_mults = [1.0]
        for x in lr_mults:
            self.lr_mults.append(self.lr_mults[-1]*x)

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        pos = bisect_right(self.lr_steps, self.last_iter)
        scale = self.warmup_lr*self.lr_mults[pos] / self.base_lr
        return [base_lr*scale for base_lr in self.base_lrs]

class CosineLRScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, max_iter, eta_min, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        super(CosineLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_steps, last_iter)
        self.max_iter = max_iter
        self.eta_min = eta_min

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        step_ratio = (self.last_iter-self.warmup_steps) / (self.max_iter-self.warmup_steps)
        target_lr = self.eta_min + (self.warmup_lr - self.eta_min)*(1 + math.cos(math.pi * step_ratio)) / 2
        scale = target_lr / self.base_lr
        return [scale*base_lr for base_lr in self.base_lrs]

class WarmupCosineLRScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, max_iter, warmup_iters,
                 warmup_factor=1e-2, warmup_method="linear", last_iter=-1, base_lr=0.8, **kwargs):
        super(WarmupCosineLRScheduler, self).__init__(optimizer, warmup_factor*base_lr, base_lr, warmup_iters, last_iter)
        if warmup_method not in ("constant", "linear"):
            raise ValueError(f"Only 'constant' or 'linear' warmup_method accepted. Got {warmup_method}")

        self.max_iter = max_iter
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method


    def _get_new_lr(self):
        warmup_factor = 1
        if self.last_iter < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_iter) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            warmup_factor * base_lr * (1 + math.cos(math.pi * self.last_iter / self.max_iter)) / 2
            for base_lr in self.base_lrs
        ]

class WarmupCosineLRCyclelimitScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, max_iter, warmup_iters,
                 warmup_factor=1e-2, warmup_method="linear", last_iter=-1, cycle_limit=0, lr_min=0, base_lr=0.8, **kwargs):
        super(WarmupCosineLRScheduler, self).__init__(optimizer, warmup_factor*base_lr, base_lr, warmup_iters, last_iter)
        if warmup_method not in ("constant", "linear"):
            raise ValueError(f"Only 'constant' or 'linear' warmup_method accepted. Got {warmup_method}")

        self.max_iter = max_iter
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.cycle_limit = 0
        self.lr_min = 0

    def _get_new_lr(self):
        warmup_factor = 1
        if self.last_iter < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_iter) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        if self.cycle_limit == 0 or (self.cycle_limit > 0 and self.last_iter < self.cycle_limit):
            return [
                warmup_factor * base_lr * (1 + math.cos(math.pi * self.last_iter / self.max_iter)) / 2
                for base_lr in self.base_lrs
            ]
        else:
            return [self.lr_min for _ in self.base_lrs]


class WarmupPolyLRScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, max_iter, warmup_iters,
                 warmup_factor=1e-2, warmup_method="linear", last_iter=-1, base_lr=0.8, power=0.9):
        super(WarmupPolyLRScheduler, self).__init__(optimizer, warmup_factor*base_lr, base_lr, warmup_iters, last_iter)
        if warmup_method not in ("constant", "linear"):
            raise ValueError(f"Only 'constant' or 'linear' warmup_method accepted. Got {warmup_method}")

        self.max_iter = max_iter
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

        self.power = power

    def _get_new_lr(self):
        warmup_factor = 1
        if self.last_iter < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_iter) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            warmup_factor * base_lr * math.pow((1.0 - self.last_iter / self.max_iter), self.power)
            for base_lr in self.base_lrs
        ]
