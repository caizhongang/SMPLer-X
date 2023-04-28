import torch
from torch._six import inf

from .utils import iter_params

__all__ = ['scale_check_overflow', 'LossScaler']

# from apex_C import scale_check_overflow

# Python stopgap, until we get a future-proof kernel into upstream
def scale_check_overflow(d_grads, scale):
    any_infinite = ((d_grads != d_grads) | (d_grads.abs() == inf)).any()
    if any_infinite:
        return True
    d_grads.mul_(scale)
    return False
      
class LossScaler(object):
    def __init__(self, scale=1.0, dynamic=False):
        self._dynamic = dynamic
        self._loss_scale = 2.**16 if self._dynamic else scale
        self._max_loss_scale = 2.**24
        self._scale_seq_len = 2000
        self._unskipped = 0
        self._has_overflow = False
 
    @property
    def loss_scale(self):
        return self._loss_scale
        
    @property
    def has_overflow(self):
        return self._has_overflow

    def unscale_and_update(self, param_groups, scale):
        if not self._dynamic:
            for p in iter_params(param_groups):
                if p.grad is not None:
                    p.grad.data.mul_(1. / scale)
            return

        self._has_overflow = False
        for p in iter_params(param_groups):
            if p.grad is not None:
                self._has_overflow = scale_check_overflow(p.grad.data,
                                                          1. / scale)
            if self._has_overflow:  
                break

        # if self._overflow_buf.any():
        if self._has_overflow:
            should_skip = True
            self._loss_scale /= 2.
            self._unskipped = 0
        else:
            should_skip = False
            self._unskipped += 1

        if self._unskipped == self._scale_seq_len:
            self._loss_scale = min(self._max_loss_scale, self._loss_scale * 2.)
            self._unskipped = 0

        return should_skip

    def backward(self, loss):
        scaled_loss = loss*self.loss_scale
        scaled_loss.backward()
