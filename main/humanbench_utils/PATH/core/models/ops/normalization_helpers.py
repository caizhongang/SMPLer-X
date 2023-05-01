import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from spring.linklink.nn import SyncBatchNorm2d

class IBN(nn.Module):
    def __init__(self, planes, BN):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = BN(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


def get_normalization(num_features, bn_type=None, **kwargs):
    if bn_type == "bn":
        return nn.BatchNorm2d(num_features, **kwargs)
    elif bn_type == 'syncbn':
        return SyncBatchNorm2d(num_features, **kwargs, group=group, \
        sync_stats=sync_stats, momentum=bn_mom, eps=bn_eps)
    elif bn_type == 'ibn':
        assert half_bn is not None, "Half BN is not set!"
        return IBN(num_features, half_bn)
    elif bn_type == 'sn':
        raise NotImplementedError('switchable normalization is not supported!')
    elif bn_type == 'gn':
        raise NotImplementedError('group normalization is not supported!')
    elif bn_type == 'ln':
        raise NotImplementedError('layer normalization is not supported!')
    else:
        raise ValueError('bn_type ({}) is not supported.'.format(bn_type))
