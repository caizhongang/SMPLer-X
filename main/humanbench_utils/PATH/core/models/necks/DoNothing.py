import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DoNothing']

class DoNothing(nn.Module):
    def __init__(self, backbone=None, bn_group=None):
        super(DoNothing, self).__init__()

    def forward(self, x):
        x.update({'neck_output': x['backbone_output']})
        return x
