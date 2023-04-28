from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



__all__ = ['CEL_Sigmoid']

def ratio2weight(targets, ratio):
    ratio = torch.from_numpy(ratio).type_as(targets)
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights[targets > 1] = 0.0

    return weights

class CEL_Sigmoid(nn.Module):
    def __init__(self, sample_weight=None, size_average=True):
        super(CEL_Sigmoid, self).__init__()

        self.sample_weight = sample_weight

        if sample_weight is not None:
            self.sample_weight = np.array(self.sample_weight)

        self.size_average = size_average

    def forward(self, input_var):
        logits = input_var['pred_logits']
        targets = input_var['label']
        batch_size = logits.shape[0]
        
        weight_mask = (targets != -1)
        loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weight_mask, reduction='none')

        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.sample_weight is not None:
            weight = ratio2weight(targets_mask, self.sample_weight)
            loss = (loss * weight.cuda())

        loss = loss.sum() / batch_size if self.size_average else loss.sum()

        output = {'loss': loss, 'top1': torch.Tensor([0]).cuda()}

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'sample_weight=' + str(self.sample_weight) \
            + ', size_average=' + str(self.size_average)  + ')'

class CEL_Sigmoid_Q2L(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(CEL_Sigmoid_Q2L, self).__init__()

        self.sample_weight = cfg['sample_weight']

        if self.sample_weight is not None:
            self.sample_weight = np.array(self.sample_weight)

        self.size_average = cfg['size_average']

    def forward(self, input_var):
        logits = input_var['pred_logits'].squeeze()
        targets = input_var['label']
        batch_size = logits.shape[0]

        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.sample_weight is not None:
            weight = ratio2weight(targets_mask, self.sample_weight)
            loss = (loss * weight.cuda())

        loss = loss.sum() / batch_size if self.size_average else loss.sum()

        output = {'loss': loss, 'top1': torch.Tensor([0]).cuda()}

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'sample_weight=' + str(self.sample_weight) \
            + ', size_average=' + str(self.size_average)  + ')'
