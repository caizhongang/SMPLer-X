import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

from ..losses import loss_entry

__all__ = ['pedattr_cls_vit_A']


class pedattr_cls_vit_A(nn.Module):
    def __init__(self, out_feature, nattr, loss_cfg, use_sync_bn=False, \
                    bn_group=None, bn_momentum=0.1, \
                    bn_eps=1.e-5, **kwargs):

        super(pedattr_cls_vit_A, self).__init__()

        global BN

        def BNFunc(*args, **kwargs):
            class SyncBatchNorm1d(torch.nn.SyncBatchNorm):
                def forward(self, input):
                    assert input.dim() == 2
                    output = super(SyncBatchNorm1d, self).forward(input.unsqueeze(-1).unsqueeze(-1))
                    return output.squeeze(dim=2).squeeze(dim=2)
            return SyncBatchNorm1d(*args, **kwargs, process_group=bn_group, momentum=bn_momentum, eps=bn_eps)

        if use_sync_bn:
            BN = BNFunc
        else:
            BN = nn.BatchNorm1d

        self.logits = nn.Sequential(
            nn.Linear(out_feature, nattr),
            nn.BatchNorm1d(nattr)
        )

        self.avg_pool2d = nn.AdaptiveAvgPool2d(1)
        self.loss = loss_entry(loss_cfg)

    def forward(self, input_var):
        output = {}
        patch_feats = input_var['neck_output']
        patch_feats = self.avg_pool2d(patch_feats).squeeze()
        logits = self.logits(patch_feats)
        output['pred_logits'] = logits
        output.update(input_var)
        losses = self.loss(output)
        output.update(losses)
        return output
