import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from ...ops.hrt_helpers.transformer_block import GeneralTransformerBlock
from ..losses import loss_entry


__all__ = ['reid_cls_vit_B']

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class reid_cls_vit_B(nn.Module):
    def __init__(self, out_feature, loss_cfg, feature_bn=True, use_sync_bn=False, \
                    bn_group=None, bn_momentum=0.1, \
                    bn_eps=1.e-5, feature_only=False, **kwargs):
        super(reid_cls_vit_B, self).__init__()

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

        self.feature_bn = feature_bn
        if feature_bn:
            self.feat_bn = BN(out_feature)
            self.feat_bn.bias.requires_grad_(False)
            self.feat_bn.apply(weights_init_kaiming)

        self.loss = loss_entry(loss_cfg)
        self.feature_only = feature_only

    def forward(self, input_var):
        output = {}
        features_nobn = input_var['neck_output']['cls_feats']
        if self.feature_bn:
            features = self.feat_bn(features_nobn)
        labels = input_var['label']
        output['feature'] = features
        output['feature_nobn'] = features_nobn
        output['label'] = labels
        if self.feature_only: return output
        logits = self.loss(output)
        output.update(logits)
        return output
