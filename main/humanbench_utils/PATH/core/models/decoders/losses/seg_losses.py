import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import HungarianMatcher, DirectMatcher, RedundantQMatcher
from .criterion import SetCriterion


class FSCELoss(nn.Module):
    def __init__(self, configer=None, **kwargs):
        super(FSCELoss, self).__init__()
        self.configer = configer
        weight = None
        if 'ce_weight' in self.configer:
            weight = self.configer['ce_weight']
            weight = torch.FloatTensor(weight).cuda()

        reduction = 'elementwise_mean'
        if 'ce_reduction' in self.configer:
            reduction = self.configer['ce_reduction']

        ignore_index = -1
        if 'ce_ignore_index' in self.configer:
            ignore_index = self.configer['ce_ignore_index']

        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        # import pdb;
        # pdb.set_trace()
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)
                
            for i in range(len(inputs)):
                if i == 0:
                    if len(targets) > 1:
                        target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                        loss = weights[i] * self.ce_loss(inputs[i], target)
                    else:
                        target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                        loss = weights[i] * self.ce_loss(inputs[i], target)
                else:
                    if len(targets) > 1:
                        target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                        loss += weights[i] * self.ce_loss(inputs[i], target)
                    else:
                        target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                        loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()

class FSCELoss_list(FSCELoss):
    def __int__(self):
        super(FSCELoss_list, self).__init__()

    def forward(self, inputs, *targets, weights=None, **kwargs):
        # import pdb;
        # pdb.set_trace()
        losses = []
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss = weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss = weights[i] * self.ce_loss(inputs[i], target)
                losses.append(loss)
        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)
            losses.append(loss)
        # pdb.set_trace()
        return losses


class FSAuxCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        # import pdb;pdb.set_trace()
        aux_out, seg_out = inputs
        seg_loss = self.ce_loss(seg_out, targets)
        aux_loss = self.ce_loss(aux_out, targets)
        loss = self.configer['loss_weights']['seg_loss'] * seg_loss
        loss = loss + self.configer['loss_weights']['aux_loss'] * aux_loss
        return loss

class FSAuxCELoss_dict(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxCELoss_dict, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss_list(self.configer)

    def forward(self, inputs, targets, **kwargs):
        # import pdb;pdb.set_trace()
        aux_out, seg_out = inputs
        seg_loss = self.ce_loss(seg_out, targets)
        aux_loss = self.ce_loss(aux_out, targets)
        # pdb.set_trace()
        losses = {}
        seg_weight = self.configer['loss_weights']['seg_loss']
        aux_weight = self.configer['loss_weights']['aux_loss']
        for idx, loss in enumerate(seg_loss):
            losses[f'loss_seg_{idx}'] = loss * seg_weight
        for idx, loss in enumerate(aux_loss):
            losses[f'loss_aux_{idx}'] = loss * aux_weight

        return losses

class FocalDiceLoss(nn.Module):
    def __init__(self, cfg):
        super(FocalDiceLoss, self).__init__()
        matcher = HungarianMatcher(
            cost_class=cfg.class_weight,
            cost_mask=cfg.mask_weight,
            cost_dice=cfg.dice_weight,
            num_points=cfg.num_points,
        )

        weight_dict = {"loss_ce": cfg.class_weight,
                       "loss_mask": cfg.mask_weight,
                       "loss_dice": cfg.dice_weight}

        if cfg.deep_supervision:
            aux_weight_dict = {}
            for i in range(cfg.dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})   # {loss_ce_i : cfg.class_weight ...}
            weight_dict.update(aux_weight_dict)

        self.fd_loss = SetCriterion(
            cfg.num_classes,
            ginfo=cfg.ginfo,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=cfg.no_object_weight,
            losses=["labels", "masks"],
            num_points=cfg.num_points,
            oversample_ratio=cfg.oversample_ratio,
            importance_sample_ratio=cfg.importance_sample_ratio,
        )

        self.cfg = cfg

    def forward(self, outputs, targets, **kwargs): # {"aux_outputs": xx, 'xx': xx}
        losses = self.fd_loss(outputs, targets)

        for k in list(losses.keys()):
            if k in self.fd_loss.weight_dict:
                losses[k] *= self.fd_loss.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses

class FocalDiceLoss_no_cls_emb(nn.Module):
    def __init__(self, cfg):
        super(FocalDiceLoss_no_cls_emb, self).__init__()
        matcher = DirectMatcher(num_points=cfg.num_points,)

        weight_dict = { #"loss_ce": cfg.class_weight,
                       "loss_mask": cfg.mask_weight,
                       "loss_dice": cfg.dice_weight}

        if cfg.deep_supervision:
            aux_weight_dict = {}
            for i in range(cfg.dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})   # {loss_ce_i : cfg.class_weight ...}
            weight_dict.update(aux_weight_dict)

        self.fd_loss = SetCriterion(
            cfg.num_classes,
            ginfo=cfg.ginfo,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=cfg.no_object_weight,
            losses=[
                # "labels",
                "masks",
            ],
            num_points=cfg.num_points,
            oversample_ratio=cfg.oversample_ratio,
            importance_sample_ratio=cfg.importance_sample_ratio,
        )

        self.cfg = cfg

    def forward(self, outputs, targets, **kwargs):  # {"aux_outputs": xx, 'xx': xx}
        losses = self.fd_loss(outputs, targets)

        for k in list(losses.keys()):
            if k in self.fd_loss.weight_dict:
                losses[k] *= self.fd_loss.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses

class FocalDiceLoss_bce_cls_emb(nn.Module):
    def __init__(self, cfg):
        super(FocalDiceLoss_bce_cls_emb, self).__init__()
        matcher = DirectMatcher(num_points=cfg.num_points,)

        weight_dict = { "loss_bce": cfg.class_weight,
                       "loss_mask": cfg.mask_weight,
                       "loss_dice": cfg.dice_weight}

        if cfg.deep_supervision:
            aux_weight_dict = {}
            for i in range(cfg.dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})   # {loss_ce_i : cfg.class_weight ...}
            weight_dict.update(aux_weight_dict)

        self.fd_loss = SetCriterion(
            cfg.num_classes,
            ginfo=cfg.ginfo,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=cfg.no_object_weight,
            losses=[
                "bce_labels",
                "masks",
            ],
            num_points=cfg.num_points,
            oversample_ratio=cfg.oversample_ratio,
            importance_sample_ratio=cfg.importance_sample_ratio,
        )

        self.cfg = cfg

    def forward(self, outputs, targets, **kwargs):  # {"aux_outputs": xx, 'xx': xx}
        losses = self.fd_loss(outputs, targets)

        for k in list(losses.keys()):
            if k in self.fd_loss.weight_dict:
                losses[k] *= self.fd_loss.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses

class FocalDiceLoss_bce_cls_emb_sample_weight(FocalDiceLoss):
    def __init__(self, cfg):
        super(FocalDiceLoss_bce_cls_emb_sample_weight, self).__init__(cfg)
        matcher = DirectMatcher(num_points=cfg.num_points,)

        weight_dict = { "loss_bce": cfg.class_weight,
                       "loss_mask": cfg.mask_weight,
                       "loss_dice": cfg.dice_weight}

        if cfg.deep_supervision:
            aux_weight_dict = {}
            for i in range(cfg.dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})   # {loss_ce_i : cfg.class_weight ...}
            weight_dict.update(aux_weight_dict)

        self.fd_loss = SetCriterion(
            cfg.num_classes,
            ginfo=cfg.ginfo,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=cfg.no_object_weight,
            losses=[
                "bce_labels",
                "masks",
            ],
            num_points=cfg.num_points,
            oversample_ratio=cfg.oversample_ratio,
            importance_sample_ratio=cfg.importance_sample_ratio,
            sample_weight = cfg.get('sample_weight', None)
        )

        self.cfg = cfg


class FocalDiceLoss_no_cls_emb_RedundantQ(nn.Module):
    def __init__(self, cfg):
        super(FocalDiceLoss_no_cls_emb_RedundantQ, self).__init__()

        matcher = RedundantQMatcher(
            cost_class=cfg.class_weight,
            cost_mask=cfg.mask_weight,
            cost_dice=cfg.dice_weight,
            num_points=cfg.num_points,
            redundant_queries=cfg.redundant_queries,
            add_zero_mask=cfg.get('add_zero_mask', False)
        )

        weight_dict = { #"loss_ce": cfg.class_weight,
                       "loss_mask": cfg.mask_weight,
                       "loss_dice": cfg.dice_weight}

        if cfg.deep_supervision:
            aux_weight_dict = {}
            for i in range(cfg.dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})   # {loss_ce_i : cfg.class_weight ...}
            weight_dict.update(aux_weight_dict)

        self.fd_loss = SetCriterion(
            cfg.num_classes,
            ginfo=cfg.ginfo,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=cfg.no_object_weight,
            losses=[
                # "labels",
                "masks",
            ],
            num_points=cfg.num_points,
            oversample_ratio=cfg.oversample_ratio,
            importance_sample_ratio=cfg.importance_sample_ratio,
        )

        self.cfg = cfg

    def forward(self, outputs, targets, **kwargs):  # {"aux_outputs": xx, 'xx': xx}
        losses = self.fd_loss(outputs, targets)

        for k in list(losses.keys()):
            if k in self.fd_loss.weight_dict:
                losses[k] *= self.fd_loss.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses

