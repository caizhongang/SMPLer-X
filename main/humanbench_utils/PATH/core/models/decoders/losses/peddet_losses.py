import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import DetectionHungarianMatcher
from .criterion import DetSetCriterion

class DetFocalDiceLoss(nn.Module):
    def __init__(self, cfg):
        super(DetFocalDiceLoss, self).__init__()
        matcher = DetectionHungarianMatcher(
            cost_class=cfg.class_weight,
            cost_bbox=cfg.bbox_weight,
            cost_giou=cfg.giou_weight,
        )

        weight_dict = {"loss_ce": cfg.class_weight,
                       "loss_bbox": cfg.bbox_weight,
                       "loss_giou": cfg.giou_weight}

        if cfg.deep_supervision:
            aux_weight_dict = {}
            for i in range(cfg.dec_layers-1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})   # {loss_ce_i : cfg.class_weight ...}
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        self.fd_loss = DetSetCriterion(
            cfg.num_classes,
            ginfo=cfg.ginfo,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=["labels", "boxes"],
            focal_alpha=cfg.focal_alpha,
            ign_thr=cfg.ign_thr,
        )

        self.cfg = cfg

    def forward(self, outputs, targets, **kwargs): # {"aux_outputs": xx, 'xx': xx}
        losses = self.fd_loss(outputs, targets)
        for k in list(losses.keys()):
            if k in self.fd_loss.weight_dict:
                losses[k] *= self.fd_loss.weight_dict[k]
            elif 'loss' in k:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        return losses
