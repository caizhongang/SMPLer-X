import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import HungarianMatcher, DirectMatcher, RedundantQMatcher, POSDirectMatcher
from .criterion import SetCriterion, POSSetCriterion

class BasePosLoss(nn.Module):
    def __init__(self, target_type, use_target_weight=True, cfg=None):
        super(BasePosLoss, self).__init__()
        self.criterion = nn.MSELoss()

        self.target_type = target_type
        self.use_target_weight = use_target_weight

        self.cfg = cfg

    def get_loss(self, num_joints, heatmaps_pred, heatmaps_gt, target_weight):
        loss = 0.
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred * target_weight[:, idx],
                                       heatmap_gt * target_weight[:, idx])
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)
        return loss

    def forward(self, outputs, target, target_weight): # {"aux_outputs": xx, 'xx': xx}
        """Forward function."""
        output = outputs['pred_masks'] # {'pred_logits':'pred_masks':}

        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = self.get_loss(num_joints, heatmaps_pred, heatmaps_gt, target_weight)
        # import pdb;
        # pdb.set_trace()
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs and self.cfg.get('aux_loss', True):
            for aux_outputs in outputs["aux_outputs"]:
                heatmaps_pred = aux_outputs['pred_masks'].reshape((batch_size, num_joints, -1)).split(1, 1)

                loss = loss + self.get_loss(num_joints, heatmaps_pred, heatmaps_gt, target_weight)

        return loss / num_joints


class JointsMSELoss(nn.Module):
    """MSE loss for heatmaps.
    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred * target_weight[:, idx],
                                       heatmap_gt * target_weight[:, idx])
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints * self.loss_weight
