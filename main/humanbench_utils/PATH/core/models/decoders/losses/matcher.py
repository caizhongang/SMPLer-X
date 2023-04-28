# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from .point_features import point_sample
from ...ops.box_ops import box_cxcywh_to_xyxy, giou_iou
import numpy as np


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


# batch_sigmoid_ce_loss_jit = torch.jit.script(
#     batch_sigmoid_ce_loss
# )  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]
            # print(f"out_prob: {out_prob}, out_prob.shape: {out_prob.shape} \n tgt_ids: {tgt_ids}") #
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]  # [valid_classes, 1, H, W]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1) # [valid_classes, self.num_points]

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1) # [num_queries, self.num_points]

            out_mask = out_mask.float()
            tgt_mask = tgt_mask.float()
            # Compute the focal loss between masks
            # cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
            cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)  # [num_queries, valid_classes]

            # Compute the dice loss betwen masks
            # cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            cost_dice = batch_dice_loss(out_mask, tgt_mask)  # [num_queries, valid_classes]

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices  # (row_index from num_queries, col_index fron valid_classes)
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class DirectMatcher(HungarianMatcher):
    def __int__(self):
        super(DirectMatcher, self).__int__(num_points=10)

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        bs, _ = outputs["pred_masks"].shape[:2]
        # import pdb;pdb.set_trace()
        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]

            row_idx = tgt_ids.cpu().tolist()
            col_idx = list(range(len(tgt_ids)))

            indices.append((row_idx, col_idx))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices  # (row_index from num_queries, col_index from valid_classes)
        ]

class POSDirectMatcher(HungarianMatcher):
    def __int__(self):
        super(POSDirectMatcher, self).__int__(num_points=10)

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        bs, _ = outputs["pred_masks"].shape[:2]
        # import pdb;pdb.set_trace()
        indices = []
        for b in range(bs):
            tgt_ids = targets['target_weight'][b].nonzero()[:,0]

            row_idx = tgt_ids.cpu().tolist()
            col_idx = list(range(len(tgt_ids)))

            indices.append((row_idx, col_idx))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices  # (row_index from num_queries, col_index from valid_classes)
        ]

class RedundantQMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0,
                 redundant_queries: int=5, add_zero_mask=False ):
        # import pdb;
        # pdb.set_trace()
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points
        self.redundant_queries = redundant_queries
        self.add_zero_mask = add_zero_mask

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        bs, _ = outputs["pred_masks"].shape[:2]

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            num_tgt_ids = len(tgt_ids)
            # (row_index from num_queries, col_index fron valid_classes)

            # import pdb;
            # pdb.set_trace()

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            num_queries = out_mask.shape[0]

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]  # [valid_classes, 1, H, W]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)  # [valid_classes, self.num_points]

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)  # [num_queries, self.num_points]

            out_mask = out_mask.float()
            tgt_mask = tgt_mask.float()
            # Compute the focal loss between masks
            # cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
            cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)  # [num_queries, valid_classes]

            # Compute the dice loss betwen masks
            # cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            cost_dice = batch_dice_loss(out_mask, tgt_mask)  # [num_queries, valid_classes]

            C = (
                    self.cost_mask * cost_mask
                    + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            cost = torch.full(cost_dice.shape, float("inf"), )
            # import pdb;pdb.set_trace()
            for i, label in enumerate(tgt_ids):
                cost[label * self.redundant_queries:(label + 1) * self.redundant_queries, i] = 0
            # pdb.set_trace()
            cost += C

            indices.append(linear_sum_assignment(cost))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices  # (row_index from num_queries, col_index fron valid_classes)
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


def masked_assignment(c, mask=None) -> list:
    if mask is None:
        return linear_sum_assignment(c)
    c[mask == 0] = 1e12
    iSet, jSet = linear_sum_assignment(c)
    keep = mask[iSet, jSet]
    return iSet[keep], jSet[keep]


def relation_net_assignment(c, alive_mask, cur_all_mask, valid_mask):
    c = c.numpy()
    i_0, j_0 = masked_assignment(c[alive_mask], valid_mask[alive_mask])
    selected = np.zeros(c.shape[1], dtype=bool)
    selected[j_0] = 1
    i_1, j_1 = masked_assignment(c[cur_all_mask][:, ~selected], valid_mask[cur_all_mask][:, ~selected])
    i_1 = np.where(cur_all_mask)[0][i_1]
    j_1 = np.where(~selected)[0][j_1]
    return i_1, j_1


# HungarianMatcher for detect
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------
class DetectionHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
            tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
            cost_giou, iou = giou_iou(out_bbox_xyxy, tgt_bbox_xyxy)
            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * -cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            if outputs.get('mask', None) is not None:
                valid_mask = (tgt_bbox_xyxy[None, :, 0] < out_bbox[:, None, 0]) \
                           & (out_bbox[:, None, 0] < tgt_bbox_xyxy[None, :, 2]) \
                           & (tgt_bbox_xyxy[None, :, 1] < out_bbox[:, None, 1]) \
                           & (out_bbox[:, None, 1] < tgt_bbox_xyxy[None, :, 3]) \
                           & (iou > 0)
                valid_mask = valid_mask.view(bs, num_queries, -1).cpu()
                valid_mask = [m[i].bool().numpy() for i, m in enumerate(valid_mask.split(sizes, -1))]
                alive_mask = outputs['mask']['seed_mask'].cpu().numpy()
                cur_all_mask = outputs['mask']['mask'].cpu().numpy()
                indices = [relation_net_assignment(c[i], alive_mask[i], cur_all_mask[i], valid_mask[i]) for i, c in enumerate(C.split(sizes, -1))]
            else:
                indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_bbox: {}".format(self.cost_bbox),
            "cost_giou: {}".format(self.cost_giou),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

