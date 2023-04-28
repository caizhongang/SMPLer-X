import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


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

num_queries = 80
num_points=12304
h,w=120,120
redundant_queries=4

tgt_ids = list(range(20))

out_masks = torch.rand((num_queries,h,w)).cuda()
tgt_masks = torch.rand((20,h,w)).cuda()

import time
s = time.time()
ind = []
for _ in range(10):
    out_prob = torch.full(
                    (num_queries, num_queries//redundant_queries), 0, dtype=torch.float,
                    device=out_masks.device
                )

    for i in range(num_queries // redundant_queries):
        out_prob[4 * i:4 * (i + 1), i] = 1

    cost_class = -out_prob[:, tgt_ids]
    out_mask = out_masks[:, None]
    tgt_mask = tgt_masks[:, None]
    point_coords = torch.rand(1, num_points, 2, device=out_mask.device)
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
    cost_dice = batch_dice_loss(out_mask, tgt_mask)

    c = torch.full(cost_dice.shape, float("inf"),)

    for i in tgt_ids:
        c[i*redundant_queries:(i+1)*redundant_queries, i]=0

    c += cost_dice.cpu()+cost_dice.cpu()

    ind.append(linear_sum_assignment(c))
print([i-4*j for (i, j) in ind])
time_cost = time.time() - s
print(time_cost)
print('----')

tgt_ids = list(range(20))

out_masks = torch.rand((num_queries,h,w)).cuda()
tgt_masks = torch.rand((20,h,w)).cuda()
s = time.time()
for _ in range(10):
    for idx, label in enumerate(tgt_ids):
        # import pdb;pdb.set_trace()
        out_mask = out_masks[idx*redundant_queries:(idx+1)*redundant_queries]
        tgt_mask = tgt_masks[idx]
        # out_mask = out_mask[None,:]
        tgt_mask = tgt_mask[None,:]

        out_mask = out_mask[:, None]
        tgt_mask = tgt_mask[:, None]

        point_coords = torch.rand(1, num_points, 2, device=out_mask.device)
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
        cost_dice = batch_dice_loss(out_mask, tgt_mask)

        indices = np.argmax(cost_dice.cpu()+cost_dice.cpu())

t = time.time() - s
print(t)



