# Copyright (c) Facebook, Inc. and its affiliates.
import warnings
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core.memory import retry_if_cuda_oom
from core.data.transforms.post_transforms import pose_pck_accuracy, flip_back, transform_preds
from core.models.ops import box_ops
from .mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from ..losses import loss_entry
from ast import literal_eval

class Norm2d(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps
        self.normalized_shape = (embed_dim,)

        #  >>> workaround for compatability
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln.weight = self.weight
        self.ln.bias = self.bias

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

def _get_activation(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class AIOHead(nn.Module):
    def __init__(self,
                 transformer_predictor_cfg,
                 loss_cfg,
                 num_classes,
                 backbone,  # placeholder
                 neck,  # placeholder
                 loss_weight,
                 ignore_value,
                 ginfo,
                 bn_group,  # placeholder
                 task_sp_list=(),
                 neck_sp_list=(),
                 task='seg',
                 test_cfg=None,
                 predictor='m2f',
                 peddet_mask_dim=False,
                 peddet_mask_forward=True,
                 feature_only=False # redundant param in compliance with past reid test code
                 ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            task_sp_list: specify params/buffers in decoder that should be treated task-specific in reduce_gradients()
            neck_sp_list: specify params/buffers in decoder that should be treated neck-specific in reduce_gradients()
        """
        super().__init__()
        self.task = task
        self.task_sp_list = task_sp_list
        self.neck_sp_list = neck_sp_list

        self.backbone = [backbone]  # avoid recursive specific param register
        self.neck = [neck]  # avoid recursive specific param register

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        if self.task == 'peddet':
            self.peddet_vis_token_dim = self.peddet_embed_dim = backbone.embed_dim
            self.peddet_mask_dim = peddet_mask_dim
            self.peddet_mask_map = nn.Sequential(
                nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
                Norm2d(self.embed_dim),
                _get_activation('gelu'),
                nn.ConvTranspose2d(self.embed_dim, self.mask_dim, kernel_size=2, stride=2),
            ) if peddet_mask_dim else False

            self.peddet_mask_forward = peddet_mask_forward

        if predictor == 'm2f' and self.task == 'peddet':
            self.predictor = MultiScaleMaskedTransformerDecoder(in_channels=self.peddet_vis_token_dim,
                                                                mask_dim=self.peddet_mask_dim,
                                                                mask_classification=True,
                                                                num_classes=num_classes,
                                                                ginfo=ginfo,
                                                                backbone_pose_embed=backbone.pos_embed,
                                                                **transformer_predictor_cfg)

        if 'FSAuxCELoss' not in loss_cfg.type :
            loss_cfg.kwargs.cfg.num_classes = num_classes
            loss_cfg.kwargs.cfg.ignore_value = ignore_value
            loss_cfg.kwargs.cfg.ginfo = ginfo

        self.loss = loss_entry(loss_cfg)

        self.test_cfg = {} if test_cfg is None else test_cfg

        self.num_classes = num_classes


    def prepare_detection_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            new_targets.append(
                {
                    "boxes": targets_per_image.boxes,
                    "labels": targets_per_image.labels,
                    "area": targets_per_image.area,
                    "iscrowd": targets_per_image.iscrowd,
                }
            )
        return new_targets

    def forward(self, features):  # input -> loss, top1, etc.
        # {'image': img_mask, 'label': target_mask, 'filename': img_name, 'backbone_output':xxx, 'neck_output': xxx}
        images = features['image']  # double padded image(s)

        # --> {'pred_logits':'pred_masks':
        #      'aux_outputs':[{'pred_logits':'pred_masks':}, ...]}}
        if self.task == 'peddet':
            if self.peddet_mask_map and self.peddet_mask_forward:
                features.update({'neck_output': {'mask_features': self.peddet_mask_map(features['neck_output']),
                                             'multi_scale_features': [features['neck_output']]}})
            else:
                features.update({'neck_output': {'mask_features': None,
                                                'multi_scale_features': [features['neck_output']]}})

            outputs = self.predictor.forward_peddet(features['neck_output']['multi_scale_features'],
                                                    features['neck_output']['mask_features'])
            if self.training:
                # pedestrain detection target
                assert "instances" in features
                gt_instances = features["instances"]
                targets = self.prepare_detection_targets(gt_instances)

                # bipartite matching-based loss for object detection
                losses = self.loss(outputs, targets)
                for k in losses:
                    if 'loss' in k:
                        losses[k] = losses[k] * self.loss_weight
                return {'loss': losses, 'top1': losses['top1']}  # torch.FloatTensor([0]).cuda()} #losses['top1']}
            else:
                # pedestrain detection target
                processed_results = ped_det_postprocess(outputs, features['orig_size'])
                return processed_results
        else:
            raise NotImplementedError

def ped_det_postprocess(outputs, target_sizes):
    """ Perform the computation
    Parameters:
        outputs: raw outputs of the model
        target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                      For evaluation, this must be the original image size (before any data augmentation)
                      For visualization, this should be the image size after data augment, but before padding
    """
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    prob = out_logits.sigmoid()
    # find the topk predictions
    num = out_logits.view(out_logits.shape[0], -1).shape[1]
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    return results
