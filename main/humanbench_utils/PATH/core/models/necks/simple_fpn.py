import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.ops.utils import ShapeSpec
from core.models.ops.utils import c2_xavier_fill
from ..decoders.mask2former.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder

from core.utils import NestedTensor

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


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def _get_activation(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SimpleFPN(nn.Module):
    def __init__(self,
                 vis_token_dim,
                 mask_dim,
                 backbone,  # placeholder
                 bn_group,
                 pixel_decoder_cfg=None):
        super(SimpleFPN, self).__init__()
        self.embed_dim = backbone.embed_dim
        self.mask_dim = mask_dim
        self.vis_token_dim = vis_token_dim
        self.pixel_decoder_cfg = pixel_decoder_cfg

        fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
            Norm2d(self.embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            Norm2d(self.embed_dim),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, self.vis_token_dim, kernel_size=1, stride=1, padding=0),
            Norm2d(self.vis_token_dim),
        )

        fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            Norm2d(self.embed_dim),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, self.vis_token_dim, kernel_size=1, stride=1, padding=0),
            Norm2d(self.vis_token_dim),
        )

        fpn3 = nn.Sequential(
            #  in compliance with decoder dim request
            nn.Conv2d(self.embed_dim, self.vis_token_dim, kernel_size=1, stride=1, padding=0),
            Norm2d(self.vis_token_dim),
        )

        fpn4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            nn.Conv2d(self.embed_dim, self.vis_token_dim, kernel_size=1, stride=1, padding=0),
            Norm2d(self.vis_token_dim),
        )

        self.fpns = nn.ModuleList([fpn1, fpn2, fpn3, fpn4])

        if self.pixel_decoder_cfg is None:
            self.mask_features = nn.Conv2d(self.vis_token_dim, self.mask_dim, kernel_size=1, stride=1, padding=0)
            c2_xavier_fill(self.mask_features)
        else:
            input_shape = {name: ShapeSpec(channels=self.vis_token_dim, stride=[4, 8, 16, 32][i])
                           for i, name in enumerate(["fpn1", "fpn2", "fpn3", "fpn4"])}
            self.pixel_decoder = MSDeformAttnPixelDecoder(conv_dim=self.vis_token_dim,
                                                          input_shape=input_shape,
                                                          mask_dim=self.mask_dim,
                                                          **pixel_decoder_cfg)

        self.maskformer_num_feature_levels = 3  # always use 3 scales

    def forward(self, features):
        x = features['backbone_output']
        if self.pixel_decoder_cfg is None:
            out = [op(x) for op in self.fpns][::-1]  # [r4, r3, r2, r1]
            num_cur_levels = 0
            multi_scale_features = []
            for o in out:
                if num_cur_levels < self.maskformer_num_feature_levels:
                    multi_scale_features.append(o)
                    num_cur_levels += 1

            features.update({'neck_output': {'mask_features': self.mask_features(out[-1]),
                                             'multi_scale_features': multi_scale_features}})
        else:
            out = {["fpn1", "fpn2", "fpn3", "fpn4"][i]: op(x) for i, op in enumerate(self.fpns)}
            mask_features, multi_scale_features = self.pixel_decoder.forward_features(out)
            features.update({'neck_output': {'mask_features': mask_features,
                                             'multi_scale_features': multi_scale_features}})

        return features


class MoreSimpleFPN(SimpleFPN):
    def __init__(self,
                 vis_token_dim,
                 mask_dim,
                 backbone,  # placeholder
                 bn_group,
                 activation='gelu',
                 task_sp_list=(),
                 maskformer_num_feature_levels=3,
                 pixel_decoder_cfg=None):
        super(SimpleFPN, self).__init__()
        self.task_sp_list = task_sp_list

        self.embed_dim = backbone.embed_dim
        self.mask_dim = mask_dim
        self.vis_token_dim = vis_token_dim
        self.pixel_decoder_cfg = pixel_decoder_cfg

        fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
            Norm2d(self.embed_dim),
            _get_activation(activation),
            nn.ConvTranspose2d(self.embed_dim, self.vis_token_dim, kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            Norm2d(self.vis_token_dim),
        )

        fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.vis_token_dim, kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            Norm2d(self.vis_token_dim),
        )

        fpn3 = nn.Sequential(
            #  in compliance with decoder dim request
            nn.Conv2d(self.embed_dim, self.vis_token_dim, kernel_size=1, stride=1, padding=0),
            Norm2d(self.vis_token_dim),
        )

        fpn4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            nn.Conv2d(self.embed_dim, self.vis_token_dim, kernel_size=1, stride=1, padding=0),
            Norm2d(self.vis_token_dim),
        )

        self.fpns = nn.ModuleList([fpn1, fpn2, fpn3, fpn4])

        if self.pixel_decoder_cfg is None:
            self.mask_features = nn.Conv2d(self.vis_token_dim, self.mask_dim, kernel_size=1, stride=1, padding=0)
            c2_xavier_fill(self.mask_features)
        else:
            input_shape = {name: ShapeSpec(channels=self.vis_token_dim, stride=[4, 8, 16, 32][i])
                           for i, name in enumerate(["fpn1", "fpn2", "fpn3", "fpn4"])}
            self.pixel_decoder = MSDeformAttnPixelDecoder(conv_dim=self.vis_token_dim,
                                                          input_shape=input_shape,
                                                          mask_dim=self.mask_dim,
                                                          **pixel_decoder_cfg)

        self.maskformer_num_feature_levels = maskformer_num_feature_levels  # always use 3 scales


class SimpleNeck(nn.Module):
    def __init__(self,
                 mask_dim,
                 backbone,  # placeholder
                 bn_group,
                 activation='gelu',
                 task_sp_list=(),
                 pixel_decoder_cfg=None,
                 mask_forward=True
                ):
        super(SimpleNeck, self).__init__()
        self.task_sp_list = task_sp_list

        self.vis_token_dim = self.embed_dim = backbone.embed_dim
        self.mask_dim = mask_dim
        self.pixel_decoder_cfg = pixel_decoder_cfg

        self.mask_map = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
            Norm2d(self.embed_dim),
            _get_activation(activation),
            nn.ConvTranspose2d(self.embed_dim, self.mask_dim, kernel_size=2, stride=2),
        ) if mask_dim else False

        self.maskformer_num_feature_levels = 1  # always use 3 scales

        self.mask_forward = mask_forward

    def forward(self, features):
        if self.mask_map and self.mask_forward:
            features.update({'neck_output': {'mask_features': self.mask_map(features['backbone_output']),
                                             'multi_scale_features': [features['backbone_output']]}})
        else:
            features.update({'neck_output': {'mask_features': None,
                                             'multi_scale_features': [features['backbone_output']]}})
        return features


class ShuffleFPN(SimpleFPN):
    def __init__(self,
                 vis_token_dim,
                 mask_dim,
                 backbone,  # placeholder
                 bn_group,
                 s_kernel,
                 pixel_decoder_cfg=None):
        super(SimpleFPN, self).__init__()
        self.embed_dim = backbone.embed_dim
        self.mask_dim = mask_dim
        self.vis_token_dim = vis_token_dim
        self.pixel_decoder_cfg = pixel_decoder_cfg

        fpn1 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.vis_token_dim * 8, kernel_size=s_kernel, stride=1, padding=s_kernel // 2),
            nn.PixelShuffle(2),
            Norm2d(self.vis_token_dim*2),
            nn.GELU(),
            nn.Conv2d(self.vis_token_dim*2, self.vis_token_dim * 4, kernel_size=s_kernel, stride=1, padding=s_kernel // 2),
            nn.PixelShuffle(2),
            Norm2d(self.vis_token_dim),
        )

        fpn2 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.vis_token_dim * 4, kernel_size=s_kernel, stride=1, padding=s_kernel // 2),
            nn.PixelShuffle(2),
            Norm2d(self.vis_token_dim),
        )

        fpn3 = nn.Sequential(
            #  in compliance with decoder dim request
            nn.Conv2d(self.embed_dim, self.vis_token_dim, kernel_size=1, stride=1, padding=0),
            Norm2d(self.vis_token_dim),
        )

        fpn4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            nn.Conv2d(self.embed_dim, self.vis_token_dim, kernel_size=1, stride=1, padding=0),
            Norm2d(self.vis_token_dim),
        )

        self.fpns = nn.ModuleList([fpn1, fpn2, fpn3, fpn4])

        if self.pixel_decoder_cfg is None:
            self.mask_features = nn.Conv2d(self.vis_token_dim, self.mask_dim, kernel_size=1, stride=1, padding=0)
            c2_xavier_fill(self.mask_features)
        else:
            input_shape = {name: ShapeSpec(channels=self.vis_token_dim, stride=[4, 8, 16, 32][i])
                           for i, name in enumerate(["fpn1", "fpn2", "fpn3", "fpn4"])}
            self.pixel_decoder = MSDeformAttnPixelDecoder(conv_dim=self.vis_token_dim,
                                                          input_shape=input_shape,
                                                          mask_dim=self.mask_dim,
                                                          **pixel_decoder_cfg)

        self.maskformer_num_feature_levels = 3  # always use 3 scales


class ShuffleL1FPN(SimpleFPN):
    def __init__(self,
                 vis_token_dim,
                 mask_dim,
                 backbone,  # placeholder
                 bn_group,
                 s_kernel,
                 pixel_decoder_cfg=None):
        super(SimpleFPN, self).__init__()
        self.embed_dim = backbone.embed_dim
        self.mask_dim = mask_dim
        self.vis_token_dim = vis_token_dim
        self.pixel_decoder_cfg = pixel_decoder_cfg

        fpn1 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.vis_token_dim * 16, kernel_size=s_kernel, stride=1, padding=s_kernel // 2),
            nn.PixelShuffle(4),
            Norm2d(self.vis_token_dim),
        )

        fpn2 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.vis_token_dim * 4, kernel_size=s_kernel, stride=1, padding=s_kernel // 2),
            nn.PixelShuffle(2),
            Norm2d(self.vis_token_dim),
        )

        fpn3 = nn.Sequential(
            #  in compliance with decoder dim request
            nn.Conv2d(self.embed_dim, self.vis_token_dim, kernel_size=1, stride=1, padding=0),
            Norm2d(self.vis_token_dim),
        )

        fpn4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            nn.Conv2d(self.embed_dim, self.vis_token_dim, kernel_size=1, stride=1, padding=0),
            Norm2d(self.vis_token_dim),
        )

        self.fpns = nn.ModuleList([fpn1, fpn2, fpn3, fpn4])

        if self.pixel_decoder_cfg is None:
            self.mask_features = nn.Conv2d(self.vis_token_dim, self.mask_dim, kernel_size=1, stride=1, padding=0)
            c2_xavier_fill(self.mask_features)
        else:
            input_shape = {name: ShapeSpec(channels=self.vis_token_dim, stride=[4, 8, 16, 32][i])
                           for i, name in enumerate(["fpn1", "fpn2", "fpn3", "fpn4"])}
            self.pixel_decoder = MSDeformAttnPixelDecoder(conv_dim=self.vis_token_dim,
                                                          input_shape=input_shape,
                                                          mask_dim=self.mask_dim,
                                                          **pixel_decoder_cfg)

        self.maskformer_num_feature_levels = 3  # always use 3 scales


class SimpleMFFPN(SimpleFPN):
    def __init__(self,
                 vis_token_dim,
                 mask_dim,
                 backbone,  # placeholder
                 bn_group,
                 pixel_decoder_cfg=None):
        super(SimpleFPN, self).__init__()
        self.embed_dim = backbone.embed_dim
        self.mask_dim = mask_dim
        self.vis_token_dim = vis_token_dim
        self.pixel_decoder_cfg = pixel_decoder_cfg

        fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.mask_dim, kernel_size=2, stride=2),
            Norm2d(self.mask_dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.mask_dim, self.mask_dim, kernel_size=2, stride=2),
        )

        fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.vis_token_dim, kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            Norm2d(self.vis_token_dim),
        )

        fpn3 = nn.Sequential(
            #  in compliance with decoder dim request
            nn.Conv2d(self.embed_dim, self.vis_token_dim, kernel_size=1, stride=1, padding=0),
            Norm2d(self.vis_token_dim),
        )

        fpn4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            nn.Conv2d(self.embed_dim, self.vis_token_dim, kernel_size=1, stride=1, padding=0),
            Norm2d(self.vis_token_dim),
        )

        self.fpns = nn.ModuleList([fpn2, fpn3, fpn4])

        if self.pixel_decoder_cfg is None:
            self.mask_features = fpn1
        else:
            raise

        self.maskformer_num_feature_levels = 3  # always use 3 scales

    def forward(self, features):
        x = features['backbone_output']

        if self.pixel_decoder_cfg is None:
            out = [op(x) for op in self.fpns][::-1]  # [r4, r3, r2]
            multi_scale_features = []
            for num_cur_levels, o in enumerate(out):
                if num_cur_levels < self.maskformer_num_feature_levels:
                    multi_scale_features.append(o)

            features.update({'neck_output': {'mask_features': self.mask_features(x),
                                             'multi_scale_features': multi_scale_features}})
        else:
            raise NotImplementedError

        return features


class MoreSimpleFPNwoNorm(SimpleFPN):
    def __init__(self,
                 vis_token_dim,
                 mask_dim,
                 backbone,  # placeholder
                 bn_group,
                 pixel_decoder_cfg=None):
        super(SimpleFPN, self).__init__()
        self.embed_dim = backbone.embed_dim
        self.mask_dim = mask_dim
        self.vis_token_dim = vis_token_dim
        self.pixel_decoder_cfg = pixel_decoder_cfg

        fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
            Norm2d(self.embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.embed_dim, self.vis_token_dim, kernel_size=2, stride=2),
            #  in compliance with decoder dim request
        )

        fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.vis_token_dim, kernel_size=2, stride=2),
            #  in compliance with decoder dim request
        )

        fpn3 = nn.Sequential(
            #  in compliance with decoder dim request
            nn.Conv2d(self.embed_dim, self.vis_token_dim, kernel_size=1, stride=1, padding=0),
        )

        fpn4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            nn.Conv2d(self.embed_dim, self.vis_token_dim, kernel_size=1, stride=1, padding=0),
        )

        self.fpns = nn.ModuleList([fpn1, fpn2, fpn3, fpn4])

        if self.pixel_decoder_cfg is None:
            self.mask_features = nn.Conv2d(self.vis_token_dim, self.mask_dim, kernel_size=1, stride=1, padding=0)
            c2_xavier_fill(self.mask_features)
        else:
            input_shape = {name: ShapeSpec(channels=self.vis_token_dim, stride=[4, 8, 16, 32][i])
                           for i, name in enumerate(["fpn1", "fpn2", "fpn3", "fpn4"])}
            self.pixel_decoder = MSDeformAttnPixelDecoder(conv_dim=self.vis_token_dim,
                                                          input_shape=input_shape,
                                                          mask_dim=self.mask_dim,
                                                          **pixel_decoder_cfg)

        self.maskformer_num_feature_levels = 3  # always use 3 scales


class PoseSimpleFPN(nn.Module):
    def __init__(self,
                 vis_token_dim,
                 backbone,  # placeholder
                 bn_group,):
        super(PoseSimpleFPN, self).__init__()
        self.embed_dim = backbone.embed_dim
        self.vis_token_dim = vis_token_dim

        fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
            Norm2d(self.embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.embed_dim, self.vis_token_dim, kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            Norm2d(self.vis_token_dim),
        )

        self.fpns = nn.ModuleList([fpn1])

    def forward(self, features):
        x = features['backbone_output']

        out = [op(x) for op in self.fpns][::-1]  # [r4, r3, r2, r1]
        multi_scale_features = []
        for o in out:
            multi_scale_features.append(o)

        features.update({'neck_output': multi_scale_features})

        return features


class PedDetSimpleResNetFPN(nn.Module):
    def __init__(self,
                 num_feature_levels,
                 hidden_dim,
                 backbone, # placeholder
                 bn_group,
                 **kwargs):
        super(PedDetSimpleResNetFPN, self).__init__()
        num_backbone_outs = len(backbone.strides)
        self.backbone = [backbone]
        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ])

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, input_var):
        features = input_var['backbone_output']['Join']['tensor']
        pos = input_var['backbone_output']['Join']['position_embedding']

        srcs = []
        masks = []

        if self.num_feature_levels == 1:
            src, mask = features[-1].decompose()
            srcs.append(self.input_proj[0](src))
            masks.append(mask)
            assert mask is not None
        else:
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                srcs.append(self.input_proj[l](src))
                masks.append(mask)
                assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = input_var['NestedImage'].mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[0][1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        neck_output = {}
        neck_output = {'multi_scale_features': srcs, 'multi_scale_masks': masks, 'multi_scale_pos': pos}
        input_var.update({'neck_output': neck_output})

        return input_var


class PedDetMoreSimpleFPN(SimpleFPN):
    def __init__(self,
                 vis_token_dim,
                 mask_dim,
                 num_feature_levels,
                 backbone,  # placeholder
                 bn_group,
                 activation='gelu',
                 pixel_decoder_cfg=None):
        super(SimpleFPN, self).__init__()
        self.embed_dim = backbone.embed_dim
        self.mask_dim = mask_dim
        self.vis_token_dim = vis_token_dim
        self.pixel_decoder_cfg = pixel_decoder_cfg
        self.backbone = [backbone]

        fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
            Norm2d(self.embed_dim),
            _get_activation(activation),
            nn.ConvTranspose2d(self.embed_dim, self.vis_token_dim, kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            Norm2d(self.vis_token_dim),
        )

        fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.vis_token_dim, kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            Norm2d(self.vis_token_dim),
        )

        fpn3 = nn.Sequential(
            #  in compliance with decoder dim request
            nn.Conv2d(self.embed_dim, self.vis_token_dim, kernel_size=1, stride=1, padding=0),
            Norm2d(self.vis_token_dim),
        )

        fpn4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            nn.Conv2d(self.embed_dim, self.vis_token_dim, kernel_size=1, stride=1, padding=0),
            Norm2d(self.vis_token_dim),
        )

        self.fpns = nn.ModuleList([fpn1, fpn2, fpn3, fpn4])

        if self.pixel_decoder_cfg is None:
            self.mask_features = nn.Conv2d(self.vis_token_dim, self.mask_dim, kernel_size=1, stride=1, padding=0)
            c2_xavier_fill(self.mask_features)
        else:
            input_shape = {name: ShapeSpec(channels=self.vis_token_dim, stride=[4, 8, 16, 32][i])
                           for i, name in enumerate(["fpn1", "fpn2", "fpn3", "fpn4"])}
            self.pixel_decoder = MSDeformAttnPixelDecoder(conv_dim=self.vis_token_dim,
                                                          input_shape=input_shape,
                                                          mask_dim=self.mask_dim,
                                                          **pixel_decoder_cfg)

        self.maskformer_num_feature_levels = num_feature_levels  # always use 3 scales

    def forward(self, features):
        x, m = features['backbone_output'].decompose()
        backbone = self.backbone[0]
        Hp, Wp = x.shape[-2:]
        pos = backbone.pos_embed.reshape(1, backbone.patch_embed.patch_shape[0],
                                         backbone.patch_embed.patch_shape[1],
                                         backbone.pos_embed.size(2))[:, :Hp, :Wp, :].permute(0, 3, 1, 2)

        if self.pixel_decoder_cfg is None:
            out = [op(x) for op in self.fpns][::-1]  # [r4, r3, r2, r1]
            num_cur_levels = 0
            multi_scale_features = []
            multi_scale_masks = []
            multi_scale_poss = []
            for o in out:
                if num_cur_levels < self.maskformer_num_feature_levels:
                    mask_o = F.interpolate(m[None].float(), size=o.shape[-2:]).to(torch.bool)[0]
                    multi_scale_features.append(o)
                    multi_scale_masks.append(mask_o)
                    pos_l = F.interpolate(pos[None], size=o.shape[-3:], mode='trilinear', align_corners=False)[0]
                    multi_scale_poss.append(pos_l)
                    num_cur_levels += 1

            features.update({'neck_output': {'mask_features': self.mask_features(out[-1]),
                                             'multi_scale_features': multi_scale_features,
                                             'multi_scale_masks': multi_scale_masks,
                                             'multi_scale_pos': multi_scale_poss}})
        else:
            raise NotImplementedError
        return features


class PedDetAlignedFPN(SimpleFPN):
    def __init__(self,
                 vis_token_dim,
                 mask_dim,
                 num_feature_levels,
                 backbone,  # placeholder
                 bn_group,
                 activation='gelu',
                 pixel_decoder_cfg=None):
        super(SimpleFPN, self).__init__()
        self.embed_dim = backbone.embed_dim
        self.vis_token_dim = vis_token_dim
        self.pixel_decoder_cfg = pixel_decoder_cfg
        self.backbone = [backbone]
        self.pos_mode = backbone.test_pos_mode

        fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim // 2, kernel_size=2, stride=2),
            Norm2d(self.embed_dim // 2),
            _get_activation(activation),
            nn.ConvTranspose2d(self.embed_dim // 2, self.embed_dim // 4, kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            Conv2d(self.embed_dim // 4, self.vis_token_dim,
                   kernel_size=1, bias=False, norm=Norm2d(self.vis_token_dim)),
            Conv2d(self.vis_token_dim, self.vis_token_dim,
                   kernel_size=3, padding=1, bias=False, norm=Norm2d(self.vis_token_dim)),
        )

        fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim // 2, kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            Conv2d(self.embed_dim // 2, self.vis_token_dim,
                   kernel_size=1, bias=False, norm=Norm2d(self.vis_token_dim)),
            Conv2d(self.vis_token_dim, self.vis_token_dim,
                   kernel_size=3, padding=1, bias=False, norm=Norm2d(self.vis_token_dim)),
        )

        fpn3 = nn.Sequential(
            #  in compliance with decoder dim request
            Conv2d(self.embed_dim, self.vis_token_dim,
                   kernel_size=1, bias=False, norm=Norm2d(self.vis_token_dim)),
            Conv2d(self.vis_token_dim, self.vis_token_dim,
                   kernel_size=3, padding=1, bias=False, norm=Norm2d(self.vis_token_dim)),
        )

        fpn4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            #  in compliance with decoder dim request
            Conv2d(self.embed_dim, self.vis_token_dim,
                   kernel_size=1, bias=False, norm=Norm2d(self.vis_token_dim)),
            Conv2d(self.vis_token_dim, self.vis_token_dim,
                   kernel_size=3, padding=1, bias=False, norm=Norm2d(self.vis_token_dim)),
        )

        self.fpns = nn.ModuleList([fpn1, fpn2, fpn3, fpn4])

        assert self.pixel_decoder_cfg is None
        self.maskformer_num_feature_levels = num_feature_levels  # always use 3 scales

    def forward(self, features):
        x, m = features['backbone_output'].decompose()
        backbone = self.backbone[0]
        Hp, Wp = x.shape[-2:]
        if self.pos_mode == 'simple_interpolate':
            pos = backbone.pos_embed.reshape(1, backbone.patch_embed.patch_shape[0],
                                             backbone.patch_embed.patch_shape[1],
                                             backbone.pos_embed.size(2)).permute(0, 3, 1, 2)
        else:
            pos = backbone.pos_embed.reshape(1, backbone.patch_embed.patch_shape[0],
                                             backbone.patch_embed.patch_shape[1],
                                             backbone.pos_embed.size(2))[:, :Hp, :Wp, :].permute(0, 3, 1, 2)

        out = [op(x) for op in self.fpns][::-1]  # [r4, r3, r2, r1]
        num_cur_levels = 0
        multi_scale_features = []
        multi_scale_masks = []
        multi_scale_poss = []
        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                mask_o = F.interpolate(m[None].float(), size=o.shape[-2:]).to(torch.bool)[0]
                multi_scale_features.append(o)
                multi_scale_masks.append(mask_o)
                pos_l = F.interpolate(pos[None], size=o.shape[-3:], mode='trilinear', align_corners=False)[0]
                multi_scale_poss.append(pos_l)
                num_cur_levels += 1

        features.update({'neck_output': {'mask_features': None,
                                         'multi_scale_features': multi_scale_features,
                                         'multi_scale_masks': multi_scale_masks,
                                         'multi_scale_pos': multi_scale_poss}})
        return features
