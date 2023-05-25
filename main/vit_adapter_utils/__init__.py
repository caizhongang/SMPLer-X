from .detection.mmdet_custom.models.backbones.adapter_modules import (
    SpatialPriorModule, MSDeformAttn, DropPath, cp, deform_inputs, ConvFFN
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from torch.nn.init import normal_
# from .detection.ops.modules import MSDeformAttn
from einops import repeat


class ExtractorForSMPLX(nn.Module):
    def __init__(self, dim, task_tokens_num, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # fix for our study
        self.task_tokens_num = task_tokens_num


    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        # fix for our study
        # note: extractor is opposite of injector
        task_token = feat[:, :self.task_tokens_num, :]
        feat = feat[:, self.task_tokens_num:, :]

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InjectorForSMPLX(nn.Module):
    def __init__(self, dim, task_tokens_num, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        # fix for our study
        self.task_tokens_num = task_tokens_num

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)

            return query + self.gamma * attn

        task_token = query[:, :self.task_tokens_num, :]
        query = query[:, self.task_tokens_num:, :]

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        query = torch.cat([task_token, query], dim=1)

        return query


class InteractionBlockForSMPLX(nn.Module):

    def __init__(self, dim, task_tokens_num, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()

        self.injector = InjectorForSMPLX(dim=dim, task_tokens_num=task_tokens_num, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = ExtractorForSMPLX(dim=dim, task_tokens_num=task_tokens_num, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                ExtractorForSMPLX(dim=dim, task_tokens_num=task_tokens_num, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        """ Modified from InteractionBlock for our study """

        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])

        for idx, blk in enumerate(blocks):
            x = blk(x)  # fix for our study, input is x only for ViTPose

        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)

        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c


class ViTAdapterWrapper(nn.Module):
    """ Ref: https://github.com/czczup/ViT-Adapter/blob/main/detection/mmdet_custom/models/backbones/vit_adapter.py
        The original class is not strictly an adapter for efficiency, rather a by-pass branch.
        Here, we follow LoRA-ViT and make it a wrapper.
    """

    def __init__(self, vit_model, pretrain_size=(256, 192), conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=True, use_extra_extractor=True,
                 drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vit_model = vit_model

        # freeze pretrained vit
        for param in self.vit_model.parameters():
            param.requires_grad = False

        # self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.vit_model.blocks)
        self.pretrain_size = pretrain_size
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.norm_layer = norm_layer
        self.drop_path_rate = drop_path_rate

        # fix for our study
        embed_dim = self.vit_model.embed_dim
        task_tokens_num = self.vit_model.task_tokens_num

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane,
                                      embed_dim=embed_dim)
        self.interactions = nn.Sequential(*[
            InteractionBlockForSMPLX(dim=embed_dim, task_tokens_num=task_tokens_num, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor))
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        B = x.shape[0]

        deform_inputs1, deform_inputs2 = deform_inputs(x)

        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, (H, W) = self.vit_model.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.vit_model.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # add task tokens that are specific to our study
        task_tokens = repeat(self.vit_model.task_tokens, '() n d -> b n d', b=B)
        x = torch.cat((task_tokens, x), dim=1)

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.vit_model.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)

        # Fix for our study
        task_tokens = x[:, :self.vit_model.task_tokens_num]  # [N,J,C]
        x = x[:, self.vit_model.task_tokens_num:]  # [N,Hp*Wp,C]

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        # return [f1, f2, f3, f4]

        # Fix for our study
        # f3 has the most proper shape
        xp = f3  # [N,Hp*Wp,C]
        return xp, task_tokens


def apply_adapter(model, model_type):
    assert model_type in ('osx_b', 'osx_l')
    pretrain_size = (256, 192)

    # config ref: https://github.com/czczup/ViT-Adapter/tree/main/detection#results-and-models
    # under mask r-cnn
    if model_type == 'osx_b':
        # https://github.com/czczup/ViT-Adapter/blob/main/detection/configs/mask_rcnn/mask_rcnn_deit_adapter_base_fpn_3x_coco.py
        conv_inplane = 64
        n_points = 4
        deform_num_heads = 12
        cffn_ratio = 0.25
        deform_ratio = 0.5
        interaction_indexes = [[0, 2], [3, 5], [6, 8], [9, 11]]

    elif model_type == 'osx_l':
        # https://github.com/czczup/ViT-Adapter/blob/main/detection/configs/mask_rcnn/mask_rcnn_augreg_adapter_large_fpn_3x_coco.py
        conv_inplane = 64
        n_points = 4
        deform_num_heads = 16
        cffn_ratio = 0.25
        deform_ratio = 0.5
        interaction_indexes = [[0, 5], [6, 11], [12, 17], [18, 23]]

    model = ViTAdapterWrapper(
        model,
        pretrain_size=pretrain_size,
        conv_inplane=conv_inplane,
        n_points=n_points,
        deform_num_heads=deform_num_heads,
        cffn_ratio=cffn_ratio,
        deform_ratio=deform_ratio,
        interaction_indexes=interaction_indexes,
    )

    return model
