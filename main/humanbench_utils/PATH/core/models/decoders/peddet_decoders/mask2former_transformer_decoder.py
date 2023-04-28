# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
from core.models.ops.utils import c2_xavier_fill
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from core.models.ops.utils import Conv2d

from .position_encoding import PositionEmbeddingSine
from ...backbones.vitdet import get_2d_sincos_pos_embed
from ...ckpt import checkpoint_wrapper
import math


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", arch=False, net_depth=9):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.arch = arch
        self.net_depth = net_depth

        self._reset_parameters()

    def _reset_parameters(self):
        if self.arch == 'deepnorm':
            for param_name, p in self.named_parameters():
                if p.dim() > 1:
                    if 'v_proj' in param_name or 'out_proj' in param_name:
                        nn.init.xavier_normal_(p, gain=(12 * self.net_depth) ** (- 0.25))
                    elif 'q_proj' in param_name or 'k_proj' in param_name:
                        nn.init.xavier_normal_(p, gain=1)
                    else:
                        nn.init.xavier_uniform_(p)
        elif self.arch == 'fan_in':
            for p in self.parameters():
                if p.dim() > 1:
                    assert p.dim() == 2
                    fan_in = p.size(1)
                    std = 1 / math.sqrt(fan_in)
                    with torch.no_grad():
                        p.normal_(0, std)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_post_deep(self, tgt,
                         tgt_mask: Optional[Tensor] = None,
                         tgt_key_padding_mask: Optional[Tensor] = None,
                         query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt * (3 * self.net_depth) ** 0.25 + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.arch == 'pre_norm':
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        elif self.arch == 'deepnorm':
            return self.forward_post_deep(tgt, tgt_mask,
                                          tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", arch=False, net_depth=9):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.arch = arch
        self.net_depth = net_depth

        self._reset_parameters()

    def _reset_parameters(self):
        if self.arch == 'deepnorm':
            for param_name, p in self.named_parameters():
                if p.dim() > 1:
                    if 'v_proj' in param_name or 'out_proj' in param_name:
                        nn.init.xavier_normal_(p, gain=(12 * self.net_depth) ** (- 0.25))
                    elif 'q_proj' in param_name or 'k_proj' in param_name:
                        nn.init.xavier_normal_(p, gain=1)
                    else:
                        nn.init.xavier_uniform_(p)
        elif self.arch == 'fan_in':
            for p in self.parameters():
                if p.dim() > 1:
                    assert p.dim() == 2
                    fan_in = p.size(1)
                    std = 1 / math.sqrt(fan_in)
                    with torch.no_grad():
                        p.normal_(0, std)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.arch == 'pre_norm':
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        elif self.arch == 'deepnorm':
            raise NotImplementedError
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", arch=False, net_depth=9):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.arch = arch
        self.net_depth = net_depth

        self._reset_parameters()

    def _reset_parameters(self):
        if self.arch == 'deepnorm':
            for param_name, p in self.named_parameters():
                if p.dim() > 1:
                    if 'linear' in param_name:
                        nn.init.xavier_normal_(p, gain=(12 * self.net_depth) ** (- 0.25))
                    else:
                        nn.init.xavier_uniform_(p)
        elif self.arch == 'fan_in':
            for p in self.parameters():
                if p.dim() > 1:
                    assert p.dim() == 2
                    fan_in = p.size(1)
                    std = 1 / math.sqrt(fan_in)
                    with torch.no_grad():
                        p.normal_(0, std)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_post_deep(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt * (3 * self.net_depth) ** 0.25 + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.arch == 'pre_norm':  # false
            return self.forward_pre(tgt)
        elif self.arch == 'deepnorm':
            return self.forward_post_deep(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def custom_replace(tensor,on_neg_1,on_zero,on_one):
    res = tensor.clone()
    res[tensor==-1] = on_neg_1
    res[tensor==0] = on_zero
    res[tensor==1] = on_one
    return res


def weights_init(module):
    """ Initialize the weights, copy from CTran"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        stdv = 1. / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class MultiScaleMaskedTransformerDecoder(nn.Module):
    _version = 2

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    def __init__(self,
                 in_channels,
                 mask_classification,  # True
                 num_classes: int,
                 hidden_dim: int,
                 num_queries: int,
                 nheads: int,
                 dim_feedforward: int,
                 *,

                 dec_layers: int,
                 pre_norm: bool,  # False
                 mask_dim: int,
                 enforce_input_project: bool,  # False
                 mask_on=True,
                 num_feature_levels=3,
                 reid_cfgs=None,
                 pedattr_cfgs=None,
                 peddet_cfgs=None,
                 cfgs=None,
                 ginfo=None,
                 arch=False,
                 cross_pos_embed="sincos",  # use pos embed in cross attention layers
                 backbone_pose_embed=None,
                 cls_out_dim=None
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        assert mask_classification, "Only support mask classification model, duplicated option"
        self.mask_on = mask_on
        self.cross_pos_embed = cross_pos_embed
        self.backbone_pose_embed = [backbone_pose_embed]

        self.reid_cfgs = {} if reid_cfgs is None else reid_cfgs
        self.pedattr_cfgs = {} if pedattr_cfgs is None else pedattr_cfgs
        self.peddet_cfgs = {} if peddet_cfgs is None else peddet_cfgs
        self.cfgs = {} if cfgs is None else cfgs

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.hidden_dim = hidden_dim
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                checkpoint_wrapper(SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    arch='pre_norm' if pre_norm else arch,
                ))
            )

            self.transformer_cross_attention_layers.append(
                checkpoint_wrapper(CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    arch='pre_norm' if pre_norm else arch,
                ))
            )

            self.transformer_ffn_layers.append(
                checkpoint_wrapper(FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    arch='pre_norm' if pre_norm else arch,
                ))
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim) if self.pedattr_cfgs.get('head_nrom_type', False) != 'post' else nn.LayerNorm(hidden_dim*num_queries)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(self.peddet_cfgs.get('share_content_query', num_queries),
                                       hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries // self.peddet_cfgs.get('share_content_query', 1), self.peddet_cfgs.get('query_pe_dim', hidden_dim))

        # level embedding (originally 3 scales)
        self.num_feature_levels = num_feature_levels
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.reid_cfgs and self.reid_cfgs.head_type == 'cat_proj' or self.pedattr_cfgs and self.pedattr_cfgs.head_type == 'cat_proj':
            self.class_embed = nn.Linear(hidden_dim * num_queries, num_classes + 1 if cls_out_dim is None else cls_out_dim)
        else:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1 if cls_out_dim is None else cls_out_dim)

        self.mask_embed = MLP(hidden_dim,
                              hidden_dim,
                              mask_dim,
                              3) if mask_dim and self.cfgs.get('mask_head_type', 'default') == 'default' else None
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) if peddet_cfgs is not None else None
        self.adapt_pos2d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ) if self.cross_pos_embed == 'anchor' else None

        self.pos_embed = None  # valid when cross_pos_embed == 'pos_prior'
        self._reset_parameters()


    def _reset_parameters(self):
        if self.cross_pos_embed == 'pos_prior':
            resolution = self.peddet_cfgs.get('pos_prior_resolution', 224)
            pos_embed = get_2d_sincos_pos_embed(self.hidden_dim, resolution, cls_token=False)  # HW x C
            self.pos_embed = nn.Parameter(torch.zeros(1, self.hidden_dim, resolution, resolution),
                                          requires_grad=self.peddet_cfgs.get('pos_prior_embed_update', True))  # BxCxHxW
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().reshape(1, resolution, resolution,
                                                                                  self.hidden_dim).permute(0, 3, 1, 2))

        if self.adapt_pos2d is not None:
            nn.init.uniform_(self.query_embed.weight.data, 0, 1)

    def get_vis_token_pos_embed(self, shape=None):
        if not self.cross_pos_embed:
            return None
        elif self.cross_pos_embed == 'pos_prior':
            pos = F.interpolate(self.pos_embed, size=shape, mode='bicubic',
                                align_corners=False).permute(2, 3, 0, 1).flatten(0, 1)
            return pos
        elif self.cross_pos_embed == 'shared':
            if shape is not None and self.peddet_cfgs != {}:
                #  assume squared initial pose_embed if interpolation is needed and in simple_interpolate mode
                H = W = int(math.sqrt(self.backbone_pose_embed[0].size(1)))
                H_n, W_n = shape
                init_pe = self.backbone_pose_embed[0].reshape(1, H, W, -1)
                pos = F.interpolate(init_pe.permute(0, 3, 1, 2)[None],
                                    size=(self.hidden_dim, H_n, W_n), mode='trilinear', align_corners=False)[0].permute(2, 3, 0, 1).flatten(0, 1)
                return pos
            if len(self.input_proj) > 0:
                pos = self.input_proj[0](self.backbone_pose_embed[0].permute(0, 2, 1).unsqueeze(3)) # b c hw 1
                return pos.squeeze(3).permute(2, 0, 1)  # -> HWxBxC
            return self.backbone_pose_embed[0].permute(1, 0, 2)  # BxHWxC -> HWxBxC
        elif self.cross_pos_embed == 'anchor':
            assert shape is not None
            mask = torch.zeros(1, *shape, dtype=torch.bool).cuda()
            H_n, W_n = shape
            pos_col, pos_row = mask2pos(mask)  # (1xh, 1xw) workaround to utilize existing codebase
            pos_2d = torch.cat([pos_row.unsqueeze(1).repeat(1, H_n, 1).unsqueeze(-1),
                                pos_col.unsqueeze(2).repeat(1, 1, W_n).unsqueeze(-1)], dim=-1)  # 1xhxwx2
            posemb_2d = self.adapt_pos2d(pos2posemb2d(pos_2d, self.hidden_dim // 2))  # 1xhxwxc
            return posemb_2d.flatten(1,2).permute(1, 0, 2)  # BxHWxC -> HWxBxC
        elif self.cross_pos_embed == 'shared_inter':
            return F.interpolate(self.backbone_pose_embed[0],
                                 size=self.hidden_dim, mode='linear', align_corners=False).permute(1, 0, 2)
        else:
            raise NotImplementedError(f"unknown self.cross_pos_embed: {self.cross_pos_embed}")

    def forward(self, x, mask_features, mask_label=None):
        # x is a list of multi-scale feature [r5, r4, r3], mask_features: r2
        assert len(x) == self.num_feature_levels
        src = []
        pos = []  # sin-cos embed supporting multi-scale feature map, used only if self.cross_pos_embed == "sincos"
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))  # pos embedding
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])  # lv_emb + x

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # >>> ** initial proposal ** prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                               attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # >>> ** cross-attn **  attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask if self.mask_on else None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index] if self.cross_pos_embed == "sincos" else self.get_vis_token_pos_embed(size_list[level_index]),
                query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output,
                                                                                   mask_features,
                                                                                   attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                outputs_class=predictions_class,
                outputs_seg_masks=predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output).squeeze()
        if self.cfgs.get('mask_head_type', 'default') == 'direct':
            mask_embed = output.transpose(0, 1)
        elif self.cfgs.get('mask_head_type', 'default') == 'norm_direct':
            mask_embed = decoder_output
        else:
            mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                         1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    def _set_aux_loss(self, outputs_class=None,
                      outputs_seg_masks=None,
                      outputs_mask=None,
                      outputs_det_bboxes=None,
                      task='seg'):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if task == 'reid':
            return [
                {"feature": a, "feature_nobn": b}
                for a, b in zip(outputs_class[0][:-1], outputs_class[1][:-1])
            ]
        elif task == 'ssl':
            return [{"feature": a} for a in outputs_class[:-1]]
        elif task == 'attr':
            return [{"logit": a} for a in outputs_class[:-1]]
        elif task == 'peddet':
            return [
                {"pred_logits": a, "mask": b, "pred_boxes": c}
                for a, b, c in zip(outputs_class[:-1],
                                   outputs_mask[:-1],
                                   outputs_det_bboxes[:-1])
            ]
        elif task == 'peddet_mask':
            return [
                {"pred_logits": a, "mask": b, "pred_masks": c}
                for a, b, c in zip(outputs_class[:-1],
                                   outputs_mask[:-1],
                                   outputs_seg_masks[:-1])
            ]
        else:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]

    def forward_reid(self, x, norm):
        # x is a list of multi-scale feature [r5, r4, r3], mask_features: r2
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))  # pos embedding
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])  # lv_emb + x

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_class_nobn = []

        # >>> ** initial proposal ** prediction heads on learnable query features
        # >>> disabled in reid tasks
        # outputs_class, outputs_class_nobn = self.forward_reid_prediction_heads(output, norm)
        # predictions_class.append(outputs_class)
        # predictions_class_nobn.append(outputs_class_nobn)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels

            # >>> ** cross-attn **  attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index] if self.cross_pos_embed == "sincos" else self.get_vis_token_pos_embed(size_list[level_index]),
                query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_class_nobn = self.forward_reid_prediction_heads(output, norm)
            predictions_class.append(outputs_class)
            predictions_class_nobn.append(outputs_class_nobn)

        assert len(predictions_class) == self.num_layers # + 1

        out = {
            'feature': predictions_class[-1],
            'feature_nobn': predictions_class_nobn[-1],
            'aux_outputs': self._set_aux_loss(outputs_class=(predictions_class, predictions_class_nobn),
                                              task='reid')
        }
        return out

    def forward_reid_prediction_heads(self, output, norm):
        if self.reid_cfgs.get('head_type', 'proj_pool') == 'proj_pool':
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)   # [Q, B, C] -> [B, Q, C]
            decoder_output = self.class_embed(decoder_output).mean(dim=1)   # [B, Q, C] -> [B, C_out]
            return norm(decoder_output) , decoder_output
        elif self.reid_cfgs.head_type == 'direct':
            # decoder_output = self.decoder_norm(output)
            decoder_output = output.transpose(0, 1).mean(dim=1)   # [Q, B, C] -> [B, Q, C] -> [B, C]
            return norm(decoder_output) , decoder_output
        elif self.reid_cfgs.head_type == 'cat':
            # decoder_output = self.decoder_norm(output)
            decoder_output = output.transpose(0, 1).flatten(1)   # [Q, B, C] -> [B, Q, C] -> [B, QxC]
            return norm(decoder_output) , decoder_output
        elif self.reid_cfgs.head_type == 'pool':
            decoder_output = output.transpose(0, 1).mean(dim=1)  # [Q, B, C] -> [B, Q, C] -> [B, C_out]
            return norm(decoder_output) , decoder_output
        elif self.reid_cfgs.head_type == 'pool_proj':
            decoder_output = output.transpose(0, 1).mean(dim=1)   # [Q, B, C] -> [B, Q, C] -> [B, C]
            decoder_output = self.decoder_norm(decoder_output)
            decoder_output = self.class_embed(decoder_output)   # [B, C] -> [B, C_out]
            return norm(decoder_output) , decoder_output
        elif self.reid_cfgs.head_type == 'proj_cat':
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)   # [Q, B, C] -> [B, Q, C]
            decoder_output = self.class_embed(decoder_output).squeeze()   # [B, Q, C] -> [B, Q]
            return norm(decoder_output) , decoder_output
        elif self.reid_cfgs.head_type == 'proj_cat_mul':
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)   # [Q, B, C] -> [B, Q, C]
            decoder_output = self.class_embed(decoder_output).flatten(1)   # [B, Q, C] -> [B, Q]
            return norm(decoder_output) , decoder_output
        elif self.reid_cfgs.head_type == 'cat_proj':
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1).flatten(1)   # [Q, B, C] -> [B, Q, C] -> [B, QxC]
            decoder_output = self.class_embed(decoder_output)   # [B, QxC] -> [B, C_out]
            return norm(decoder_output) , decoder_output

    def forward_attr(self, x):
        # x is a list of multi-scale feature [r5, r4, r3], mask_features: r2
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))  # pos embedding
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])  # lv_emb + x

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels

            # >>> ** cross-attn **  attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index] if self.cross_pos_embed == "sincos" else self.get_vis_token_pos_embed(size_list[level_index]),
                query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class = self.forward_attr_prediction_heads(output)
            predictions_class.append(outputs_class)

        assert len(predictions_class) == self.num_layers # + 1

        out = {
            'logit': predictions_class[-1],
            'aux_outputs': self._set_aux_loss(outputs_class=predictions_class,
                                              task='attr')
        }
        return out

    def forward_attr_prediction_heads(self, output):
        if self.pedattr_cfgs.head_type == 'per_q':
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)   # [Q, B, C] -> [B, Q, C]
            decoder_output = self.class_embed(decoder_output).squeeze()   # [B, Q(class), 1] -> [B, Q(class)]
            return decoder_output
        elif self.pedattr_cfgs.head_type == 'cat_proj':
            if self.pedattr_cfgs.head_nrom_type == 'pre':
                decoder_output = self.decoder_norm(output)
                decoder_output = decoder_output.transpose(0, 1).flatten(1)   # [Q, B, C] -> [B, Q, C] -> [B, QxC]
            # elif self.pedattr_cfgs.head_nrom_type == 'post':
            #     raise
            else:
                raise
            decoder_output = self.class_embed(decoder_output)   # [B, QxC] -> [B, class]
            return decoder_output
        elif self.pedattr_cfgs.head_type == 'pool_proj':
            decoder_output = output.transpose(0, 1).mean(dim=1)   # [Q, B, C] -> [B, Q, C] -> [B, C]
            decoder_output = self.decoder_norm(decoder_output)
            decoder_output = self.class_embed(decoder_output)   # [B, C] -> [B, class]
            return decoder_output
        elif self.pedattr_cfgs.head_type == 'proj_pool':
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)   # [Q, B, C] -> [B, Q, C]
            decoder_output = self.class_embed(decoder_output).mean(dim=1).squeeze()   # [B, Q, C] -> [B, Q, class] -> [B, class]
            return decoder_output

    def forward_ssl(self, x):
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))  # pos embedding
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])  # lv_emb + x

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels

            # >>> ** cross-attn **  attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index] if self.cross_pos_embed == "sincos" else self.get_vis_token_pos_embed(size_list[level_index]),
                query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class = self.forward_ssl_prediction_heads(output)
            predictions_class.append(outputs_class)

        assert len(predictions_class) == self.num_layers # + 1

        out = {
            'feature': predictions_class[-1],
            'aux_outputs': self._set_aux_loss(outputs_class=predictions_class,
                                              task='ssl')
        }
        return out

    def forward_ssl_prediction_heads(self, output):
        decoder_output = output.transpose(0, 1).flatten(1)   # [Q, B, C] -> [B, Q, C] -> [B, QxC]
        return decoder_output

    def forward_peddet(self, x, mask_features):
        # x is a list of multi-scale feature [r5, r4, r3], mask_features: r2
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))  # pos embedding
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])  # lv_emb + x

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxBxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(self.peddet_cfgs.get('share_content_query', 1), bs, 1)
        output = self.query_feat.weight.unsqueeze(1).unsqueeze(1).repeat(1,
                                                                         self.query_embed.weight.size(0) if self.peddet_cfgs.get('share_content_query', False) else 1,
                                                                         bs,
                                                                         1).reshape(-1, bs, self.hidden_dim)  # QxC -> Qx1x1xC -> (Qxq)xBxC

        predictions_mask = []
        predictions_class = []
        predictions_bbox = []

        predictions_class_one2many = []  # for one2many hybrid loss only
        predictions_bbox_one2many = []

        cur_all_mask = 1
        seed_mask = None

        if self.peddet_cfgs.get('one2many', False):

            self_attn_mask = (
                torch.zeros([self.num_queries, self.num_queries, ]).bool().cuda()  # QxQ
            )
            self_attn_mask[self.peddet_cfgs.num_queries_one2one:, 0: self.peddet_cfgs.num_queries_one2one, ] = True
            self_attn_mask[0: self.peddet_cfgs.num_queries_one2one, self.peddet_cfgs.num_queries_one2one:, ] = True
        else:
            self_attn_mask = None
        # >>> ** initial proposal ** prediction heads on learnable query features
        # outputs_class, outputs_bbox = self.forward_peddet_prediction_heads(output)
        # predictions_class.append(outputs_class)
        # predictions_mask.append(None)
        # predictions_bbox.append(outputs_bbox)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels

            assert not self.mask_on
            if self.cross_pos_embed == 'pos_prior': # sample pos embed
                if self.peddet_cfgs.get('fix_range_bug', False):
                    _query_embed = F.grid_sample(self.pos_embed, self.query_embed.weight.unsqueeze(0).unsqueeze(0)*2-1,
                                                 mode='bicubic', padding_mode='border', align_corners=False)  # 1xCx1xQ
                else:
                    _query_embed = F.grid_sample(self.pos_embed, self.query_embed.weight.unsqueeze(0).unsqueeze(0), mode='bicubic', padding_mode='border', align_corners=False) # 1xCx1xQ
                _query_embed = _query_embed[0].permute(2, 1, 0).repeat(self.peddet_cfgs.get('share_content_query', 1), bs, 1)  # q x b x c
            else:
                _query_embed = self.adapt_pos2d(pos2posemb2d(query_embed, self.hidden_dim // 2))  # q x b x c
            reference = inverse_sigmoid(query_embed)

            # >>> ** cross-attn **  attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output * cur_all_mask,  # QxBxC
                src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index] if self.cross_pos_embed == "sincos" else self.get_vis_token_pos_embed(size_list[level_index]),
                query_pos=_query_embed  # QxBxC
            ) * cur_all_mask
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=self_attn_mask,
                tgt_key_padding_mask=None,
                query_pos=_query_embed
            ) * cur_all_mask

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_bbox = self.forward_peddet_prediction_heads(output, reference)

            if self.peddet_cfgs.get('one2many', False):
                predictions_mask.append(None)
                predictions_class.append(outputs_class[:, 0:self.peddet_cfgs.num_queries_one2one])
                predictions_bbox.append(outputs_bbox[:, 0:self.peddet_cfgs.num_queries_one2one])

                predictions_class_one2many.append(outputs_class[:, self.peddet_cfgs.num_queries_one2one:])
                predictions_bbox_one2many.append(outputs_bbox[:, self.peddet_cfgs.num_queries_one2one:])
            else:
                predictions_mask.append(None)
                predictions_class.append(outputs_class)
                predictions_bbox.append(outputs_bbox)

                predictions_class_one2many.append(None)
                predictions_bbox_one2many.append(None)

            if self.peddet_cfgs.get('refine_box', False):
                assert self.cross_pos_embed != 'pos_prior'
                if self.peddet_cfgs.get('refine_box_detach', False):
                    query_embed = outputs_bbox[..., :2].transpose(0, 1).detach()  # BxQx2
                else:
                    query_embed = outputs_bbox[..., :2].transpose(0, 1)  # BxQx2

            if self.peddet_cfgs.get('filter', False):
                if i == self.num_layers - 2:
                    scores = outputs_class.detach().sigmoid().transpose(0, 1)
                    seed_mask = scores > self.peddet_cfgs.score_thr
                    cur_all_mask = (~seed_mask) & (scores > self.peddet_cfgs.floor_thr)

        assert len(predictions_class) == self.num_layers # + 1
        assert len(predictions_mask) == self.num_layers  #+ 1
        assert len(predictions_bbox) == self.num_layers  #+ 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_boxes': predictions_bbox[-1],
            "pred_logits_one2many": predictions_class_one2many[-1],
            "pred_boxes_one2many": predictions_bbox_one2many[-1],
            "mask": None,  # if type(cur_all_mask) == int else {"mask": cur_all_mask.transpose(0, 1).squeeze(2), "seed_mask": seed_mask.transpose(0, 1).squeeze(2)}
            'aux_outputs': self._set_aux_loss(
                outputs_class=predictions_class,
                outputs_mask=predictions_mask,
                outputs_det_bboxes=predictions_bbox,
                task='peddet'
            ),
            'aux_outputs_one2many': self._set_aux_loss(
                outputs_class=predictions_class_one2many,
                outputs_mask=predictions_mask,
                outputs_det_bboxes=predictions_bbox_one2many,
                task='peddet'
            )
        }
        return out

    def forward_peddet_mask(self, x, mask_features):
        # x is a list of multi-scale feature [r5, r4, r3], mask_features: r2
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))  # pos embedding
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])  # lv_emb + x

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxBxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(self.peddet_cfgs.get('share_content_query', 1), bs, 1)
        output = self.query_feat.weight.unsqueeze(1).unsqueeze(1).repeat(1,
                                                                         self.query_embed.weight.size(0) if self.peddet_cfgs.get('share_content_query', False) else 1,
                                                                         bs,
                                                                         1).reshape(-1, bs, self.hidden_dim)  # QxC -> Qx1x1xC -> (Qxq)xBxC

        predictions_mask = []
        predictions_class = []
        predictions_seg_mask = []
        # predictions_bbox = []

        # predictions_class_one2many = []  # for one2many hybrid loss only
        # predictions_bbox_one2many = []

        cur_all_mask = 1
        seed_mask = None

        if self.peddet_cfgs.get('one2many', False):

            self_attn_mask = (
                torch.zeros([self.num_queries, self.num_queries, ]).bool().cuda()  # QxQ
            )
            self_attn_mask[self.peddet_cfgs.num_queries_one2one:, 0: self.peddet_cfgs.num_queries_one2one, ] = True
            self_attn_mask[0: self.peddet_cfgs.num_queries_one2one, self.peddet_cfgs.num_queries_one2one:, ] = True
        else:
            self_attn_mask = None
        # >>> ** initial proposal ** prediction heads on learnable query features
        # outputs_class, outputs_bbox = self.forward_peddet_prediction_heads(output)
        # predictions_class.append(outputs_class)
        # predictions_mask.append(None)
        # predictions_bbox.append(outputs_bbox)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels

            assert not self.mask_on
            if self.cross_pos_embed == 'pos_prior': # sample pos embed
                if self.peddet_cfgs.get('fix_range_bug', False):
                    _query_embed = F.grid_sample(self.pos_embed, self.query_embed.weight.unsqueeze(0).unsqueeze(0)*2-1,
                                                 mode='bicubic', padding_mode='border', align_corners=False)  # 1xCx1xQ
                else:
                    _query_embed = F.grid_sample(self.pos_embed, self.query_embed.weight.unsqueeze(0).unsqueeze(0), mode='bicubic', padding_mode='border', align_corners=False) # 1xCx1xQ
                _query_embed = _query_embed[0].permute(2, 1, 0).repeat(self.peddet_cfgs.get('share_content_query', 1), bs, 1)  # q x b x c
            else:
                _query_embed = self.adapt_pos2d(pos2posemb2d(query_embed, self.hidden_dim // 2))  # q x b x c
            reference = inverse_sigmoid(query_embed)

            # >>> ** cross-attn **  attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output * cur_all_mask,  # QxBxC
                src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index] if self.cross_pos_embed == "sincos" else self.get_vis_token_pos_embed(size_list[level_index]),
                query_pos=_query_embed  # QxBxC
            ) * cur_all_mask
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=self_attn_mask,
                tgt_key_padding_mask=None,
                query_pos=_query_embed
            ) * cur_all_mask

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            import pdb;
            # pdb.set_trace()
            outputs_class, outputs_mask = self.forward_peddet_mask_prediction_heads(output, mask_features, reference)

            if self.peddet_cfgs.get('one2many', False):
                predictions_mask.append(None)
                predictions_class.append(outputs_class[:, 0:self.peddet_cfgs.num_queries_one2one])
                # predictions_bbox.append(outputs_bbox[:, 0:self.peddet_cfgs.num_queries_one2one])

                # predictions_class_one2many.append(outputs_class[:, self.peddet_cfgs.num_queries_one2one:])
                # predictions_bbox_one2many.append(outputs_bbox[:, self.peddet_cfgs.num_queries_one2one:])
            else:
                predictions_mask.append(None)
                predictions_class.append(outputs_class)
                predictions_seg_mask.append(outputs_mask)
                # predictions_bbox.append(outputs_bbox)

                # predictions_class_one2many.append(None)
                # predictions_bbox_one2many.append(None)

            if self.peddet_cfgs.get('refine_box', False):
                assert self.cross_pos_embed != 'pos_prior'
                if self.peddet_cfgs.get('refine_box_detach', False):
                    query_embed = outputs_bbox[..., :2].transpose(0, 1).detach()  # BxQx2
                else:
                    query_embed = outputs_bbox[..., :2].transpose(0, 1)  # BxQx2

            if self.peddet_cfgs.get('filter', False):
                if i == self.num_layers - 2:
                    scores = outputs_class.detach().sigmoid().transpose(0, 1)
                    seed_mask = scores > self.peddet_cfgs.score_thr
                    cur_all_mask = (~seed_mask) & (scores > self.peddet_cfgs.floor_thr)

        assert len(predictions_class) == self.num_layers # + 1
        assert len(predictions_mask) == self.num_layers  #+ 1
        assert len(predictions_seg_mask) == self.num_layers  #+ 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_seg_mask[-1],
            # 'pred_boxes': predictions_bbox[-1],
            # "pred_logits_one2many": predictions_class_one2many[-1],
            # "pred_boxes_one2many": predictions_bbox_one2many[-1],
            "mask": None,  # if type(cur_all_mask) == int else {"mask": cur_all_mask.transpose(0, 1).squeeze(2), "seed_mask": seed_mask.transpose(0, 1).squeeze(2)}
            'aux_outputs': self._set_aux_loss(
                outputs_class=predictions_class,
                outputs_seg_masks=predictions_seg_mask,
                outputs_mask=predictions_mask,
                task='peddet_mask'
            ),
        }
        # import pdb;
        # pdb.set_trace()
        return out

    def forward_peddet_mask_prediction_heads(self, output, mask_features, reference=None):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)

        if self.cfgs.get('mask_head_type', 'default') == 'direct':
            mask_embed = output.transpose(0, 1)
        elif self.cfgs.get('mask_head_type', 'default') == 'norm_direct':
            mask_embed = decoder_output
        else:
            mask_embed = self.mask_embed(decoder_output)

        mask_list = torch.split(mask_embed, mask_features.shape[1], dim=-1)
        outputs_mask_tl = torch.einsum("bqc,bchw->bqhw", mask_list[0], mask_features)
        outputs_mask_br = torch.einsum("bqc,bchw->bqhw", mask_list[1], mask_features)
        outputs_mask = torch.cat([outputs_mask_tl[:,:,None,...], outputs_mask_br[:,:,None,...]],dim=2)
        # outputs_bbox = self.bbox_embed(decoder_output)
        # if not self.peddet_cfgs.get('predict_absolute_coord', False):
        #     outputs_bbox[..., :2] += reference.transpose(0, 1)
        return outputs_class, outputs_mask

    def forward_peddet_prediction_heads(self, output, reference=None):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)

        outputs_bbox = self.bbox_embed(decoder_output)
        if not self.peddet_cfgs.get('predict_absolute_coord', False):
            outputs_bbox[..., :2] += reference.transpose(0, 1)
        return outputs_class, outputs_bbox.sigmoid()


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    # QxBx2
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t  # QxBx128
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)

    return posemb


def mask2pos(mask):
    not_mask = ~mask
    y_embed = not_mask[:, :, 0].cumsum(1, dtype=torch.float32)
    x_embed = not_mask[:, 0, :].cumsum(1, dtype=torch.float32)
    y_embed = (y_embed - 0.5) / y_embed[:, -1:]
    x_embed = (x_embed - 0.5) / x_embed[:, -1:]
    return y_embed, x_embed
