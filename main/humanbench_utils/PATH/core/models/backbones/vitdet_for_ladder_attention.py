# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
import math
import os
import torch
import numpy as np
from functools import partial
from dict_recursive_update import recursive_update

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_train
from PATH.core import distributed_utils as dist

from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from PATH.core.utils import NestedTensor
from ..ckpt import checkpoint_wrapper


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, window_size=None, rel_pos_spatial=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.rel_pos_spatial = rel_pos_spatial
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.window_size = window_size
        if COMPAT:
            if COMPAT == 2:
                self.rel_pos_h = nn.Parameter(torch.zeros(2 * window_size[0] - 1, head_dim))
                self.rel_pos_w = nn.Parameter(torch.zeros(2 * window_size[1] - 1, head_dim))
            else:
                q_size = window_size[0]
                kv_size = q_size
                rel_sp_dim = 2 * q_size - 1
                self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
                self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        if self.rel_pos_spatial:
            raise
            attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def calc_rel_pos_spatial(
        attn,
        q,
        q_shape,
        k_shape,
        rel_pos_h,
        rel_pos_w,
):
    """
    Spatial Relative Positional Embeddings.

    Source: https://github.com/facebookresearch/mvit/
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio)
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio)
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
            attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, :, None]
            + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, rel_pos_spatial=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.rel_pos_spatial=rel_pos_spatial

        if COMPAT:
            q_size = window_size[0]
            kv_size = window_size[1]
            rel_sp_dim = 2 * q_size - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        x = x.reshape(B_, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size[0])  # nW*B, window_size, window_size, C
        x = x.view(-1, self.window_size[1] * self.window_size[0], C)  # nW*B, window_size*window_size, C

        B_w = x.shape[0]
        N_w = x.shape[1]
        qkv = self.qkv(x).reshape(B_w, N_w, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)   --> (batchsize, heads, len, head_dim)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        if self.rel_pos_spatial:
            raise

        attn = attn.softmax(dim=-1)
        _attn_mask = (torch.isinf(attn) + torch.isnan(attn))
        attn = attn.masked_fill(_attn_mask, 0)

        x = (attn @ v).transpose(1, 2).reshape(B_w, N_w, C)
        x = self.proj(x)

        x = x.view(-1, self.window_size[1], self.window_size[0], C)
        x = window_reverse(x, self.window_size[0], Hp, Wp)  # B H' W' C

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B_, H * W, C)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, window=False, rel_pos_spatial=False, prompt=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if not window:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                window_size=window_size, rel_pos_spatial=rel_pos_spatial)
        else:
            self.attn = WindowAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                window_size=window_size, rel_pos_spatial=rel_pos_spatial
            )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, H, W, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # could be dynamic
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]  # could be dynamic
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, mask=None, **kwargs):
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)

        if mask is not None:
            mask = F.interpolate(mask[None].float(), size=(Hp, Wp)).to(torch.bool)[0]

        return x, (Hp, Wp), mask


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False,
                 drop_path_rate=0., norm_layer=None, window=True,
                 use_abs_pos_emb=False, interval=3, bn_group=None, test_pos_mode=False,
                 task_sp_list=(), neck_sp_list=(), learnable_pos=False, rel_pos_spatial=False, lms_checkpoint_train=False,
                 prompt=None, pad_attn_mask=False, freeze_iters=0,
                 act_layer='GELU', pre_ln=False, mask_input=False, ending_norm=True,
                 round_padding=False, compat=False, use_cls_token=False):
        super().__init__()
        self.pad_attn_mask = pad_attn_mask  # only effective for detection task input w/ NestedTensor wrapping
        self.lms_checkpoint_train = lms_checkpoint_train
        self.use_cls_token = use_cls_token
        self.task_sp_list = task_sp_list
        self.neck_sp_list = neck_sp_list
        self.freeze_iters = freeze_iters
        self.mask_input = mask_input
        self.ending_norm = ending_norm
        self.round_padding = round_padding

        global COMPAT
        COMPAT = compat

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches
        if use_abs_pos_emb:
            if self.use_cls_token:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=learnable_pos)
                trunc_normal_(self.cls_token, std=.02)
                trunc_normal_(self.pos_embed, std=.02)
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=learnable_pos)
                pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.patch_shape, cls_token=False)
                self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            raise

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop_path=dpr[i], norm_layer=norm_layer,
                window_size=(14, 14) if ((i + 1) % interval != 0) else self.patch_embed.patch_shape,
                window=((i + 1) % interval != 0) if window else False,
                rel_pos_spatial=rel_pos_spatial, prompt=prompt,
                act_layer=QuickGELU if act_layer == 'QuickGELU' else nn.GELU
            )
            if self.lms_checkpoint_train == 'fairscale':
                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self.ln_pre = norm_layer(embed_dim) if pre_ln else nn.Identity()  # for clip model only
        self.norm = norm_layer(embed_dim)

        ### duplicated init, only affects network weights and has no effect given pretrain
        self.apply(self._init_weights)
        self.fix_init_weight()
        ###
        self.test_pos_mode = test_pos_mode
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.mask_input else None

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _normalization(x):
        assert len(x.shape) == 4
        x = x.sub(torch.tensor([123.675, 116.280, 103.530]).view(1, 3, 1, 1).cuda()).div(torch.tensor([58.395, 57.120, 57.375]).view(1, 3, 1, 1).cuda())
        return x

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, mask=None):
        B, C, H, W = x.shape
        x, (Hp, Wp), mask = self.patch_embed(x, mask)
        batch_size, seq_len, _ = x.size()

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.test_pos_mode is False:
            if x.size(1) == self.pos_embed.size(1):
                x = x + self.pos_embed  # BxHWxC
            else: # take top-left if pos_embed > x's dimension
                x = x + self.pos_embed.reshape(1, self.patch_embed.patch_shape[0],
                                                  self.patch_embed.patch_shape[1],
                                                  self.pos_embed.size(2))[:,:Hp, :Wp, :].reshape(1, x.size(1),
                                                                                                 self.pos_embed.size(2))
        elif self.test_pos_mode == 'learnable_interpolate':
            patch_shape = (Hp, Wp)
            orig_size = (14, 14)

            # as in original scale
            pos_embed = self.pos_embed

            # as in finetuning scale
            pos_embed = pos_embed.reshape(-1, orig_size[0], orig_size[1], self.pos_embed.shape[-1]).permute(0, 3, 1, 2)
            pos_embed = torch.nn.functional.interpolate(pos_embed, size=patch_shape, mode='bicubic', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)

            x = x + pos_embed

        elif self.test_pos_mode == 'regenerate':
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (Hp, Wp), cls_token=False)
            x = x + torch.from_numpy(pos_embed).float().unscqueeze(0).cuda()
        elif self.test_pos_mode == 'scaled_regenerate':
            patch_shape = (Hp, Wp)
            orig_size = (math.ceil(Hp/20)*7, math.ceil(Wp/20)*7)

            # as in original scale
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], orig_size, cls_token=False)
            pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).cuda()

            # as in finetuning scale
            pos_embed = pos_embed.reshape(-1, orig_size[0], orig_size[1], self.pos_embed.shape[-1]).permute(0, 3, 1, 2)
            pos_embed = torch.nn.functional.interpolate(pos_embed, size=(orig_size[0]//7*20, orig_size[1]//7*20),
                                                         mode='bicubic', align_corners=False)

            # as in test image
            pos_embed = pos_embed[:, :, :patch_shape[0], :patch_shape[1]].permute(0, 2, 3, 1).flatten(1, 2)

            x = x + pos_embed
        elif self.test_pos_mode == 'simple_interpolate':
            patch_shape = (Hp, Wp)
            orig_size = (14, 14)

            # as in original scale
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], orig_size, cls_token=False)
            pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).cuda()

            # as in finetuning scale
            pos_embed = pos_embed.reshape(-1, orig_size[0], orig_size[1], self.pos_embed.shape[-1]).permute(0, 3, 1, 2)
            pos_embed = torch.nn.functional.interpolate(pos_embed, size=patch_shape, mode='bicubic', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)

            x = x + pos_embed
        elif self.test_pos_mode == 'learnable_simple_interpolate':
            patch_shape = (Hp, Wp)
            x = x + get_abs_pos(self.pos_embed, self.use_cls_token, patch_shape)
        else:
            raise NotImplementedError

        # x = self.random_masking(x) # effective only if self.mask_input=True (default False), for mask based ssl
        x = self.ln_pre(x)  # effective for clip model only, otherwise nn.Identity

        mid_features = []
        mid_features.append(x)


        for i, blk in enumerate(self.blocks):
            # *Warning*: official ckpt implementation leads to NaN loss in many cases, use fairscale if that's the case
            # lms_checkpoint_train = {False, True, 'fairscale'}
            if self.lms_checkpoint_train == True:
                x = checkpoint_train(lambda x: blk(x, Hp, Wp, mask), x, preserve_rng_state=True)
            else:
                x = blk(x, Hp, Wp)

            if i != len(self.blocks) - 1:
                mid_features.append(x)

        if self.ending_norm:
            x = self.norm(x)  # b h*w c

        # x = self.unmasking(x)  # effective only if self.mask_input=True (default False), for mask based ssl
        return {'mid_features': mid_features, 'model_args': (B, Hp, Wp)}
        # return x.permute(0, 2, 1).reshape(B, -1, Hp, Wp)

    def forward(self, input_var):
        output = {}
        x = input_var['image']

        if isinstance(x, NestedTensor):
            x, mask = x.decompose()
        else:
            mask = None

        # pre_input padding for test support
        x = self._normalization(x)

        if self.round_padding:
        # pre_input padding for non standard img size support, *** used when test image size varies and not divisible by 32 ***
            stride = self.patch_embed.patch_size
            assert stride[0] == stride[1]
            stride = max(stride[0], self.round_padding)
            output["prepad_input_size"] = [x.shape[-2], x.shape[-1]]  # h, w for sem_seg_postprocess
            target_size = (torch.tensor((x.shape[-1], x.shape[-2])) + (stride - 1)).div(stride, rounding_mode="floor") * stride  # w, h
            padding_size = [  # [l,r,t,b]
                0,
                target_size[0] - x.shape[-1],
                0,
                target_size[1] - x.shape[-2],
            ]
            x = F.pad(x, padding_size, value=0.).contiguous()
            if mask is not None:
                mask = F.pad(mask, padding_size, value=True).contiguous()  # 0: content, 1: pad
        # pre_input padding for test support >>> end
        output["image"] = x
        outs = self.forward_features(x)
        output['backbone_output'], output['model_args'] = outs['mid_features'], outs['model_args']

        # if isinstance(input_var['image'], NestedTensor) and self.pad_attn_mask:
        #     output['backbone_output'] = NestedTensor(self.forward_features(x), mask)
        # else:
        #     output['backbone_output'] = self.forward_features(x)
        input_var.update(output)
        return input_var


def vit_base_patch16_ladder_attention(pretrained=False, load_pos_embed=True, **kwargs):
    default = dict(
        drop_path_rate=0.1, use_abs_pos_emb=True,  # as in table 11
        ####
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    recursive_update(default, kwargs)
    model = ViT(**default)
    # del model.head

    if pretrained:
        script_dir = os.path.dirname(__file__)

        if pretrained == 'supervised-80ecf9dd':
            rel_path = "pretrain_weights/jx_vit_base_p16_224-80ecf9dd.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))
        elif pretrained == 'clip':
            rel_path = "pretrain_weights/CLIP-ViT-B-16.pt"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))
            # rename & clean loaded keys
            checkpoint = clip_checkpoint_preprocess(checkpoint)
        else:
            rel_path = "pretrain_weights/mae_pretrain_vit_base.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))['model']

        # load while interpolates position embedding
        load_checkpoint(model, checkpoint, load_pos_embed, strict=False, logger=dummy_logger)
        del checkpoint

    return model


def vit_large_patch16_ladder_attention(pretrained=False, load_pos_embed=True, **kwargs):
    default = dict(
        drop_path_rate=0.5, use_abs_pos_emb=True,  # as in table 11
        ####
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    recursive_update(default, kwargs)
    model = ViT(**default)
    # del model.head

    # if pretrained:
    #     script_dir = os.path.dirname(__file__)

    #     if pretrained == 'supervised-80ecf9dd':
    #         rel_path = "pretrain_weights/jx_vit_base_p16_224-80ecf9dd.pth"
    #         checkpoint = torch.load(os.path.join(script_dir, rel_path))
    #     elif pretrained == 'clip':
    #         rel_path = "pretrain_weights/CLIP-ViT-B-16.pt"
    #         checkpoint = torch.load(os.path.join(script_dir, rel_path))
    #         # rename & clean loaded keys
    #         checkpoint = clip_checkpoint_preprocess(checkpoint)
    #     else:
    #         rel_path = "pretrain_weights/mae_pretrain_vit_base.pth"
    #         checkpoint = torch.load(os.path.join(script_dir, rel_path))['model']

    #     # load while interpolates position embedding
    #     load_checkpoint(model, checkpoint, load_pos_embed, strict=False, logger=dummy_logger)
    #     del checkpoint

    return model


def vit_base_patch16_ladder_attention_ema(**kwargs):
    backbone = vit_base_patch16_ladder_attention(**kwargs)
    backbone.ema = [vit_base_patch16_ladder_attention(**kwargs)]
    backbone.ema[0].mask_input = False
    return backbone


class dummy_logger:
    def info(self, **kwargs):
        print(**kwargs)

    def warning(self, **kwargs):
        print(**kwargs)


def clip_checkpoint_preprocess(checkpoint):
    for k in list(checkpoint.keys()):
        if k.startswith('visual'):
            if k in ["visual.proj", "visual.class_embedding"]:
                new_k = k
            elif k.startswith('visual.transformer.resblocks'):
                new_k = k[len("visual.transformer.res"):]
                new_k = new_k.replace('in_proj_weight', 'qkv.weight')
                new_k = new_k.replace('in_proj_bias', 'qkv.bias')
                new_k = new_k.replace('out_proj', 'proj')
                new_k = new_k.replace('ln_', 'norm')
                new_k = new_k.replace('c_fc', 'fc1')
                new_k = new_k.replace('c_proj', 'fc2')
            else:
                new_k = k[len("visual."):]
                new_k = new_k.replace('positional_embedding', 'pos_embed')
                new_k = new_k.replace('conv1', 'patch_embed.proj')
                new_k = new_k.replace('ln_post', 'norm')
            checkpoint[new_k] = checkpoint[k]
        del checkpoint[k]
    return checkpoint


def load_checkpoint(model, state_dict, load_pos_embed, strict=False, logger=None):
    """
    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    # checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    # if not isinstance(checkpoint, dict):
    #     raise RuntimeError(
    #         f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'pos_embed' in state_dict:
        if load_pos_embed:
            if model.use_cls_token:
                state_dict['pos_embed'] = interpolate_pos_embed_with_cls_token(pos_embed_checkpoint=state_dict['pos_embed'],
                                                                                patch_shape=model.patch_embed.patch_shape,
                                                                                num_extra_tokens=1)
            else:
                state_dict['pos_embed'] = interpolate_pos_embed(pos_embed_checkpoint=state_dict['pos_embed'],
                                                                patch_shape=model.patch_embed.patch_shape,
                                                                num_extra_tokens=1)
        else:
            del state_dict['pos_embed']
            print("checkpoint pos_embed removed")

    model_dict = model.state_dict()
    load_dict = {
        k: v for k, v in state_dict.items() if k in model_dict.keys()
    }
    print("Missing keys: {}".format(list(set(model_dict) - set(load_dict))))

    load_state_dict(model, state_dict, strict, logger)


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        # if is_module_wrapper(module):
        #     module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank = dist.get_rank()

    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    print("finish load")


def interpolate_pos_embed(pos_embed_checkpoint, patch_shape, num_extra_tokens):
    embedding_size = pos_embed_checkpoint.shape[-1]
    orig_size = to_2tuple(int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5))
    # class_token and dist_token are kept unchanged
    print(f"[rank {dist.get_rank()}] Position interpolate from {orig_size} to {patch_shape}")
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:] if pos_embed_checkpoint.size(0) == 1 else pos_embed_checkpoint[num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=patch_shape, mode='bicubic', align_corners=False)
    new_pos_embed = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)  # (b, h*w, c)
    return new_pos_embed


def interpolate_pos_embed_with_cls_token(pos_embed_checkpoint, patch_shape, num_extra_tokens):
    posemb_tok, posemb_grid = (
        pos_embed_checkpoint[:, :num_extra_tokens],
        pos_embed_checkpoint[0, num_extra_tokens:],
    )
    gs_old_h, gs_old_w = to_2tuple(int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5))
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=patch_shape, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, patch_shape[0] * patch_shape[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_abs_pos(abs_pos, has_cls_token, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.
    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        cls_pos = abs_pos[:, 0].reshape(abs_pos.shape[0], 1, abs_pos.shape[-1])
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        if has_cls_token:
            return torch.cat([cls_pos, new_abs_pos.permute(0, 2, 3, 1).reshape(1, h*w, -1)], dim=1)
        else:
            return new_abs_pos.permute(0, 2, 3, 1).reshape(1, h*w, -1)
    else:
        if has_cls_token:
            return torch.cat([cls_pos, abs_pos.reshape(1, h*w, -1)], dim=1)
        else:
            return abs_pos.reshape(1, h*w, -1)
