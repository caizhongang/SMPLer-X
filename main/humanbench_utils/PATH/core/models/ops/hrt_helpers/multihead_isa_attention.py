# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Lang Huang, RainbowSecret from:
#   https://github.com/openseg-group/openseg.pytorch/blob/master/lib/models/modules/isa_block.py
# --------------------------------------------------------


import torch
import math
import warnings
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import to_2tuple

from .multihead_attention import MultiheadAttentionRPE


class PadBlock(object):
    """ "Make the size of feature map divisible by local group size."""

    def __init__(self, local_group_size=7):
        self.lgs = local_group_size
        if not isinstance(self.lgs, (tuple, list)):
            self.lgs = to_2tuple(self.lgs)
        assert len(self.lgs) == 2

    def pad_if_needed(self, x, size):
        n, h, w, c = size
        pad_h = math.ceil(h / self.lgs[0]) * self.lgs[0] - h
        pad_w = math.ceil(w / self.lgs[1]) * self.lgs[1] - w
        if pad_h > 0 or pad_w > 0:  # center-pad the feature on H and W axes
            return F.pad(
                x,
                (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            )
        return x

    def depad_if_needed(self, x, size):
        n, h, w, c = size
        pad_h = math.ceil(h / self.lgs[0]) * self.lgs[0] - h
        pad_w = math.ceil(w / self.lgs[1]) * self.lgs[1] - w
        if pad_h > 0 or pad_w > 0:  # remove the center-padding on feature
            return x[:, pad_h // 2 : pad_h // 2 + h, pad_w // 2 : pad_w // 2 + w, :]
        return x


class LocalPermuteModule(object):
    """ "Permute the feature map to gather pixels in local groups, and the reverse permutation"""

    def __init__(self, local_group_size=7):
        self.lgs = local_group_size
        if not isinstance(self.lgs, (tuple, list)):
            self.lgs = to_2tuple(self.lgs)
        assert len(self.lgs) == 2

    def permute(self, x, size):
        n, h, w, c = size
        return rearrange(
            x,
            "n (qh ph) (qw pw) c -> (ph pw) (n qh qw) c",
            n=n,
            qh=h // self.lgs[0],
            ph=self.lgs[0],
            qw=w // self.lgs[0],
            pw=self.lgs[0],
            c=c,
        )

    def rev_permute(self, x, size):
        n, h, w, c = size
        return rearrange(
            x,
            "(ph pw) (n qh qw) c -> n (qh ph) (qw pw) c",
            n=n,
            qh=h // self.lgs[0],
            ph=self.lgs[0],
            qw=w // self.lgs[0],
            pw=self.lgs[0],
            c=c,
        )


class MultiheadISAAttention(nn.Module):
    r"""interlaced sparse multi-head self attention (ISA) module with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): Window size.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        window_size=7,
        attn_type="isa_local",
        rpe=True,
        **kwargs,
    ):
        super(MultiheadISAAttention, self).__init__()

        self.dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn_type = attn_type
        self.with_rpe = rpe

        self.attn = MultiheadAttentionRPE(
            embed_dim, num_heads, rpe=rpe, window_size=window_size, **kwargs
        )
        self.pad_helper = PadBlock(window_size)
        assert attn_type in ["isa_local"]
        if attn_type == "isa_local":
            self.permute_helper = LocalPermuteModule(window_size)
        else:
            raise NotImplementedError("We only support ['isa_local'] Now.")

    def forward(self, x, H, W, **kwargs):
        # H, W = self.input_resolution
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        # attention
        if self.attn_type in ["isa_local"]:
            # pad
            x_pad = self.pad_helper.pad_if_needed(x, x.size())
            # permute
            x_permute = self.permute_helper.permute(x_pad, x_pad.size())
            # attention
            out, _, _ = self.attn(
                x_permute, x_permute, x_permute, rpe=self.with_rpe, **kwargs
            )
            # reverse permutation
            out = self.permute_helper.rev_permute(out, x_pad.size())
        else:
            raise NotImplementedError("We only support ['isa_local'] Now.")
        # de-pad, pooling with `ceil_mode=True` will do implicit padding, so we need to remove it, too
        out = self.pad_helper.depad_if_needed(out, x.size())
        return out.reshape(B, N, C)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
