import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ECAAttention, CBAMBlock, SEAttention

from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from ..ckpt import checkpoint_wrapper

__all__ = ['LadderSideAttentionFPN', 'ResidualLadderSideAttentionFPN']

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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class LadderSideAttentionFPN(nn.Module):
    """
    https://arxiv.org/abs/2206.06522
    """
    def __init__(self, layer_feat_nums=12, 
                       hidden_dim=768,
                       reduct_ration=16, 
                       transformer_block_nums=1, 
                       transformer_block_num_heads=2, 
                       gate_T=0.1, 
                       gate_alpha=0, 
                       use_cls_token=False, 
                       reduct_after_use_attention=None, 
                       backbone=None, 
                       bn_group=None,
                       lms_checkpoint_train=None,
                       block_weights=None):
        super(LadderSideAttentionFPN, self).__init__()
        self.lms_checkpoint_train = lms_checkpoint_train
        self.block_weights = block_weights
        
        self.layer_feat_nums = layer_feat_nums
        self.transformer_block_nums = transformer_block_nums
        self.transformer_block_num_heads = transformer_block_num_heads
        self.reduct_after_use_attention = reduct_after_use_attention

        self.hidden_dim = hidden_dim
        self.reduction_dim = hidden_dim // reduct_ration
        self.gate_T = gate_T
        self.gate_alpha = gate_alpha
        self.use_cls_token = use_cls_token
        
        self.reduction_layers = self.generate_reduction_layers()
        self.side_gate_params = self.generate_gate_params()
        self.transformer_blocks = self.generate_transformer_blocks()
        self.attention_layers = self.generate_attention_layers()
        
        if self.lms_checkpoint_train == 'fairscale':
            self.last_proj = checkpoint_wrapper(nn.Linear(self.reduction_dim, self.hidden_dim))
        else:
            self.last_proj = nn.Linear(self.reduction_dim, self.hidden_dim)
    
        self.apply(self._init_weights)

    def generate_reduction_layers(self):
        reduction_layers = []
        for i in range(self.layer_feat_nums + 1):
            if self.lms_checkpoint_train == 'fairscale':
                reduction_layers.append(checkpoint_wrapper(nn.Linear(self.hidden_dim, self.reduction_dim)))
            else:
                reduction_layers.append(nn.Linear(self.hidden_dim, self.reduction_dim))
        return nn.ModuleList(reduction_layers)
    
    def generate_gate_params(self):
        side_gate_params = nn.ParameterList(
                    [nn.Parameter(torch.ones(1) * self.gate_alpha) 
                    for i in range(self.layer_feat_nums)]
                )
        return side_gate_params
    
    def generate_transformer_blocks(self):
        transformer_blocks = nn.ModuleList()
        for i in range(self.layer_feat_nums):
            sub_blocks = nn.ModuleList()
            for _ in range(self.transformer_block_nums):
                if self.lms_checkpoint_train == 'fairscale':
                    sub_blocks.append(checkpoint_wrapper(TransformerBlock(dim=self.reduction_dim, num_heads=self.transformer_block_num_heads)))
                else:
                    sub_blocks.append(TransformerBlock(dim=self.reduction_dim, num_heads=self.transformer_block_num_heads))
                    
            transformer_blocks.append(sub_blocks)
        return transformer_blocks
    
    def generate_attention_layers(self):
        attention_blocks = nn.ModuleList()
        for i in range(self.layer_feat_nums + 1):
            if self.reduct_after_use_attention:
                if self.lms_checkpoint_train == 'fairscale':
                    attention_blocks.append(checkpoint_wrapper(self.attention_type_map(self.reduct_after_use_attention)))
                else:
                    attention_blocks.append(self.attention_type_map(self.reduct_after_use_attention))
            else:
                if self.lms_checkpoint_train == 'fairscale':
                    attention_blocks.append(checkpoint_wrapper(nn.Identity()))
                else:
                    attention_blocks.append(nn.Identity())
        return attention_blocks

    def attention_type_map(self, attention_name=None):
        assert attention_name in ['ECA', 'CBAM', 'SENet'], "invalid attention type"
        if attention_name == 'SENet':
            return SEAttention(channel=self.reduction_dim, use_cls_token=self.use_cls_token)
        elif attention_name == 'ECA':
            return ECAAttention(use_cls_token=self.use_cls_token)
        elif attention_name == 'CBAM':
            return CBAMBlock(channel=self.reduction_dim, use_cls_token=self.use_cls_token)
        else:
            raise ValueError("invalid attention type")

    
    def forward(self, x):
        """
        x: [patch embedding, trans_block1_output, trans_block2_output, ...]
        """
        # import pdb; pdb.set_trace()
        B, Hp, Wp = x['model_args']
        if self.block_weights:
            for i in range(len(x['backbone_output'])):
                x['backbone_output'][i] = self.block_weights[i] * x['backbone_output'][i]
                
        feats = None
        for i, block_feat in enumerate(x['backbone_output']):
            if i == 0:
                feats = self.reduction_layers[i](block_feat)
                if self.reduct_after_use_attention:
                    feats = self.attention_layers[i](feats, Hp, Wp)
                else:
                    feats = self.attention_layers[i](feats)
            else:
                gate = torch.sigmoid(self.side_gate_params[i - 1] / self.gate_T)
                if self.reduct_after_use_attention:
                    feats = gate * feats + (1 - gate) * self.attention_layers[i](self.reduction_layers[i](block_feat), Hp, Wp)
                else:
                    feats = gate * feats + (1 - gate) * self.attention_layers[i](self.reduction_layers[i](block_feat))
                    
                for transformer_block in self.transformer_blocks[i - 1]:
                    feats = transformer_block(feats, Hp, Wp)
        
        out = self.last_proj(feats)

        if self.use_cls_token:  # for reid baseline exp only
            output = {}
            output['cls_feats'] = out[:, 0]
            output['patch_feats'] = out[:, 1:].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
            x.update({'neck_output': output})
        else:
            out = out.permute(0, 2, 1).reshape(B, -1, Hp, Wp)
            x.update({'neck_output': out})
        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
