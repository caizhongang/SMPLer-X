import numpy as np
import shutil
import torch
import os
import io
import logging
from collections import defaultdict

from torch.nn import BatchNorm2d

def param_group_no_wd(model):
    pgroup_no_wd = []
    names_no_wd = []
    pgroup_normal = []

    type2num = defaultdict(lambda : 0)
    for name,m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if m.bias is not None:
                pgroup_no_wd.append(m.bias)
                names_no_wd.append(name+'.bias')
                type2num[m.__class__.__name__+'.bias'] += 1
        elif isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                pgroup_no_wd.append(m.bias)
                names_no_wd.append(name+'.bias')
                type2num[m.__class__.__name__+'.bias'] += 1
        elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, SyncBatchNorm2d):
            if m.weight is not None:
                pgroup_no_wd.append(m.weight)
                names_no_wd.append(name+'.weight')
                type2num[m.__class__.__name__+'.weight'] += 1
            if m.bias is not None:
                pgroup_no_wd.append(m.bias)
                names_no_wd.append(name+'.bias')
                type2num[m.__class__.__name__+'.bias'] += 1

    for name,p in model.named_parameters():
        if not name in names_no_wd:
            pgroup_normal.append(p)

    return [{'params': pgroup_normal}, {'params': pgroup_no_wd, 'weight_decay': 0.0}], type2num

def param_group_fc(model):
    logits_w_id = id(model.module.logits.weight)
    fc_group = []
    normal_group = []
    for p in model.parameters():
        if id(p) == logits_w_id:
            fc_group.append(p)
        else:
            normal_group.append(p)
    param_group = [{'params': fc_group}, {'params': normal_group}]

    return param_group

def param_group_multitask(model):
    backbone_group = []
    neck_group = []
    decoder_group = []
    other_group = []
    for name, p in model.named_parameters():
        if 'module.backbone_module' in name:
            backbone_group.append(p)
        elif 'module.neck_module' in name:
            neck_group.append(p)
        elif 'module.decoder_module' in name:
            decoder_group.append(p)
        else:
            other_group.append(p)

    if len(other_group) > 0:
        param_group = [{'params': backbone_group}, {'params': neck_group}, \
                        {'params': decoder_group}, {'params', other_group}]
    else:
        param_group = [{'params': backbone_group}, {'params': neck_group}, \
                        {'params': decoder_group}]
    return param_group


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed"):
        return 0
    elif var_name.startswith("backbone.patch_embed"):
        return 0
    elif var_name.startswith("backbone.blocks"):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return num_max_layer - 1


def vit_layer_decay_param_group(model, num_layers=12, layer_decay_rate=0.75, base_lr=1e-3, base_weight_decay=1e-1):
    parameter_groups = []
    num_layers = num_layers + 2

    # vit backbone params group
    for name, param in model.module.backbone_module.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or 'pos_embed' in name:
            this_weight_decay = 0.
        else:
            this_weight_decay = base_weight_decay

        layer_id = get_num_layer_for_vit(name, num_layers)
        scale = layer_decay_rate ** (num_layers - layer_id - 1)
        parameter_groups.append({
                                "weight_decay": this_weight_decay,
                                "params": [param],
                                "lr_scale": scale,
                                "lr": scale * self.base_lr,
                                })
    neck_group = []
    decoder_group = []
    other_group = []

    for name, p in model.named_parameters():
        if 'module.backbone_module' in name:
            continue
        elif 'module.neck_module' in name:
            neck_group.append(p)
        elif 'module.decoder_module' in name:
            decoder_group.append(p)
        else:
            other_group.append(p)

    parameter_groups.append({'params': neck_group})
    parameter_groups.append({'params': decoder_group})
    parameter_groups.append({'params': other_group})

    return param_group
