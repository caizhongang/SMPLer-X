import numpy as np
import shutil
import torch
import os
import io
import copy
import math
import logging
from collections import defaultdict

from PATH.core import distributed_utils as dist

from torch.nn import BatchNorm2d
from torch.utils.checkpoint import checkpoint
import cv2
import subprocess
from PIL import Image
import PATH.core.fp16 as fp16
from typing import Optional, List
from torch import Tensor

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torch.nn as nn

cv2.ocl.setUseOpenCL(False)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length):
        self.length = length
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0

    def empty(self):
        return len(self.history) == 0

    def update(self, val):
        self.history.append(val)
        if self.length > 0 and len(self.history) > self.length:
            del self.history[0]

        self.val = val
        self.avg = np.mean(self.history)


class AverageMinMaxMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length):
        self.length = length
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.min = 10000
        self.max = 0
        self.avg = 0

    def empty(self):
        return len(self.history) == 0

    def update(self, val):
        self.history.append(val)
        if self.length > 0 and len(self.history) > self.length:
            del self.history[0]

        self.val = val
        self.avg = np.mean(self.history)
        self.min = min(self.min, val)
        self.max = max(self.max, val)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_multi(output, target):
    pred = (output > 0).float()
    tf = (pred == target).float()
    acc = tf.sum() / output.size(0) / output.size(1) * 100
    return acc

def save_state(state, path, step):
    path, filename = os.path.split(path)
    assert path != ''
    if not os.path.exists(path):
        os.makedirs(path)
    print('saving to {}/{}_iter_{}.pth.tar'.format(path, filename, step))
    torch.save(state, '{}/{}_iter_{}.pth.tar'.format(path, filename, step))

def load_last_iter(path):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location='cpu')
        dist.barrier()
        print("=> loaded last_iter={} from {}".format(checkpoint['step'], path))
        dist.barrier()
        return checkpoint['step']
    else:
        raise RuntimeError("=> no checkpoint found at {}".format(path))


def remove_prefix_string(string, prefix):
    assert string.startswith(prefix), "can not remove prefix."
    return string[len(prefix):]


def remove_prefix_from_state_dict(state_dict, prefix):
    for old_key in list(state_dict.keys()):
        if old_key.startswith(prefix):
            new_key = remove_prefix_string(old_key, prefix)
            state_dict[new_key] = state_dict.pop(old_key)


def load_state(path, model, ignore=[], optimizer=None, cuda=False, recover=False,
               remove_prefix=None, strict=False):
    def map_func_cuda(storage, location):
        return storage.cuda()
    def map_func_cpu(storage, location):
        return storage.cpu()
    if cuda:
        map_func = map_func_cuda
    else:
        map_func = map_func_cpu

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)

        if 'state_dict' in checkpoint.keys():
            pretrained_state_dict = checkpoint['state_dict']
        else:
            pretrained_state_dict = checkpoint

        if len(ignore) > 0:
            assert optimizer == None

            for k in list(pretrained_state_dict.keys()):
                flag = False
                for prefix in ignore:
                     if k.startswith(prefix):
                         flag = True
                         the_prefix = prefix
                         break
                if flag:
                    print('ignoring {} (prefix: {})'.format(k, the_prefix))
                    del pretrained_state_dict[k]
        if remove_prefix:
            remove_prefix_from_state_dict(pretrained_state_dict, remove_prefix)
        model.load_state_dict(pretrained_state_dict, strict=strict)
        dist.barrier()
        if dist.get_rank() == 0:
            keys1 = set(pretrained_state_dict.keys())
            keys2 = set([k for k,_ in model.named_parameters()])
            not_loaded = keys2 - keys1
            for k in not_loaded:
                print('caution: {} not loaded'.format(k))
        dist.barrier()
        if optimizer != None:
            assert len(ignore) == 0

            #TODO currently a workaround for gpu memory leak
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
                    else:
                        state[k] = v
                        print("k: {} do not move to cuda".format(k))

            print("=> loaded checkpoint '{}' (step {})".format(path, checkpoint['step']))
            return checkpoint['step']
        if recover:
            return checkpoint['step']
    else:
        assert False, "=> no checkpoint found at '{}'".format(path)


def load_state_model(model, state, ginfo):
    if ginfo.task_rank == 0:
        printlog(f'======= loading model state for task {ginfo.task_id} ... =======')

    msg = model.load_state_dict(state, strict=False)

    state_keys = set(state.keys())
    model_keys = set(model.state_dict().keys())
    missing_keys = model_keys - state_keys
    if ginfo.task_rank == 0:
        for k in missing_keys:
            printlog(f'missing key: {k}')
    printlog(f'load msg: {msg}')


def load_state_optimizer(optimizer, state, ginfo):
    if ginfo.task_rank == 0:
        printlog(f'======= loading optimizer state for task {ginfo.task_id} ... =======')

    optimizer.load_state_dict(state)

def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s][%(filename)20s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l

class IterLRScheduler(object):
    def __init__(self, optimizer, milestones, lr_mults, last_iter=-1):
        assert len(milestones) == len(lr_mults), "{} vs {}".format(len(milestones), len(lr_mults))
        self.milestones = milestones
        self.lr_mults = lr_mults
        if not isinstance(optimizer, torch.optim.Optimizer) and not isinstance(optimizer, fp16.FP16_Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        for i, group in enumerate(optimizer.param_groups):
            if 'lr' not in group:
                raise KeyError("param 'lr' is not specified "
                               "in param_groups[{}] when resuming an optimizer".format(i))
        self.last_iter = last_iter

    def _get_lr(self):
        try:
            pos = self.milestones.index(self.last_iter)
        except ValueError:
            return list(map(lambda group: group['lr'], self.optimizer.param_groups))
        except:
            raise Exception('wtf?')
        return list(map(lambda group: group['lr']*self.lr_mults[pos], self.optimizer.param_groups))

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.last_iter + 1
        self.last_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
            param_group['lr'] = lr

def reset_bn(module):
    if isinstance(module, BatchNorm2d) or isinstance(module, torch.nn.SyncBatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img

def cv2_loader(img_str):
    img_array = np.frombuffer(img_str, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def param_groups(model):
    bn_group = []
    fc_group = []
    feature_group = []
    normal_group = []

    bn_names = set()
    for name,m in model.named_modules():
        if isinstance(m, BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            if not m.weight is None:
                bn_group.append(m.weight)
                bn_names.add(name+'.weight')
            if not m.bias is None:
                bn_group.append(m.bias)
                bn_names.add(name+'.bias')

    for name,param in model.named_parameters():
        if name in bn_names:
            continue
        elif name.startswith('module.base.fc'):
            feature_group.append(param)
        elif name.startswith('module.logits'):
            fc_group.append(param)
        else:
            normal_group.append(param)

    return bn_group, feature_group, fc_group, normal_group

def clip_grad_value(parameters, clip_value):
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.data.clamp_(min=-clip_value, max=clip_value)

def compute_grad_norm(parameters):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    return total_norm


class SIMSELoss(nn.Module):
    def __init__(self):
        super(SIMSELoss, self).__init__()

    def forward(self, pred, real):
        diffs = real - pred
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        simse = torch.sum(diffs).pow(2) / (n ** 2)
        return mse - simse

class GradRejust(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grad_scale):
        ctx.grad_scale = grad_scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.grad_scale * grad_output, None

def grad_rejust(x, grad_scale=1.0):
    return GradRejust.apply(x, grad_scale)

def count_parameters_num(model):
    count = 0
    count_fc = 0
    param_dict = {name:param for name,param in model.named_parameters()}
    param_keys = param_dict.keys()
    for m_name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            weight_name = m_name + '.weight'
            bias_name = m_name + '.bias'
            if weight_name in param_keys:
                temp_params = param_dict[weight_name]
                count += temp_params.data.nelement()
            if bias_name in param_keys:
                temp_params = param_dict[bias_name]
                count += temp_params.data.nelement()
        elif isinstance(m, nn.Linear):
            weight_name = m_name + '.weight'
            bias_name = m_name + '.bias'
            if weight_name in param_keys:
                temp_params = param_dict[weight_name]
                count_fc += temp_params.data.nelement()
            if bias_name in param_keys:
                temp_params = param_dict[bias_name]
                count_fc += temp_params.data.nelement()
    sync_print('Number of conv/bn params: %.2fM' % (count / 1e6))
    sync_print('Number of linear params: %.2fM' % (count_fc / 1e6))

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

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
        elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm):
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

def freeze_bn(model):
    names = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            m.eval()
            names.append(name)

    return names

def named_buffers(self, memo=None, prefix=''):
    if memo is None:
        memo = set()
    for name, b in self._buffers.items():
        if b is not None and b not in memo:
            memo.add(b)
            yield prefix + ('.' if prefix else '') + name, b
    for mname, module in self.named_children():
        submodule_prefix = prefix + ('.' if prefix else '') + mname
        for name, b in module.named_buffers(memo, submodule_prefix):
            yield name, b

def change_tensor_cuda():
    sync_print('override tensor.cuda() to preserve task_specific flag')
    # change .cuda of Tensor
    ori_tensor_cuda = torch.Tensor.cuda
    torch.Tensor.ori_cuda = ori_tensor_cuda
    def new_cuda(self, *args, **kwargs):  # could be written as decorator I guess...
        cuda_t = self.ori_cuda(*args, **kwargs)
        if hasattr(self, 'task_specific'):
            cuda_t.task_specific = self.task_specific
        if hasattr(self, 'backbone_specific'):
            cuda_t.backbone_specific = self.backbone_specific
        if hasattr(self, 'neck_specific'):
            cuda_t.neck_specific = self.neck_specific
        if hasattr(self, 'decoder_specific'):
            cuda_t.decoder_specific = self.decoder_specific
        return cuda_t
    torch.Tensor.cuda = new_cuda

def add_task_specific(m, task_specific):
    for name, param in m.named_parameters():
        param.task_specific = task_specific
        param.backbone_specific = False
        param.neck_specific = False
        param.decoder_specific = False
        if task_specific:
            printlog('add param {} as task_specific'.format(name))

    if not hasattr(torch.nn.Module, 'named_buffers'):
        printlog('registering named_buffers for nn.Module at add_task_specific')
        torch.nn.Module.named_buffers = named_buffers

    #m.cuda() # neccesary for broadcast in DistModule,
    # since buffers are tensors which will be changed after .cuda()
    for name, buffer in m.named_buffers():
        buffer.task_specific = task_specific
        buffer.backbone_specific = False
        buffer.neck_specific = False
        buffer.decoder_specific = False
        if task_specific:
            printlog('add buffer {} as task_specific'.format(name))

def add_backbone_specific(m, backbone_specific):
    for name, param in m.named_parameters():
        param.task_specific = False
        param.backbone_specific = backbone_specific
        param.neck_specific = False
        param.decoder_specific = False
        if backbone_specific:
            printlog('add param {} as backbone_specific'.format(name))

    if not hasattr(torch.nn.Module, 'named_buffers'):
        printlog('registering named_buffers for nn.Module at add_backbone_specific')
        torch.nn.Module.named_buffers = named_buffers

    #m.cuda() # neccesary for broadcast in DistModule, since buffers are tensors which will be changed after .cuda()
    for name, buffer in m.named_buffers():
        buffer.task_specific = False
        buffer.backbone_specific = backbone_specific
        buffer.neck_specific = False
        buffer.decoder_specific = False
        if backbone_specific:
            printlog('add buffer {} as backbone_specific'.format(name))

def add_neck_specific(m, neck_specific):
    for name, param in m.named_parameters():
        param.task_specific = False
        param.backbone_specific = False
        param.neck_specific = neck_specific
        param.decoder_specific = False
        if neck_specific:
            printlog('add param {} as neck_specific'.format(name))

    if not hasattr(torch.nn.Module, 'named_buffers'):
        printlog('registering named_buffers for nn.Module at add_neck_specific')
        torch.nn.Module.named_buffers = named_buffers

    #m.cuda() # neccesary for broadcast in DistModule, since buffers are tensors which will be changed after .cuda()
    for name, buffer in m.named_buffers():
        buffer.task_specific = False
        buffer.backbone_specific = False
        buffer.neck_specific = neck_specific
        buffer.decoder_specific = False
        if neck_specific:
            printlog('add buffer {} as neck_specific'.format(name))

def add_decoder_specific(m, decoder_specific):
    for name, param in m.named_parameters():
        param.task_specific = False
        param.backbone_specific = False
        param.neck_specific = False
        param.decoder_specific = decoder_specific
        if decoder_specific:
            printlog('add param {} as decoder_specific'.format(name))

    if not hasattr(torch.nn.Module, 'named_buffers'):
        printlog('registering named_buffers for nn.Module at add_decoder_specific')
        torch.nn.Module.named_buffers = named_buffers

    #m.cuda() # neccesary for broadcast in DistModule, since buffers are tensors which will be changed after .cuda()
    for name, buffer in m.named_buffers():
        buffer.task_specific = False
        buffer.backbone_specific = False
        buffer.neck_specific = False
        buffer.decoder_specific = decoder_specific
        if decoder_specific:
            printlog('add buffer {} as decoder_specific'.format(name))

def add_aio_backbone_specific(m, backbone_specific, task_sp_list=(), neck_sp_list=()):
    for name, param in m.named_parameters():
        _task_sp_flag = any(name.startswith(sp_name) or name.endswith(sp_name) for sp_name in task_sp_list)
        _neck_sp_flag = any(name.startswith(sp_name) or name.endswith(sp_name) for sp_name in neck_sp_list)

        param.task_specific = _task_sp_flag
        param.backbone_specific = False if _task_sp_flag or _neck_sp_flag else backbone_specific
        param.neck_specific = _neck_sp_flag
        param.decoder_specific = False
        if _task_sp_flag:
            printlog('add param {} as task_specific'.format(name))
        elif _neck_sp_flag:
            printlog('add param {} as neck_specific'.format(name))
        elif backbone_specific:
            printlog('add param {} as backbone_specific'.format(name))

    if not hasattr(torch.nn.Module, 'named_buffers'):
        printlog('registering named_buffers for nn.Module at add_backbone_specific')
        torch.nn.Module.named_buffers = named_buffers

    #m.cuda() # neccesary for broadcast in DistModule, since buffers are tensors which will be changed after .cuda()
    for name, buffer in m.named_buffers():
        _task_sp_flag = any(name.startswith(sp_name) or name.endswith(sp_name) for sp_name in task_sp_list)
        _neck_sp_flag = any(name.startswith(sp_name) or name.endswith(sp_name) for sp_name in neck_sp_list)

        buffer.task_specific = _task_sp_flag
        buffer.backbone_specific = False if _task_sp_flag or _neck_sp_flag else backbone_specific
        buffer.neck_specific = _neck_sp_flag
        buffer.decoder_specific = False
        if _task_sp_flag:
            printlog('add buffer {} as task_specific'.format(name))
        elif _neck_sp_flag:
            printlog('add buffer {} as neck_specific'.format(name))
        elif backbone_specific:
            printlog('add buffer {} as backbone_specific'.format(name))

def add_aio_neck_specific(m, neck_specific, task_sp_list=()):
    for name, param in m.named_parameters():
        _task_sp_flag = any(name.startswith(sp_name) or name.endswith(sp_name) for sp_name in task_sp_list)

        param.task_specific = _task_sp_flag
        param.backbone_specific = False
        param.neck_specific = False if _task_sp_flag else neck_specific
        param.decoder_specific = False
        if _task_sp_flag:
            printlog('add param {} as task_specific'.format(name))
        elif neck_specific:
            printlog('add param {} as neck_specific'.format(name))

    if not hasattr(torch.nn.Module, 'named_buffers'):
        printlog('registering named_buffers for nn.Module at add_neck_specific')
        torch.nn.Module.named_buffers = named_buffers

    #m.cuda() # neccesary for broadcast in DistModule, since buffers are tensors which will be changed after .cuda()
    for name, buffer in m.named_buffers():
        _task_sp_flag = any(name.startswith(sp_name) or name.endswith(sp_name) for sp_name in task_sp_list)

        buffer.task_specific = _task_sp_flag
        buffer.backbone_specific = False
        buffer.neck_specific = False if _task_sp_flag else neck_specific
        buffer.decoder_specific = False
        if _task_sp_flag:
            printlog('add buffer {} as task_specific'.format(name))
        elif neck_specific:
            printlog('add buffer {} as neck_specific'.format(name))

def add_aio_decoder_specific(m, decoder_specific, task_sp_list=(), neck_sp_list=()):
    for name, param in m.named_parameters():
        _task_sp_flag = any(name.startswith(sp_name) or name.endswith(sp_name) for sp_name in task_sp_list)
        _neck_sp_flag = any(name.startswith(sp_name) or name.endswith(sp_name) for sp_name in neck_sp_list)

        param.task_specific = _task_sp_flag
        param.backbone_specific = False
        param.neck_specific = _neck_sp_flag
        param.decoder_specific = False if _task_sp_flag or _neck_sp_flag else decoder_specific

        if _task_sp_flag:
            printlog('add param {} as task_specific'.format(name))
        elif _neck_sp_flag:
            printlog('add param {} as neck_specific'.format(name))
        elif decoder_specific:
            printlog('add param {} as decoder_specific'.format(name))

    if not hasattr(torch.nn.Module, 'named_buffers'):
        printlog('registering named_buffers for nn.Module at add_decoder_specific')
        torch.nn.Module.named_buffers = named_buffers

    #m.cuda() # neccesary for broadcast in DistModule, since buffers are tensors which will be changed after .cuda()
    for name, buffer in m.named_buffers():
        _task_sp_flag = any(name.startswith(sp_name) or name.endswith(sp_name) for sp_name in task_sp_list)
        _neck_sp_flag = any(name.startswith(sp_name) or name.endswith(sp_name) for sp_name in neck_sp_list)

        buffer.task_specific = _task_sp_flag
        buffer.backbone_specific = False
        buffer.neck_specific = _neck_sp_flag
        buffer.decoder_specific = False if _task_sp_flag or _neck_sp_flag else decoder_specific
        if _task_sp_flag:
            printlog('add buffer {} as task_specific'.format(name))
        elif _neck_sp_flag:
            printlog('add buffer {} as neck_specific'.format(name))
        elif decoder_specific:
            printlog('add buffer {} as decoder_specific'.format(name))


def copy_state_dict_cpu(state_dict):
    new_state = {}
    for k,v in state_dict.items():
        new_state[k] = v.cpu()
    return new_state

def copy_optim_state_dict_cpu(state_dict):
    new_state = {}
    new_state['param_groups'] = copy.deepcopy(state_dict['param_groups'])
    new_state['state'] = {}
    for k,v in state_dict['state'].items():
        new_state['state'][k] = {}
        for name,x in v.items():
            if isinstance(x, torch.Tensor):
                new_state['state'][k][name] = x.cpu()
            else:
                new_state['state'][k][name] = copy.deepcopy(x)
    return new_state

def copy_optim_state_dict_cpu_fp16(state_dict):
    new_state = {}
    new_state['optimizer_state_dict'] = copy_optim_state_dict_cpu(state_dict['optimizer_state_dict'])
    for k in state_dict.keys():
        if k != 'optimizer_state_dict':
            new_state[k] = copy.deepcopy(state_dict[k])
    return new_state

def sync_print(*args, **kwargs):
    rank = dist.get_rank()
    # dist.barrier()
    print('sync_print: rank {}, '.format(rank) + ' '.join(args), **kwargs)

def fully_checkpoint_sequential(functions, segments, input, **kwargs):
    r"""Modified version of torch.utils.checkpoint.checkpoint_sequential for memory efficiency.
    It is assumed that at least one of the inputs have requires_grad=True, so we can checkpoint
    all of the segments at ease.
    Please refer to https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint_sequential
    for more details.

    -1 -> sqrt chunk checkpoint
    0  -> no checkpoint
    others ->
    """
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end + 1):
                input = functions[j](input)
            return input
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    # no checkpoint
    if segments == 0:
        return run_function(0, len(functions) - 1, functions)(input)

    # auto determin the chunksize
    if segments < 0:
        segments = int(math.ceil(len(functions)))

    segments = min(segments, len(functions))
    segment_size = len(functions) // segments
    # the last chunk has to be non-volatile
    end = -1
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        input = checkpoint(run_function(start, end, functions), input)
#                           preserve_rng_state=preserve)
    return checkpoint(run_function(end + 1, len(functions) - 1, functions), input)#,
#                      preserve_rng_state=preserve)

def printlog(*args, **kwargs):
    print(f"[rank {dist.get_rank()}]", *args, **kwargs)

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def cuda(self):
        return self.to('cuda')

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False  # 0: content, 1: pad
    else:
        raise ValueError('not supported')

    return NestedTensor(tensor, mask)


def get_num_layer_for_vit(var_name, config):
    if (var_name == "module.backbone_module" or var_name.endswith("prompt_embed_kv")) and config.get('lpe_lr', False):
        return config.num_layers - 1
    if var_name in ("module.backbone_module", "module.backbone_module.cls_token", "module.backbone_module.mask_token"):
        return 0
    elif var_name.startswith("module.backbone_module.patch_embed"):
        return 0
    elif var_name.startswith("module.backbone_module") and not (var_name.startswith("module.backbone_module.norm") or
                                                                var_name.startswith("module.backbone_module.ln_pre")):
        layer_id = int(var_name.split('.')[3])
        return layer_id + 1
    else:
        return config.num_layers - 1


class SegmentationRunningMetrics(object):

    def __init__(self, num_classes=None, ignore_index=None):

        self.n_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.reduced_confusion_matrix = None

    def get_F1_score(self):
        assert self.n_classes == 2
        TN, FN, FP, TP = self.confusion_matrix.flatten()
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        return 2 / (1 / precision + 1 / recall), precision, recall

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        mask &= (label_pred >= 0) & (label_pred < n_class)

        if self.ignore_index is not None:
            mask = mask & (label_true != self.ignore_index)

        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2)

        # print(np.unique(label_true))
        # print(np.unique(label_pred))
        hist = hist.reshape(n_class, n_class)

        return hist

    def update(self, label_preds, label_trues):
        self.reduced_confusion_matrix = None
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def reduce_scores(self):
        hist = self.confusion_matrix
        self.reduced_confusion_matrix = hist

    def _get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        if self.reduced_confusion_matrix is None:
            self.reduce_scores()
        hist = self.reduced_confusion_matrix

        acc = np.diag(hist).sum() / hist.sum()
        acc_cls_list = acc_cls = np.diag(hist) / hist.sum(axis=1)

        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return acc, acc_cls_list, fwavacc, mean_iu, cls_iu

    def get_mean_iou(self):
        return self._get_scores()[3]

    def get_pixel_acc(self):
        return self._get_scores()[0]

    def get_mean_acc(self):
        return self._get_scores()[1]

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.reduced_confusion_matrix = None

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def cuda(self):
        return self.to('cuda')

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')

    return NestedTensor(tensor, mask)

def nested_tensor_from_tensor_list_fix_shape(tensor_list: List[Tensor],max=1334,short=801,idx=None):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        #from IPython import embed;embed()
        # # for coco, resize to 1333, 800
        # for i,idxs in zip(tensor_list,idx):
        #     print(i.shape, idxs)

        for i in range(len(tensor_list)):
            _, _h, _w = tensor_list[i].shape
            if _h != _w:
                break

        if _w > _h:
            max_size = [3, short, max]
        else:
            max_size = [3, max, short]
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        try:
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], :img.shape[2]] = False
        except:
            import pdb; pdb.set_trace()
            # raise ValueError("nested_tensor_from_tensor_list_fix_shape")
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)
