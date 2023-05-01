import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch
import torch.nn as nn
from functools import reduce
import operator

try:
    import spring.linklink as link
    from spring.linklink.nn import SyncBatchNorm2d
except:
    import linklink as link
    from linklink.nn import SyncBatchNorm2d


def count_parameters_num(model):
    count = 0
    count_fc = 0
    param_dict = {name:param for name,param in model.named_parameters()}
    param_keys = param_dict.keys()
    for m_name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, SyncBatchNorm2d):
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
    print('Number of conv/bn params: %.2fM' % (count / 1e6))
    print('Number of linear params: %.2fM' % (count_fc / 1e6))
    print('Number of all params: %.2fM' % ( (count+count_fc) / 1e6))

# def count_flops(model, input_image_size):
#     counts = []

#     # loop over all model parts
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d):
#             def hook(module, input):
#                 factor = 2*module.in_channels*module.out_channels
#                 factor *= module.kernel_size[0]*module.kernel_size[1]
#                 factor //= module.stride[0]*module.stride[1]
#                 counts.append(
#                     factor*input[0].data.shape[2]*input[0].data.shape[3]
#                 )
#             m.register_forward_pre_hook(hook)
#         elif isinstance(m, nn.Linear):
#             counts += [
#                 2*m.in_features*m.out_features
#             ]
        
#     noise_image = torch.rand(
#         2, 3, input_image_size, input_image_size
#     )
#     # one forward pass
#     with torch.no_grad():
#         _ = model(torch.autograd.Variable(noise_image.cuda()))
#     return sum(counts)

def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])

def measure_model(model, input_image_size, forward_param=None):
    flop_counts = []
    param_counts = []
    multi_add = 2

    # loop over all model parts
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            def hook(module, x):
                out_h = int((x[0].size()[2] + 2 * module.padding[0] - module.kernel_size[0]) / module.stride[0] + 1)
                out_w = int((x[0].size()[3] + 2 * module.padding[1] - module.kernel_size[1]) / module.stride[1] + 1)
                ops = module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1] * out_h * out_w / module.groups * multi_add
                flop_counts.append(ops)
                param_counts.append(get_layer_param(module))
            m.register_forward_pre_hook(hook)

        elif isinstance(m, nn.ReLU) or isinstance(m, nn.PReLU):
            def hook(module, x):
                ops = x[0].numel()
                flop_counts.append(ops)
                param_counts.append(get_layer_param(module))
            m.register_forward_pre_hook(hook)

        elif isinstance(m, nn.AvgPool2d):
            def hook(module, x):
                in_w = x[0].size()[2]
                kernel_ops = module.kernel_size * module.kernel_size
                out_w = int((in_w + 2 * module.padding - module.kernel_size) / module.stride + 1)
                out_h = int((in_w + 2 * module.padding - module.kernel_size) / module.stride + 1)
                ops = x[0].size()[0] * x[0].size()[1] * out_w * out_h * kernel_ops
                param_counts.append(get_layer_param(module))
            m.register_forward_pre_hook(hook)

        elif isinstance(m, nn.AdaptiveAvgPool2d):
            def hook(module, x):
                ops = x[0].size()[0] * x[0].size()[1] * x[0].size()[2] * x[0].size()[3]
                flop_counts.append(ops)
                param_counts.append(get_layer_param(module))
            m.register_forward_pre_hook(hook)

        elif isinstance(m, nn.Linear):
            def hook(module, x):
                weight_ops = module.weight.numel() * multi_add
                bias_ops = module.bias.numel()
                ops = x[0].size()[0] * (weight_ops + bias_ops)
                flop_counts.append(ops)
                param_counts.append(get_layer_param(module))
            m.register_forward_pre_hook(hook)
        
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, SyncBatchNorm2d) or isinstance(m, nn.BatchNorm1d) \
            or isinstance(m, nn.Dropout2d) or isinstance(m, nn.Dropout):
            def hook(module, x):
                param_counts.append(get_layer_param(module))
            m.register_forward_pre_hook(hook)

        else:
            # print('unknown layer type: %s' % type(m))
            pass

    if isinstance(input_image_size, int):
        noise_image = torch.rand(1, 3, input_image_size, input_image_size)
    else:
        noise_image = torch.rand(1, 3, input_image_size[0], input_image_size[1])
    # one forward pass
    with torch.no_grad():
        if forward_param is not None:
            _ = model(noise_image.cuda(), forward_param)
        else:
            _ = model(noise_image.cuda())
    # _ = model(torch.autograd.Variable(noise_image.cuda(), requires_grad=False))

    return sum(param_counts), sum(flop_counts)
