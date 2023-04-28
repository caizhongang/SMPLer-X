import torch
import torch.nn as nn
import logging
from collections import Iterable

# from .misc_helper import to_device


logger = logging.getLogger('global')


def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        num = int(num)
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)

    return clever_nums


def flops_cal(model, input_shape):
    inputs = {
        'image': torch.randn(1, input_shape[0], input_shape[1], input_shape[2]),
        'image_info': [[input_shape[1], input_shape[2], 1, input_shape[1], input_shape[2], False]],
        'filename': ['Test.jpg'],
        'label': torch.LongTensor([[0]]),
    }
    # flops, params = profile(model, inputs=(to_device(inputs),))
    flops, params = profile(model, inputs=(inputs,))
    flops_str, params_str = clever_format([flops, params], "%.3f")
    flops = flops / 1e6
    params = flops / 1e6
    return flops, params, flops_str, params_str


def profile(model, inputs, verbose=True):
    handler_collection = []

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        m_type = type(m)
        fn = None
        if m_type in register_hooks:
            fn = register_hooks[m_type]

        if fn is None:
            if verbose:
                print("No implemented counting method for {} in flops_helper".format(m))
        else:
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    # original_device = model.parameters().__next__().device
    training = model.training

    model.eval()
    model.apply(add_hooks)

    # with torch.no_grad():
    model(*inputs)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_params += m.total_params

    # total_ops = total_ops.item()
    # total_params = total_params.item()
    total_ops = total_ops[0]
    total_params = total_params[0]

    # reset model to original status
    model.train(training)
    for handler in handler_collection:
        handler.remove()

    return total_ops, total_params


multiply_adds = 1


def count_zero(m, x, y):
    m.total_ops = torch.Tensor([0])
    m.total_params = torch.Tensor([0])


def count_conv2d(m, x, y):
    cin = m.in_channels
    cout = m.out_channels
    kh, kw = m.kernel_size
    out_h = y.size(2)
    out_w = y.size(3)
    batch_size = x[0].size(0)

    kernel_ops = multiply_adds * kh * kw
    bias_ops = 1 if m.bias is not None else 0

    output_elements = batch_size * out_w * out_h * cout
    total_ops = output_elements * kernel_ops * cin // m.groups + bias_ops * output_elements
    m.total_ops = torch.Tensor([int(total_ops)])

    total_params = kh * kw * cin * cout // m.groups + bias_ops * cout
    m.total_params = torch.Tensor([int(total_params)])


def count_bn(m, x, y):
    x = x[0]
    c_out = y.size(1)
    nelements = x.numel()
    # subtract, divide, gamma, beta
    total_ops = 4 * nelements

    m.total_ops = torch.Tensor([int(total_ops)])
    m.total_params = torch.Tensor([int(c_out) * 2])


def count_relu(m, x, y):
    x = x[0]
    nelements = x.numel()
    total_ops = nelements

    m.total_ops = torch.Tensor([int(total_ops)])


def count_softmax(m, x, y):
    x = x[0]
    batch_size, nfeatures = x.size()
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops = torch.Tensor([int(total_ops)])


def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size]))
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops = torch.Tensor([int(total_ops)])


def count_adap_avgpool(m, x, y):
    kernel = torch.Tensor([*(x[0].shape[2:])]) // torch.Tensor(list((m.output_size,))).squeeze()
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops = torch.Tensor([int(total_ops)])


def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_ops = torch.Tensor([int(total_ops)])
    m.total_params = torch.Tensor([m.in_features * m.out_features])


register_hooks = {
    nn.Conv2d: count_conv2d,
    nn.BatchNorm2d: count_zero,
    nn.InstanceNorm2d: count_zero,
    nn.ConvTranspose2d: count_conv2d,
    nn.ReLU: count_zero,
    nn.ReLU6: count_zero,
    nn.Tanh: count_zero,
    nn.LeakyReLU: count_zero,
    nn.AvgPool2d: count_zero,
    nn.AdaptiveAvgPool2d: count_zero,
    nn.Linear: count_linear,
    nn.Dropout: count_zero,
    nn.Sigmoid: count_zero,
    nn.Softmax: count_zero,
    # VarChannelConv2d: VarChannelConv2d.flops_count,
    # VarChannelBatchNorm2d: VarChannelBatchNorm2d.flops_count,
    # VarChannelSyncBatchNorm2d: VarChannelSyncBatchNorm2d.flops_count,
    # VarChannelSyncMultiBatchNorm2d: VarChannelSyncMultiBatchNorm2d.flops_count,
    # VarChannelLinear: VarChannelLinear.flops_count,
    # DeprecatedGroupSyncBatchNorm: count_zero,
    # Identity: count_zero,
    # VcIdentity: count_zero,
    nn.MaxPool2d: count_zero,
    nn.CrossEntropyLoss: count_zero,
    # SamePadConv2d: count_conv2d,
    # conv_bn_swish: count_zero,
    # Swish: count_zero
}
