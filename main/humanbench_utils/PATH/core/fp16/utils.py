import functools
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


__all__ = ['BN_convert_float', 'network_to_half', 'prep_param_lists', 
           'model_grads_to_master_grads', 'master_params_to_model_params']

# True for post-0.4, when Variables/Tensors merged.
def variable_is_tensor():
    v = torch.autograd.Variable()
    return isinstance(v, torch.Tensor)

def tensor_is_variable():
    x = torch.Tensor()
    return type(x) == torch.autograd.Variable

# False for post-0.4
def tensor_is_float_tensor():
    x = torch.Tensor()
    return type(x) == torch.FloatTensor

# Akin to `torch.is_tensor`, but returns True for Variable
# objects in pre-0.4.
def is_tensor_like(x):
    return torch.is_tensor(x) or isinstance(x, torch.autograd.Variable)

# Wraps `torch.is_floating_point` if present, otherwise checks
# the suffix of `x.type()`.
def is_floating_point(x):
    if hasattr(torch, 'is_floating_point'):
        return torch.is_floating_point(x)
    try:
        torch_type = x.type()
        return torch_type.endswith('FloatTensor') or \
            torch_type.endswith('HalfTensor') or \
            torch_type.endswith('DoubleTensor')
    except AttributeError:
        return False

def scalar_python_val(x):
    if hasattr(x, 'item'):
        return x.item()
    else:
        if isinstance(x, torch.autograd.Variable):
            return x.data[0]
        else:
            return x[0]

def iter_params(param_groups):
    for group in param_groups:
        for p in group['params']:
            yield p

def is_fp_tensor(x):
    if is_nested(x):
        # Fast-fail version of all(is_fp_tensor)
        for y in x:
            if not is_fp_tensor(y):
                return False
        return True
    return is_tensor_like(x) and is_floating_point(x)

def is_nested(x):
    return isinstance(x, tuple) or isinstance(x, list)

def should_cache(x):
    if is_nested(x):
        # Fast-fail version of all(should_cache)
        for y in x:
            if not should_cache(y):
                return False
        return True
    return isinstance(x, torch.nn.parameter.Parameter) and \
        type_string(x) == 'FloatTensor'

def collect_fp_tensor_types(args, kwargs):
    def collect_types(x, types):
        if is_nested(x):
            for y in x:
                collect_types(y, types)
        else:
            types.add(type_string(x))

    all_args = itertools.chain(args, kwargs.values())
    types = set()
    for x in all_args:
        if is_fp_tensor(x):
            collect_types(x, types)
    return types

def type_string(x):
    return x.type().split('.')[-1]

def maybe_half(x, name='', verbose=False):
    if is_nested(x):
        return type(x)([maybe_half(y) for y in x])

    if not x.is_cuda or type_string(x) == 'HalfTensor':
        return x
    else:
        if verbose:
            print('Float->Half ({})'.format(name))
        return x.half()

def maybe_float(x, name='', verbose=False):
    if is_nested(x):
        return type(x)([maybe_float(y) for y in x])

    if not x.is_cuda or type_string(x) == 'FloatTensor':
        return x
    else:
        if verbose:
            print('Half->Float ({})'.format(name))
        return x.float()

# NB: returneds casted `args`, mutates `kwargs` in-place
def casted_args(cast_fn, args, kwargs):
    new_args = []
    for x in args:
        if is_fp_tensor(x):
            new_args.append(cast_fn(x))
        else:
            new_args.append(x)
    for k in kwargs:
        val = kwargs[k]
        if is_fp_tensor(val):
            kwargs[k] = cast_fn(val)
    return new_args

def cached_cast(cast_fn, x, cache):
    if is_nested(x):
        return type(x)([cached_cast(y) for y in x])
    if x in cache:
        cached_x = cache[x]
        # During eval, it's possible to end up caching casted weights
        # with requires_grad == False. This is then a problem when they
        # get reused on the next train iter. So we ensure that cached
        # weights have same requires_grad flag of most recent request.
        if x.requires_grad != cached_x.requires_grad:
            cached_x.requires_grad_(x.requires_grad)
        return cache[x]

    casted_x = cast_fn(x)
    cache[x] = casted_x
    return casted_x

def verbosify(cast_fn, fn_name, verbose):
    if verbose:
        return functools.partial(cast_fn, name=fn_name, verbose=verbose)
    else:
        return cast_fn

def as_inplace(fns):
    for x in fns:
        yield x + '_'

def has_func(mod, fn):
    if isinstance(mod, torch.nn.backends.backend.FunctionBackend):
        return fn in mod.function_classes
    elif isinstance(mod, dict):
        return fn in mod
    else:
        return hasattr(mod, fn)

def get_func(mod, fn):
    if isinstance(mod, torch.nn.backends.backend.FunctionBackend):
        return mod.function_classes[fn]
    elif isinstance(mod, dict):
        return mod[fn]
    else:
        return getattr(mod, fn)

def set_func(mod, fn, new_fn):
    if isinstance(mod, torch.nn.backends.backend.FunctionBackend):
        mod.function_classes[fn] = new_fn
    elif isinstance(mod, dict):
        mod[fn] = new_fn
    else:
        setattr(mod, fn, new_fn)

def set_func_save(handle, mod, fn, new_fn):
    cur_fn = get_func(mod, fn)
    handle._save_func(mod, fn, cur_fn)
    set_func(mod, fn, new_fn)

class tofp16(nn.Module):
    """
    Model wrapper that implements::

        def forward(self, input):
            return input.half()
    """

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


def BN_convert_float(module):
    '''
    Designed to work with network_to_half.
    BatchNorm layers need parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    '''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def network_to_half(network):
    """
    Convert model to half precision in a batchnorm-safe way.
    """
    return nn.Sequential(tofp16(), BN_convert_float(network.half()))


def prep_param_lists(model, flat_master=False):
    """
    Creates a list of FP32 master parameters for a given model, as in 
    `Training Neural Networks with Mixed Precision:  Real Examples`_.

    Args:
        model (torch.nn.Module): Existing Pytorch model
        flat_master (bool, optional, default=False):  Flatten the master 
            parameters into a single tensor, as a performance optimization.
    Returns:
        A tuple (``model_params``, ``master_params``). ``model_params`` is a 
            list of the model's parameters for later use with 
            :func:`model_grads_to_master_grads` and 
            :func:`master_params_to_model_params`.  
            ``master_params`` is a list of FP32 master gradients.  
            If ``flat_master=True``, ``master_params`` will be a list with one 
            element.

    Example::

        model_params, master_params = prep_param_lists(model)

    .. warning::
        Currently, if ``flat_master=True``, all the model's parameters must be 
        the same type.  If the model has parameters of different types, use 
        ``flat_master=False``, or use :class:`FP16_Optimizer`.

    .. _`Training Neural Networks with Mixed Precision:  Real Examples`:
        http://on-demand.gputechconf.com/gtc/2018/video/S81012/
    """
    model_params = [param for param in model.parameters() if param.requires_grad]

    if flat_master:
        # Give the user some more useful error messages
        try:
            # flatten_dense_tensors returns a contiguous flat array.
            # http://pytorch.org/docs/master/_modules/torch/_utils.html
            master_params = _flatten_dense_tensors([param.data for param in 
                                                    model_params]).float()
        except:
            print("Error in prep_param_lists:  model may contain a mixture of parameters "
                      "of different types.  Use flat_master=False, or use F16_Optimizer.")
            raise
        master_params = torch.nn.Parameter(master_params)
        master_params.requires_grad = True
        # master_params.register_hook(backwards_debug_hook)
        if master_params.grad is None:
            master_params.grad = master_params.new(*master_params.size())
        return model_params, [master_params]
    else:
        master_params = [param.clone().float().detach() for param in model_params]
        for param in master_params:
            param.requires_grad = True
        return model_params, master_params


def model_grads_to_master_grads(model_params, master_params, flat_master=False):
    """
    Copy model gradients to master gradients.  

    Args:
        model_params:  List of model parameters created by :func:`prep_param_lists`.
        master_params:  List of FP32 master parameters created by 
        :func:`prep_param_lists`.  If ``master_params`` was created with 
        ``flat_master=True``, ``flat_master=True`` should also be supplied to 
        :func:`model_grads_to_master_grads`.
    """
    if flat_master:
        # The flattening may incur one more deep copy than is necessary.
        master_params[0].grad.data.copy_(
            _flatten_dense_tensors([p.grad.data for p in model_params]))
    else:
        for model, master in zip(model_params, master_params):
            if model.grad is not None:
                if master.grad is None:
                    master.grad = Variable(master.data.new(*master.data.size()))
                master.grad.data.copy_(model.grad.data)
            else:
                master.grad = None


def master_params_to_model_params(model_params, master_params, flat_master=False):
    """
    Copy master parameters to model parameters.

    Args:
        model_params:  List of model parameters created by :func:`prep_param_lists`.
        master_params:  List of FP32 master parameters created by 
            :func:`prep_param_lists`.  If ``master_params`` was created with 
            ``flat_master=True``, ``flat_master=True`` should also be supplied 
            to :func:`master_params_to_model_params`.
    """
    if flat_master:
        for model, master in zip(model_params, 
                                 _unflatten_dense_tensors(master_params[0].data, 
                                                          model_params)):
            model.data.copy_(master)
    else:
        for model, master in zip(model_params, master_params):
            model.data.copy_(master.data)

# Backward compatibility fixes

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 0 and TORCH_MINOR <= 4:
    clip_grad_norm = torch.nn.utils.clip_grad_norm
else:
    clip_grad_norm = torch.nn.utils.clip_grad_norm_
