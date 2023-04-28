# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from dataclasses import dataclass
import functools
import threading
import weakref

from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint

from collections import OrderedDict
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union, cast, Generator

import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence

"""Useful functions to deal with tensor types with other python container types."""


def apply_to_type(
    type_fn: Callable, fn: Callable, container: Union[torch.Tensor, np.ndarray, Dict, List, Tuple, Set, NamedTuple]
) -> Any:
    """Recursively apply to all objects in different kinds of container types that matches a type function."""

    def _apply(x: Union[torch.Tensor, np.ndarray, Dict, List, Tuple, Set]) -> Any:
        if type_fn(x):
            return fn(x)
        elif isinstance(x, OrderedDict):
            od = x.__class__()
            for key, value in x.items():
                od[key] = _apply(value)
            return od
        elif isinstance(x, PackedSequence):
            _apply(x.data)
            return x
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            f = getattr(x, "_fields", None)
            if f is None:
                return tuple(_apply(x) for x in x)
            else:
                assert isinstance(f, tuple), "This needs to be a namedtuple"
                # convert the namedtuple to a dict and _apply().
                x = cast(NamedTuple, x)
                _dict: Dict[str, Any] = x._asdict()
                _dict = {key: _apply(value) for key, value in _dict.items()}
                return type(x)(**_dict)  # make a copy of the namedtuple
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(container)


def apply_to_tensors(fn: Callable, container: Union[torch.Tensor, Dict, List, Tuple, Set]) -> Any:
    """Recursively apply to all tensor in different kinds of container types."""
    return apply_to_type(torch.is_tensor, fn, container)


def to_np(tensor_or_container: Union[torch.Tensor, Dict, List, Tuple, Set]) -> Any:
    """Convert a tensor or a container to numpy."""
    return apply_to_type(torch.is_tensor, lambda x: x.cpu().numpy(), tensor_or_container)


def from_np(ndarray_or_container: Union[np.ndarray, Dict, List, Tuple, Set]) -> Any:
    """Convert a ndarray or a container to tensor."""
    return apply_to_type(lambda x: isinstance(x, np.ndarray), lambda x: torch.from_numpy(x), ndarray_or_container)


def pack_kwargs(*args: Any, **kwargs: Any) -> Tuple[Tuple[str, ...], Tuple[Any, ...]]:
    """
    Turn argument list into separate key list and value list (unpack_kwargs does the opposite)
    Usage::
        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        assert kwarg_keys == ("a", "b")
        assert flat_args == (1, 2, 3, 4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == (1, 2)
        assert kwargs == {"a": 3, "b": 4}
    """
    kwarg_keys: List[str] = []
    flat_args: List[Any] = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)
    return tuple(kwarg_keys), tuple(flat_args)


def unpack_kwargs(kwarg_keys: Tuple[str, ...], flat_args: Tuple[Any, ...]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """See pack_kwargs."""
    assert len(kwarg_keys) <= len(flat_args), f"too many keys {len(kwarg_keys)} vs. {len(flat_args)}"
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[: -len(kwarg_keys)]
    kwargs = {k: v for k, v in zip(kwarg_keys, flat_args[-len(kwarg_keys) :])}
    return args, kwargs


def split_non_tensors(
    mixed: Union[torch.Tensor, Tuple[Any, ...]]
) -> Tuple[Tuple[torch.Tensor, ...], Optional[Dict[str, List[Any]]]]:
    """
    Split a tuple into a list of tensors and the rest with information
    for later reconstruction.
    When called with a tensor X, will return: (x,), None
    Usage::
        x = torch.Tensor([1])
        y = torch.Tensor([2])
        tensors, packed_non_tensors = split_non_tensors((x, y, None, 3))
        assert tensors == (x, y)
        assert packed_non_tensors == {
            "is_tensor": [True, True, False, False],
            "objects": [None, 3],
        }
        recon = unpack_non_tensors(tensors, packed_non_tensors)
        assert recon == (x, y, None, 3)
    """
    if isinstance(mixed, torch.Tensor):
        return (mixed,), None
    tensors: List[torch.Tensor] = []
    packed_non_tensors: Dict[str, List[Any]] = {"is_tensor": [], "objects": []}
    for o in mixed:
        if isinstance(o, torch.Tensor):
            packed_non_tensors["is_tensor"].append(True)
            tensors.append(o)
        else:
            packed_non_tensors["is_tensor"].append(False)
            packed_non_tensors["objects"].append(o)
    return tuple(tensors), packed_non_tensors


def unpack_non_tensors(
    tensors: Tuple[torch.Tensor, ...], packed_non_tensors: Optional[Dict[str, List[Any]]]
) -> Tuple[Any, ...]:
    """See split_non_tensors."""
    if packed_non_tensors is None:
        return tensors
    assert isinstance(packed_non_tensors, dict), type(packed_non_tensors)
    mixed: List[Any] = []
    is_tensor_list = packed_non_tensors["is_tensor"]
    objects = packed_non_tensors["objects"]
    assert len(tensors) + len(objects) == len(is_tensor_list), (
        f"len(tensors) {len(tensors)} len(objects) {len(objects)} " f"len(is_tensor_list) {len(is_tensor_list)}"
    )
    obj_i = tnsr_i = 0
    for is_tensor in is_tensor_list:
        if is_tensor:
            mixed.append(tensors[tnsr_i])
            tnsr_i += 1
        else:
            mixed.append(objects[obj_i])
            obj_i += 1
    return tuple(mixed)

# https://docs.python.org/3/library/threading.html#thread-local-data
# Manage the checkpoint context with thread-local data.

def patch_batchnorm(module: nn.Module) -> List:
    """Patch all batchnorm instances (1d, 2d, 3d, sync_bn, etc.) of a module
       so that they don't track running stats when torch.no_grad() is enabled.
       This is important in activation checkpointing to ensure stats are tracked
       correctly as if there were no activation checkpointing. The reason is
       that activation checkpointing runs the forward function twice, first
       with torch.no_grad(), then with torch.grad().
    Args:
        module (nn.Module):
            The module to be patched in-place.
    Returns:
        (list):
            A list of hook handles, late can be freed.
    """

    def pre_forward(module: _BatchNorm, input: Tensor) -> None:
        if torch.is_grad_enabled():
            return
        module._track_running_stats_backup = module.track_running_stats
        module.track_running_stats = False

    def post_forward(module: _BatchNorm, input: Tensor, result: Tensor) -> None:
        if torch.is_grad_enabled():
            return
        module.track_running_stats = module._track_running_stats_backup

    hooks = []
    for name, child in module.named_modules():
        # _BatchNorm is base for bn1d, bn2d, bn3d and sync_bn, apex_sync_bn, etc.
        if isinstance(child, _BatchNorm) and not hasattr(child, "disable_patch_batchnorm"):
            # Register the pre/post hooks.
            pre_handle = child.register_forward_pre_hook(pre_forward)
            post_handle = child.register_forward_hook(post_forward)
            hooks += [pre_handle, post_handle]
    return hooks

@dataclass
class ThreadLocalCheckpointingState(threading.local):
    is_checkpointing: bool = False
    is_recomputing: bool = False
    is_checkpointing_disabled: bool = False


thread_local = ThreadLocalCheckpointingState()


@contextmanager
def disable_checkpointing() -> Generator[None, None, None]:
    """Makes :func:`is_checkpointing_disabled` return :data:`True` within a context."""
    orig = thread_local.is_checkpointing_disabled
    thread_local.is_checkpointing_disabled = True
    try:
        yield
    finally:
        thread_local.is_checkpointing_disabled = orig


@contextmanager
def enable_checkpointing() -> Generator[None, None, None]:
    """Makes :func:`is_checkpointing` return :data:`True` within a context."""
    orig = thread_local.is_checkpointing
    thread_local.is_checkpointing = True
    try:
        yield
    finally:
        thread_local.is_checkpointing = orig


@contextmanager
def enable_recomputing() -> Generator[None, None, None]:
    """Makes :func:`is_recomputing` return :data:`True` within a context."""
    orig = thread_local.is_recomputing
    thread_local.is_recomputing = True
    try:
        yield
    finally:
        thread_local.is_recomputing = orig


def is_checkpointing() -> bool:
    """Whether the current forward propagation is under checkpointing.
    Returns:
        bool: :data:`True` if it's under checkpointing.
    """
    return thread_local.is_checkpointing


def is_recomputing() -> bool:
    """Whether the current forward propagation is under checkpoint
    recomputation. Use this to prevent duplicated side-effects at forward
    propagation::
        class Counter(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0
            def forward(self, input):
                if not is_recomputing():
                    self.counter += 1
                return input
    Returns:
        bool: :data:`True` if it's under checkpoint recomputation.
    """
    return thread_local.is_recomputing


def checkpoint_wrapper(
    module: nn.Module,
    offload_to_cpu: bool = False,
) -> nn.Module:
    """
    A friendlier wrapper for performing activation checkpointing.
    Compared to the PyTorch version, this version:
        - wraps an nn.Module, so that all subsequent calls will use checkpointing
        - handles keyword arguments in the forward
        - handles non-Tensor outputs from the forward
        - supports offloading activations to CPU
    Usage::
        checkpointed_module = checkpoint_wrapper(my_module, offload_to_cpu=True)
        a, b = checkpointed_module(x, y=3, z=torch.Tensor([1]))
    To understand the benefits of checkpointing and the `offload_to_cpu` flag,
    let's divide activations into 2 types: inner activations and outer
    activations w.r.t. the checkpointed modules. The inner ones are saved
    by activation checkpointing, the outer ones are saved by offload_to_cpu.
    In terms of GPU memory savings:
        - When inner ones are large in size and outer ones are small,
          checkpointing helps a lot, offload_to_cpu may help a little.
        - When inner ones are small and outer ones are large,
          checkpointing helps little, offload_to_cpu helps a lot.
        - When both inner and outer are large, both help and the
          benefit is additive.
    ..Note::
        The first and last layers are not likely to benefit from the `offload_to_cpu` flag
        because (1) there are typically other references to the first layer's input, so
        the GPU memory won't be freed; (2) the input to the last layer is immediately
        used by the backward pass and won't result in memory savings.
    Args:
        module (nn.Module):
            The module to be wrapped
        offload_to_cpu (bool):
            Whether to offload activations to CPU.
    Returns:
        (nn.Module):
            Wrapped module
    """
    # Patch the batchnorm layers in case there are any in this module.
    patch_batchnorm(module)

    # The use of weakref here is to prevent creating a ref cycle: m -> m.forward -> m.
    # When such cycle exists, gc won't collect the module when the module is freed.
    # That causes GPU memory to be leaked. See the unit test for how we catch that.
    #
    # We prefer this over a class wrapper since the class wrapper would have to
    # proxy a lot of fields and methods.
    module.forward = functools.partial(  # type: ignore
        _checkpointed_forward, type(module).forward, weakref.ref(module), offload_to_cpu
    )
    return module


def _checkpointed_forward(
    original_forward: Any, weak_self: Any, offload_to_cpu: bool, *args: Any, **kwargs: Any
) -> Any:
    module = weak_self()

    # If gradients are disabled, just use original `.forward()` method directly.
    if not torch.is_grad_enabled() or thread_local.is_checkpointing_disabled:
        return original_forward(module, *args, **kwargs)

    # Autograd Functions in PyTorch work best with positional args, since
    # the backward must return gradients (or None) for every input argument.
    # We can flatten keyword arguments to make this easier.
    args = (module,) + args
    kwarg_keys, flat_args = pack_kwargs(*args, **kwargs)
    parent_ctx_dict: Dict[str, Any] = {
        "offload": offload_to_cpu,
    }
    # Dummy tensor with grad is used to ensure the backward pass is called. This is needed
    # when original_forward's input are non-tensor (i.e. a tuple). Using this dummy tensor
    # avoids requiring users to set their input tensors's requires_grad flag. In the case
    # of tuple type inputs, setting the flag won't even trigger the backward pass.
    #
    # One implication of this is that since we always feed in a dummy tensor
    # needing grad, then the output will always require grad, even if it originally
    # wouldn't, such as if the module and original input both do not require grad.
    # We get around this by saving the desired requires_grad value in output and
    # detaching the output if needed.
    output = CheckpointFunction.apply(
        torch.tensor([], requires_grad=True), original_forward, parent_ctx_dict, kwarg_keys, *flat_args
    )
    output_requires_grad = parent_ctx_dict["output_requires_grad"]
    if not isinstance(output, torch.Tensor):
        # If output should not require grad, then detach it, since otherwise it will
        # always have requires_grad = True due to our dummy tensor input above that
        # requires_grad
        output = [x.detach() if not output_requires_grad else x for x in output]

        packed_non_tensor_outputs = parent_ctx_dict["packed_non_tensor_outputs"]
        if packed_non_tensor_outputs:
            output = unpack_non_tensors(output, packed_non_tensor_outputs)

    else:
        # If output should not require grad, then detach it, since otherwise it will
        # always have requires_grad = True due to our dummy tensor input above that
        # requires_grad
        if not output_requires_grad:
            output = output.detach()

    return output


def get_rng_state() -> Dict[str, Any]:
    state = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])


def is_autocast_enabled() -> bool:
    """Similar to torch.is_autocast_enabled, but compatible with torch 1.5.1"""
    if hasattr(torch, "is_autocast_enabled"):
        return torch.is_autocast_enabled()
    return False


@contextmanager
def autocast(enabled: bool) -> Generator:
    """Similar to torch.cuda.amp.autocast, but compatible with torch 1.5.1"""
    if enabled:
        with torch.cuda.amp.autocast(enabled):
            yield
    else:
        yield


class CheckpointFunction(torch.autograd.Function):
    """Similar to the torch version, but support non-Tensor outputs.
    The caller is expected to provide a dict (*parent_ctx_dict*) that will hold
    the non-Tensor outputs. These should be combined with the Tensor *outputs*
    by calling :func:`unpack_non_tensors`.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        dummy_tensor_requires_grad: torch.Tensor,
        run_function: Any,
        parent_ctx_dict: Dict[str, Any],
        kwarg_keys: Tuple[str, ...],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        torch_checkpoint.check_backward_validity(args)

        ctx.run_function = run_function
        ctx.kwarg_keys = kwarg_keys
        ctx.fwd_rng_state = get_rng_state()
        ctx.had_autocast_in_fwd = is_autocast_enabled()

        tensor_inputs, packed_non_tensor_inputs = split_non_tensors(args)
        if parent_ctx_dict["offload"]:
            ctx.fwd_device = tuple(x.device for x in tensor_inputs)
            ctx.grad_requirements = tuple(x.requires_grad for x in tensor_inputs)
            tensor_inputs = tuple(x.to("cpu", non_blocking=True) for x in tensor_inputs)
        else:
            ctx.fwd_device, ctx.grad_requirements = None, None

        ctx.save_for_backward(*tensor_inputs)
        ctx.packed_non_tensor_inputs = packed_non_tensor_inputs

        with torch.no_grad(), enable_checkpointing():
            unpacked_args, unpacked_kwargs = unpack_kwargs(kwarg_keys, args)
            outputs = run_function(*unpacked_args, **unpacked_kwargs)
            the_module = unpacked_args[0]

        # Because we run with torch.no_grad(), we can't actually access
        # outputs.requires_grad. Instead, we manually compute it by
        # checking if either the input or the module needs grads
        parameters = list(the_module.parameters())

        # If the module is wrapped by FlattenParamsWrapper, then the
        # parameters would have been deleted. If so, we need to access
        # the views into the flattened parameters.
        if hasattr(the_module, "_unflattened_param_views"):
            parameters += the_module._unflattened_param_views

        output_requires_grad = any(param.requires_grad for param in parameters) or any(
            x.requires_grad for x in tensor_inputs
        )
        parent_ctx_dict["output_requires_grad"] = output_requires_grad

        if not isinstance(outputs, torch.Tensor):
            # Autograd Functions don't like non-Tensor outputs. We can split the
            # non-Tensor and Tensor outputs, returning the former by reference
            # through *parent_ctx_dict* and returning the latter directly.
            outputs, packed_non_tensor_outputs = split_non_tensors(outputs)
            parent_ctx_dict["packed_non_tensor_outputs"] = packed_non_tensor_outputs

        return outputs

    @staticmethod
    def backward(ctx: Any, *args: Any) -> Tuple[Optional[Tensor], ...]:
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")

        tensor_inputs: Tuple = ctx.saved_tensors
        tensor_inputs = torch_checkpoint.detach_variable(tensor_inputs)
        if ctx.fwd_device is not None:
            tensor_inputs = tuple(t.to(ctx.fwd_device[i], non_blocking=True) for i, t in enumerate(tensor_inputs))
            for i, need_grad in enumerate(ctx.grad_requirements):
                tensor_inputs[i].requires_grad = need_grad
        inputs = unpack_non_tensors(tensor_inputs, ctx.packed_non_tensor_inputs)

        # Store the current states.
        bwd_rng_state = get_rng_state()

        # Set the states to what it used to be before the forward pass.
        set_rng_state(ctx.fwd_rng_state)

        with torch.enable_grad(), enable_recomputing(), autocast(ctx.had_autocast_in_fwd):
            unpacked_args, unpacked_kwargs = unpack_kwargs(ctx.kwarg_keys, inputs)
            outputs = ctx.run_function(*unpacked_args, **unpacked_kwargs)
            tensor_outputs, _ = split_non_tensors(outputs)

        # Set the states back to what it was at the start of this function.
        set_rng_state(bwd_rng_state)

        # Run backward() with only Tensors that require grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(tensor_outputs)):
            if tensor_outputs[i].requires_grad:
                outputs_with_grad.append(tensor_outputs[i])
                args_with_grad.append(args[i])

        if len(outputs_with_grad) == 0:
            raise RuntimeError("None of the outputs have requires_grad=True, " "this checkpoint() is not necessary")

        torch.autograd.backward(outputs_with_grad, args_with_grad)

        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in inputs)

        return (None, None, None, None) + grads
