import contextlib
import functools
import itertools

import torch

from . import utils, wrap

__all__ = ['half_function', 'float_function',
           'register_half_function', 'register_float_function', 
           'register_float_module', 'init', 'reset']

_DECORATOR_HANDLE = None
_USER_CAST_REGISTRY = set()
_USER_FLOAT_MODULE = set()
_ORIGINAL_MODULE_HALF = None

def _decorator_helper(orig_fn, cast_fn, wrap_fn):
    def wrapper(*args, **kwargs):
        handle = _DECORATOR_HANDLE
        if handle is None or not handle.is_active():
            return orig_fn(*args, **kwargs)
        inner_cast_fn = utils.verbosify(cast_fn, orig_fn.__name__,
                                  handle.verbose)
        return wrap_fn(orig_fn, inner_cast_fn, handle)(*args, **kwargs)
    return wrapper

# Decorator form
def half_function(fn):
    wrap_fn = functools.partial(wrap.make_cast_wrapper, try_caching=True)
    return _decorator_helper(fn, utils.maybe_half, wrap_fn)

def float_function(fn):
    wrap_fn = functools.partial(wrap.make_cast_wrapper, try_caching=False)
    return _decorator_helper(fn, utils.maybe_float, wrap_fn)

# Registry form
def register_half_function(module, name):
    if not hasattr(module, name):
        raise ValueError('No function named {} in module {}.'.format(
            name, module))
    _USER_CAST_REGISTRY.add((module, name, utils.maybe_half))

def register_float_function(module, name):
    if not hasattr(module, name):
        raise ValueError('No function named {} in module {}.'.format(
            name, module))
    _USER_CAST_REGISTRY.add((module, name, utils.maybe_float))

def register_float_module(module, cast_args=True):
    if not issubclass(module, torch.nn.modules.module.Module):
        raise ValueError('{} is not a torch Module'.format(module))

    if cast_args:
        register_float_function(module, 'forward')

    _USER_FLOAT_MODULE.add(module)

class AmpHandle(object):
    def __init__(self, enable_caching=True, verbose=False):
        self._enable_caching = enable_caching
        self._verbose = verbose
        self._cache = dict()
        self._is_active = True
        self._all_wrappers = []

    def is_active(self):
        return self._is_active

    @contextlib.contextmanager
    def _disable_casts(self):
        self._is_active = False
        yield
        self._is_active = True

    def _clear_cache(self):
        self._cache.clear()

    # Experimental support for saving / restoring uncasted versions of functions
    def _save_func(self, mod, fn, func):
        self._all_wrappers.append((mod, fn, func))

    def _deactivate(self):
        for mod, fn, func in self._all_wrappers:
            utils.set_func(mod, fn, func)
        self._all_wrappers = []

    @property
    def has_cache(self):
        return self._enable_caching

    @property
    def cache(self):
        return self._cache

    def remove_cache(self, param):
        if self.has_cache and param in self.cache:
            del self.cache[param]

    @property
    def verbose(self):
        return self._verbose

def _half_helper(verbose=False):
    def _half_wrapper(self):
        for module in self.children():
            module.half()

        if self.__class__ in _USER_FLOAT_MODULE:
            if verbose:
                print('Skip half convert for {}'.format(self.__class__))
            return self

        fn = lambda t: t.half() if t.is_floating_point() else t
        for param in self._parameters.values():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self
    return _half_wrapper

def init(enable_caching=True, verbose=False):
    global _DECORATOR_HANDLE
    global _ORIGINAL_MODULE_HALF

    handle = AmpHandle(enable_caching, verbose)

    if len(_USER_FLOAT_MODULE) > 0:
        _ORIGINAL_MODULE_HALF = torch.nn.modules.module.Module.half
        utils.set_func(torch.nn.modules.module.Module, 'half', 
                       _half_helper(verbose))

    # Force-{fp16, fp32} for user-annotated functions
    for mod, fn, cast_fn in _USER_CAST_REGISTRY:
        try_caching = (cast_fn == utils.maybe_half)
        wrap.cached_cast(mod, fn, cast_fn, handle,
                         try_caching, verbose)
    _USER_CAST_REGISTRY.clear()

    _DECORATOR_HANDLE = handle
    return handle

def _clear_cache():
    handle = _DECORATOR_HANDLE
    if handle is None or not handle.is_active():
        return
    handle._clear_cache()

def reset():
    handle = _DECORATOR_HANDLE
    if handle is None or not handle.is_active():
        return
    handle._deactivate()
    utils.set_func(torch.nn.modules.module.Module, 'half', _ORIGINAL_MODULE_HALF)