from . import utils

import functools

import torch

def make_cast_wrapper(orig_fn, cast_fn, handle,
                      try_caching=False):
    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        if not handle.is_active():
            return orig_fn(*args, **kwargs)
        
        input_types = [
            v.data.type() for v in list(args) + list(kwargs.values())
                if utils.is_fp_tensor(v)
        ]
        #print('wrapper: orig_fn:{}, input_types:{}'.format(orig_fn, input_types))
        input_type = input_types[0]

        if try_caching and handle.has_cache:
            args = list(args)
            for i in range(len(args)):
                if utils.should_cache(args[i]):
                    args[i] = utils.cached_cast(cast_fn, args[i], handle.cache)
            for k in kwargs:
                if utils.should_cache(kwargs[k]):
                    kwargs[k] = utils.cached_cast(cast_fn, kwargs[k], handle.cache)
        new_args = utils.casted_args(cast_fn,
                                     args,
                                     kwargs)
        output = orig_fn(*new_args, **kwargs)
        
        #if output.type() != input_type:
        #    print('ori output type: {}, input type: {}'.format(output.type(), input_type))
        #    return output.type(input_type)    
        #return output
        return cast_output(output, input_type, verbose=False)

    return wrapper

def cast_output(output, input_type, verbose=False):
    if isinstance(output, dict):
        keys = output.keys()
        for k in keys:
            output[k] = cast_output(output[k], input_type)
        return output
    
    if utils.is_fp_tensor(output) and output.type() != input_type:
        if verbose:
            print('ori output type: {}, input type: {}'.format(output.type(), input_type))
        return output.type(input_type)
    return output

def cached_cast(mod, fn, cast_fn, handle,
                try_caching=False, verbose=False):
    if not utils.has_func(mod, fn):
        return

    orig_fn = utils.get_func(mod, fn)
    cast_fn = utils.verbosify(cast_fn, fn, verbose)
    wrapper = make_cast_wrapper(orig_fn, cast_fn, handle, try_caching)
    utils.set_func_save(handle, mod, fn, wrapper)

