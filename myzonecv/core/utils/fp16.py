import functools
import inspect
from collections import abc

import torch
import torch.nn as nn
from torch.cuda.amp import autocast


def cast_tensor_type(inputs, dst_type):
    if isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type)
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({k: cast_tensor_type(v, dst_type) for k, v in inputs.items()})
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(cast_tensor_type(item, dst_type) for item in inputs)
    return inputs


def auto_fp16(apply_to=None, out_fp32=False):
    """ Decorator to enable fp16 training automatically

    If input arguments are fp32 tensors, they will be converted to fp16 automatically.
    Arguments other than fp32 tensors are ignored. Requires PyTorch > 1.6.

    Args:
        apply_to (Iterable): The argument names to be converted. `None` means all arguments.
        out_fp32 (bool): Whether to convert the output back to fp32.
    """
    def auto_fp16_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            self = args[0]
            if not isinstance(self, nn.Module):
                raise TypeError("@auto_fp16 can only be used to decorate the method of nn.Moule")
            if not (hasattr(self, 'fp16_enabled') and self.fp16_enabled):
                return old_func(*args, **kwargs)

            args_info = inspect.getfullargspec(old_func)
            args_to_cast = args_info.args if apply_to is None else apply_to
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(cast_tensor_type(args[i], torch.half))
                    else:
                        new_args.append(args[i])

            new_kwargs = {}
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(arg_value, torch.half)
                    else:
                        new_kwargs[arg_name] = arg_value

            with autocast(enabled=True):
                output = old_func(*new_args, **new_kwargs)

            if out_fp32:
                output = cast_tensor_type(output, torch.float)
            return output

        return new_func

    return auto_fp16_wrapper
