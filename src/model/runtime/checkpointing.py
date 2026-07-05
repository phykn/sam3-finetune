import inspect
from functools import wraps
from typing import Callable, TypeVar

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.utils._pytree import tree_map_only

T = TypeVar("T")


def activation_ckpt_wrapper(module: nn.Module | Callable) -> Callable:
    @wraps(module)
    def act_ckpt_wrapper(
        *args, act_ckpt_enable: bool = True, use_reentrant: bool = False, **kwargs
    ):
        if not act_ckpt_enable:
            return module(*args, **kwargs)

        if len(args) > 0:
            raise ValueError(
                "This wrapper expects keyword arguments only when `act_ckpt_enable=True`"
            )

        callable_fn = module.forward if isinstance(module, nn.Module) else module
        signature = inspect.signature(callable_fn)
        param_defaults = {
            name: param.default for name, param in signature.parameters.items()
        }

        args = []
        for name, default in param_defaults.items():
            if name in kwargs:
                args.append(kwargs.pop(name))
            elif default is not inspect.Parameter.empty:
                args.append(default)
            elif signature.parameters[name].kind is not inspect.Parameter.VAR_KEYWORD:
                raise ValueError(f"Missing positional argument: {name}")

        remaining_keys = list(kwargs.keys())
        for key in remaining_keys:
            if isinstance(kwargs[key], torch.Tensor):
                kwargs[key] = "_REMOVED_BY_ACT_CKPT_WRAPPER_"

        return checkpoint.checkpoint(
            module, *args, use_reentrant=use_reentrant, **kwargs
        )

    return act_ckpt_wrapper


def clone_output_wrapper(f: Callable[..., T]) -> Callable[..., T]:
    @wraps(f)
    def wrapped(*args, **kwargs):
        outputs = f(*args, **kwargs)
        return tree_map_only(
            torch.Tensor, lambda t: t.clone() if t.is_cuda else t, outputs
        )

    return wrapped
