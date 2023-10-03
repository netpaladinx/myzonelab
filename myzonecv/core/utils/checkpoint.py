import os.path as osp
from collections import OrderedDict
import re

import torch
import torch.nn as nn

from . import mkdir

CKPT_MODEL_KEY = 'model'
CKPT_MODEL_KEY2 = 'state_dict'
CKPT_OPTIMIZER_KEY = 'optimizer'
CKPT_META_KEY = 'meta'


def load_checkpoint(ckpt_path, map_location=None):
    if not osp.isfile(ckpt_path):
        raise IOError(f"'{ckpt_path}' is not a checkpoint file")
    ckpt = torch.load(ckpt_path, map_location=map_location)
    return ckpt


def load_model_state_dict(model,
                          state_dict,
                          revise_keys=[(r'^module\.', '')],
                          src_prefix=None,
                          dst_prefix=None,
                          strict=False):
    metadata = getattr(state_dict, '_metadata', OrderedDict())

    for p, r in revise_keys:
        state_dict = OrderedDict([(re.sub(p, r, k), v) for k, v in state_dict.items()])

    if src_prefix:
        if not src_prefix.endswith('.'):
            src_prefix += '.'
        src_prefix_len = len(src_prefix)
        state_dict = OrderedDict([(k[src_prefix_len:], v) for k, v in state_dict.items() if k.startswith(src_prefix)])
        assert state_dict, f"{src_prefix} is not in the pretrained model"

    state_dict._metadata = metadata

    if hasattr(model, 'module'):
        model = model.module
    if dst_prefix:
        dst_prefix = dst_prefix.split('.')
        while len(dst_prefix) > 0:
            assert hasattr(model, dst_prefix[0])
            model = getattr(model, dst_prefix[0])
            dst_prefix = dst_prefix[1:]

    if hasattr(model, 'revise_state_dict'):
        state_dict = model.revise_state_dict(state_dict)

    try:
        model.load_state_dict(state_dict, strict=strict)
    except Exception as e:
        if hasattr(model, 'preprocess_loaded_state_dict') and callable(model.preprocess_loaded_state_dict):
            model.load_state_dict(model.preprocess_loaded_state_dict(model, state_dict), strict=strict)
        else:
            raise e


def load_model(ckpt,
               map_location=None,
               revise_keys=[(r'^module\.', '')],
               src_prefix=None,
               dst_prefix=None,
               model=None,
               strict=False):
    ckpt = load_checkpoint(ckpt, map_location=map_location) if isinstance(ckpt, str) else ckpt
    assert isinstance(ckpt, dict), f"Loaded checkpoint should be a dict but got {type(ckpt)}"

    if CKPT_MODEL_KEY in ckpt:
        state_dict = ckpt[CKPT_MODEL_KEY]
    elif CKPT_MODEL_KEY2 in ckpt:
        state_dict = ckpt[CKPT_MODEL_KEY2]
    else:
        state_dict = ckpt

    if isinstance(state_dict, dict):
        if model:
            load_model_state_dict(model, state_dict, revise_keys, src_prefix, dst_prefix, strict)
    elif isinstance(state_dict, nn.Module):
        model = state_dict
    return model


def save_checkpoint(model, ckpt_path, optimizer=None, meta=None):
    def _filter_out(k, v):
        if not isinstance(v, nn.parameter.Parameter) and k.startswith('_'):
            print(f"Tensor {k} is filtered out during model saving")
            return True
        return False

    if hasattr(model, 'module'):
        model = model.module
    state_dict = model.state_dict()
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    state_dict = OrderedDict([(k, v.cpu()) for k, v in state_dict.items() if not _filter_out(k, v)])
    state_dict._metadata = metadata
    ckpt = {CKPT_MODEL_KEY: state_dict}

    if optimizer:
        ckpt[CKPT_OPTIMIZER_KEY] = optimizer.state_dict()

    if meta:
        ckpt[CKPT_META_KEY] = meta

    mkdir(osp.dirname(ckpt_path), exist_ok=True)
    torch.save(ckpt, ckpt_path)
