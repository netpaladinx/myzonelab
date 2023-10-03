import copy

import torch.optim as optim
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from ..registry import OPTIMIZERS

OPTIMIZERS.register_class('adadelta', optim.Adadelta)
OPTIMIZERS.register_class('adagrad', optim.Adagrad)
OPTIMIZERS.register_class('adam', optim.Adam)
OPTIMIZERS.register_class('adamw', optim.AdamW)
OPTIMIZERS.register_class('adamax', optim.Adamax)
OPTIMIZERS.register_class('sgd', optim.SGD)

EXT_PG_DECAY_WEIGHT = 'decay_weight'
EXT_PG_BIAS = 'bias'


def create_optimizer(optimizer_cfg, module):
    module = [module] if not isinstance(module, (list, tuple)) else module
    module = [mod.module if hasattr(mod, 'module') else mod for mod in module]

    optimizer_cfg = copy.deepcopy(optimizer_cfg)

    ext_param_groups = optimizer_cfg.pop('ext_param_groups', [])
    ext_param_groups = get_param_groups_by_names(module, ext_param_groups)
    params = []
    if ext_param_groups:
        ext_params = set()
        for ext_pg in ext_param_groups.values():
            ext_params.update(ext_pg)
        for mod in module:
            for p in mod.parameters():
                if p not in ext_params:
                    params.append(p)
    else:
        for mod in module:
            for p in mod.parameters():
                params.append(p)
    optimizer_cfg['params'] = params

    if EXT_PG_DECAY_WEIGHT in ext_param_groups:
        weight_decay = optimizer_cfg.pop('weight_decay', None)

    optimizer = OPTIMIZERS.create(optimizer_cfg)

    for name, ext_pg in ext_param_groups.items():
        if name == EXT_PG_DECAY_WEIGHT and weight_decay is not None:
            optimizer.add_param_group({'params': ext_pg, 'weight_decay': weight_decay})
        else:
            optimizer.add_param_group({'params': ext_pg})

    return optimizer


def get_param_groups_by_names(module, names):
    if isinstance(names, str):
        names = [names]

    param_groups = {}
    for name in names:
        if name == EXT_PG_DECAY_WEIGHT:
            param_groups[name] = get_decay_weight_params(module)
        elif name == EXT_PG_BIAS:
            param_groups[name] = get_bias_params(module)

    return param_groups


def get_decay_weight_params(module):
    params = []
    for mod in module:
        for m in mod.modules():
            if hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter) and not isinstance(m, (_BatchNorm, nn.GroupNorm)):
                params.append(m.weight)
    return params


def get_bias_params(module):
    params = []
    for mod in module:
        for m in mod.modules():
            if hasattr(m, 'bias') and isinstance(m.bias, nn.Parameter):
                params.append(m.bias)
    return params
