import inspect

import torch.nn as nn

from ...registry import NORM_LAYERS

NORM_LAYERS.register_class('bn', nn.BatchNorm2d)
NORM_LAYERS.register_class('bn1d', nn.BatchNorm1d)
NORM_LAYERS.register_class('bn2d', nn.BatchNorm2d)
NORM_LAYERS.register_class('bn3d', nn.BatchNorm3d)
NORM_LAYERS.register_class('sync_bn', nn.SyncBatchNorm)
NORM_LAYERS.register_class('gn', nn.GroupNorm)
NORM_LAYERS.register_class('ln', nn.LayerNorm)
NORM_LAYERS.register_class('in', nn.InstanceNorm2d)
NORM_LAYERS.register_class('in1d', nn.InstanceNorm1d)
NORM_LAYERS.register_class('in2d', nn.InstanceNorm2d)
NORM_LAYERS.register_class('in3d', nn.InstanceNorm3d)


def infer_norm_name(norm_cfg):
    norm_type = norm_cfg.get('type')
    norm_cls = NORM_LAYERS.get(norm_type)
    assert inspect.isclass(norm_cls)
    cls_name = norm_cls.__name__.lower()
    if 'batch' in cls_name:
        return 'bn'
    elif 'group' in cls_name:
        return 'gn'
    elif 'layer' in cls_name:
        return 'ln'
    elif 'instance' in cls_name:
        return 'in'
    else:
        return 'norm_layer'


def create_norm_layer(norm_cfg, n_features):
    norm_type = norm_cfg.get('type')
    norm_cfg = norm_cfg.copy()
    requires_grad = norm_cfg.pop('requires_grad', True)
    norm_cfg.setdefault('eps', 1e-5)

    if norm_type != 'gn':
        norm_args = (n_features,)
        norm_layer = NORM_LAYERS.create(norm_cfg, norm_args)
    else:
        assert 'num_groups' in norm_cfg
        norm_cfg['num_channels'] = n_features
        norm_layer = NORM_LAYERS.create(norm_cfg)

    for param in norm_layer.parameters():
        param.requires_grad = requires_grad

    return norm_layer
