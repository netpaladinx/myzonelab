import copy
from abc import ABCMeta

import torch.nn as nn

from ..utils import get_root_logger
from .initializers import initialize


class _BaseModule(metaclass=ABCMeta):
    def __init__(self, init_cfg=None):
        self.initialized = False

        if init_cfg is not None:
            init_cfg = copy.deepcopy(init_cfg)
            if isinstance(init_cfg, dict):
                init_cfg = [init_cfg]
        self.init_cfg = init_cfg

    def init_weights(self):
        logger = get_root_logger()

        if not self.initialized:
            pre_children_init_cfg = None
            post_children_init_cfg = None
            if self.init_cfg is not None:
                pre_children_init_cfg = [cfg for cfg in self.init_cfg if 'type' in cfg and cfg['type'] not in ('pretrained', 'checkpoint')]
                post_children_init_cfg = [cfg for cfg in self.init_cfg if 'type' in cfg and cfg['type'] in ('pretrained', 'checkpoint')]

            if pre_children_init_cfg:
                logger.info(f"Initialize {self.__class__.__name__} with init_cfg {pre_children_init_cfg}")
                initialize(self, pre_children_init_cfg)

            for m in self.children():
                if hasattr(m, 'init_weights'):
                    m.init_weights()

            if post_children_init_cfg:
                logger.info(f"Initialize {self.__class__.__name__} with init_cfg {post_children_init_cfg}")
                initialize(self, post_children_init_cfg)

            self.initialized = True

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device


class BaseModule(nn.Module, _BaseModule):
    def __init__(self, init_cfg=None):
        nn.Module.__init__(self)
        _BaseModule.__init__(self, init_cfg)

        self.is_dummy = False

    def forward(self, *args, **kwargs):
        if kwargs.pop('is_dummy', self.is_dummy):
            return self.forward_dummy(*args, **kwargs)

        training = kwargs.pop('training', self.training)
        if training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_predict(*args, **kwargs)

    def forward_train(self, *args, **kwargs):
        raise NotImplementedError

    def forward_predict(self, *args, **kwargs):
        raise NotImplementedError

    def forward_dummy(self, *args, **kwargs):
        raise NotImplementedError


class Sequential(nn.Sequential, _BaseModule):
    def __init__(self, *args, init_cfg=None):
        nn.Sequential.__init__(self, *args)
        _BaseModule.__init__(self, init_cfg)


class ModuleList(nn.ModuleList, _BaseModule):
    def __init__(self, modules=None, init_cfg=None):
        nn.ModuleList.__init__(self, modules)
        _BaseModule.__init__(self, init_cfg)


def create_sequential_if_list(registry, cfg, args=None):
    if isinstance(cfg, (list, tuple)):
        modules = [registry.default_create(c, args) for c in cfg]
        return Sequential(*modules)
    else:
        return registry.default_create(cfg, args)


def create_modulelist_if_list(registry, cfg, args=None):
    if isinstance(cfg, (list, tuple)):
        modules = [registry.default_create(c, args) for c in cfg]
        return ModuleList(modules)
    else:
        return registry.default_create(cfg, args)
