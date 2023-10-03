import torch.nn as nn

from ...registry import PAD_LAYERS

PAD_LAYERS.register_class('zero', nn.ZeroPad2d)
PAD_LAYERS.register_class('reflect', nn.ReflectionPad2d)
PAD_LAYERS.register_class('replicate', nn.ReflectionPad2d)


def create_pad_layer(pad_cfg, *args, **kwargs):
    pad_cfg = pad_cfg.copy()
    pad_cfg.update(kwargs)
    if args:
        pad_layer = PAD_LAYERS.create(pad_cfg, args)
    else:
        pad_layer = PAD_LAYERS.create(pad_cfg)
    return pad_layer
