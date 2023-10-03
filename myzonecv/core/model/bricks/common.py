import torch

from ...registry import BLOCKS
from ..base_module import BaseModule


def create_block(block_cfg, *args, **kwargs):
    block_cfg = block_cfg.copy()
    block_cfg.update(kwargs)
    if args:
        block = BLOCKS.create(block_cfg, args)
    else:
        block = BLOCKS.create(block_cfg)
    return block


@BLOCKS.register_class('contract')
class Contract(BaseModule):
    # Contract height-width into channels, i.e. 1,64,80,80 -> 1,256,40,40
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        bs, c, h, w = x.shape
        assert h % self.gain == 0 and w % self.gain == 0, 'Indivisible gain'
        out = x.view(bs, c, h // self.gain, self.gain, w // self.gain, self.gain)
        out = out.permute(0, 3, 5, 1, 2, 4).contiguous()
        out = out.view(bs, self.gain**2 * c, h // self.gain, w // self.gain)
        return out


@BLOCKS.register_class('expand')
class Expand(BaseModule):
    # Expand channels into height-width, i.e. 1,64,80,80 -> 1,16,160,160
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        bs, c, h, w = x.shape
        assert bs % (self.gain**2) == 0, 'Indivisible gain'
        out = x.view(bs, self.gain, self.gain, c // (self.gain**2), h, w)
        out = out.permute(0, 3, 4, 1, 5, 2).contiguous()
        out = out.view(bs, c // (self.gain**2), h * self.gain, w * self.gain)
        return out


@BLOCKS.register_class('concat')
class Concat(BaseModule):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out = torch.cat(x, self.dim)
        return out


@BLOCKS.register_class('lambda')
class Lambda(BaseModule):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x, *args, **kwargs):
        return self.func(x, *args, **kwargs)
