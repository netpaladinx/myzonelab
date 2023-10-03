import warnings
import math

import torch
import torch.nn as nn

from ...registry import CONV_LAYERS, BLOCKS
from ..base_module import BaseModule, Sequential, ModuleList
from ..initializers import kaiming_init, constant_init
from .norm import infer_norm_name, create_norm_layer
from .acti import create_acti_layer
from .pad import create_pad_layer
from .transformer import TransformerLayer

CONV_LAYERS.register_class('conv', nn.Conv2d)
CONV_LAYERS.register_class('conv1d', nn.Conv1d)
CONV_LAYERS.register_class('conv2d', nn.Conv2d)
CONV_LAYERS.register_class('conv3d', nn.Conv3d)


def create_conv_layer(conv_cfg, *args, **kwargs):
    conv_cfg = conv_cfg.copy()
    conv_cfg.update(kwargs)
    if args:
        conv_layer = CONV_LAYERS.create(conv_cfg, args)
    else:
        conv_layer = CONV_LAYERS.create(conv_cfg)
    return conv_layer


@BLOCKS.register_class('conv_block')
class ConvBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding='auto',
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg={'type': 'conv'},
                 norm_cfg={'type': 'bn'},
                 acti_cfg={'type': 'relu'},
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'acti')):
        super().__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.acti_cfg = acti_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_pad = padding_mode not in ('zeros', 'circular')
        self.with_norm = norm_cfg is not None
        self.with_acti = acti_cfg is not None
        self.with_bias = not self.with_norm if bias == 'auto' else bias
        assert isinstance(order, (list, tuple)) and len(set(order) & set(['conv', 'norm', 'acti'])) == 3
        self.order = order
        self.fuse_enabled = False

        if padding == 'auto':
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]

        if self.with_explicit_pad:
            pad_cfg = {'type': padding_mode}
            self.pad_layer = create_pad_layer(pad_cfg, padding)
            conv_padding = 0
        else:
            conv_padding = padding

        self.conv = create_conv_layer(conv_cfg, in_channels, out_channels,
                                      kernel_size=kernel_size, stride=stride, padding=conv_padding,
                                      dilation=dilation, groups=groups, bias=self.with_bias)
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        if self.with_norm:
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name = infer_norm_name(norm_cfg)
            norm = create_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
            if self.with_bias:
                if isinstance(norm_cfg['type'], ('bn', 'bn1d', 'bn2d', 'bn3d', 'in', 'in1d', 'in2d', 'in3d')):
                    warnings.warn("Unnecessary conv bias before batch/instance norm")

        if self.with_acti:
            acti_cfg = acti_cfg.copy()
            if acti_cfg['type'] not in ('tanh', 'prelu', 'sigmoid', 'hardsigmoid', 'hsigmoid', 'swish', 'clamp', 'silu', 'softmax'):
                acti_cfg.setdefault('inplace', inplace)
            self.acti = create_acti_layer(acti_cfg)

    @property
    def norm(self):
        if self.with_norm:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        if not hasattr(self.conv, 'init_weights'):
            if self.with_acti and self.acti_cfg['type'] == 'leaky_relu':
                nonlinearity = 'leaky_relu'
                a = self.acti_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, acti=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_pad:
                    x = self.pad_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'acti' and acti and self.with_acti:
                x = self.acti(x)
        return x

    def fuse(self, enable=True):
        if enable and not self.fuse_enabled:
            self.fuse_enabled = True
            self._orig_order = tuple(self.order)
            assert self._orig_order == ('conv', 'norm', 'acti')
            self.order = ('conv', 'acti')
        elif not enable and self.fuse_enabled:
            self.fuse_enabled = False
            self.order = self._orig_order


@BLOCKS.register_class('conv')
class Conv(ConvBlock):
    """ A frequently used ConvBlock """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding='auto',
                 groups=1,
                 activation=True):
        if padding == 'auto':
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]

        if activation is True:
            acti_cfg = {'type': 'silu'}
        elif isinstance(activation, str) and activation:
            acti_cfg = {'type': activation}
        else:
            acti_cfg = None

        super().__init__(in_channels,
                         out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=1,
                         groups=groups,
                         bias=False,
                         conv_cfg={'type': 'conv'},
                         norm_cfg={'type': 'bn'},
                         acti_cfg=acti_cfg,
                         inplace=True,
                         with_spectral_norm=False,
                         padding_mode='zeros',
                         order=('conv', 'norm', 'acti'))


@BLOCKS.register_class('depthwise_conv')
class DepthwiseConv(Conv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding='auto',
                 activation=True):
        groups = math.gcd(in_channels, out_channels)

        super().__init__(in_channels,
                         out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         groups=groups,
                         activation=activation)


@BLOCKS.register_class('bottleneck')
class Bottleneck(BaseModule):
    # shortcut added after the last activation
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 groups=1,
                 expansion=0.5):
        super().__init__()

        mid_channels = int(out_channels * expansion)

        self.conv1 = Conv(in_channels, mid_channels, 1, 1)
        self.conv2 = Conv(mid_channels, in_channels, 3, 1, groups=groups)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.shortcut:
            out += x
        return out


@BLOCKS.register_class('csp_bottleneck')
class CSPBottleneck(BaseModule):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self,
                 in_channels,
                 out_channels,
                 n_inner_blocks=1,
                 shortcut=True,
                 groups=1,
                 expansion=0.5):
        super().__init__()

        mid_channels = int(out_channels * expansion)

        self.conv1 = Conv(in_channels, mid_channels, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, mid_channels, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, 1, 1, bias=False)
        self.conv4 = Conv(2 * mid_channels, out_channels, 1, 1)
        self.mod = Sequential(*[Bottleneck(mid_channels, mid_channels, shortcut, groups, expansion=1) for _ in range(n_inner_blocks)])
        self.bn = nn.BatchNorm2d(2 * mid_channels)
        self.acti = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.mod(out1)
        out1 = self.conv3(out1)

        out2 = self.conv2(x)

        out = torch.cat((out1, out2), dim=1)  # mid_channes -> 2 * mid_channels
        out = self.bn(out)
        out = self.acti(out)
        out = self.conv4(out)
        return out


@BLOCKS.register_class('c3_bottleneck')
class C3Bottleneck(BaseModule):
    # CSP Bottleneck with 3 conv blocks
    def __init__(self,
                 in_channels,
                 out_channels,
                 n_inner_blocks=1,
                 shortcut=True,
                 groups=1,
                 expansion=0.5):
        super().__init__()

        mid_channels = int(out_channels * expansion)

        self.conv1 = Conv(in_channels, mid_channels, 1, 1)
        self.conv2 = Conv(in_channels, mid_channels, 1, 1)
        self.conv3 = Conv(2 * mid_channels, out_channels, 1, 1)
        self.mod = Sequential(*[Bottleneck(mid_channels, mid_channels, shortcut, groups, expansion=1) for _ in range(n_inner_blocks)])

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.mod(out1)

        out2 = self.conv2(x)

        out = torch.cat((out1, out2), dim=1)  # mid_channes -> 2 * mid_channels
        out = self.conv3(out)
        return out


@BLOCKS.register_class('conv_transformer')
class ConvTransformer(BaseModule):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, in_dims, out_dims, n_heads, n_layers):
        super().__init__()

        if in_dims != out_dims:
            self.conv = Conv(in_dims, out_dims)
        else:
            self.conv = None

        self.linear = nn.linear(out_dims, out_dims)  # learnable position embedding
        self.transform = Sequential(*[TransformerLayer(out_dims, n_heads) for _ in range(n_layers)])
        self.n_dims = out_dims

    def forward(self, x):
        out = x if self.conv is None else self.conv(x)
        bs, _, h, w = out.shape
        out = out.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)  # bs,c,h,w -> bs,c,hw, -> 1,bs,c,hw -> hw,bs,c,1 -> hw,bs,c
        out = self.linear(out) + out
        out = self.transform(out)
        out = out.unsqueeze(3).transpose(0, 3).reshape(bs, self.n_dims, w, h)  # hw,bs,c -> hw,bs,c,1 -> 1,bs,c,hw -> bs,c,h,w


@BLOCKS.register_class('c3_transformer')
class C3Transformer(BaseModule):
    # CSP Transformer block with 3 conv blocks
    def __init__(self,
                 in_channels,
                 out_channels,
                 n_inner_blocks=1,
                 expansion=0.5,
                 n_heads=4):
        super().__init__()

        mid_channels = int(out_channels * expansion)

        self.conv1 = Conv(in_channels, mid_channels, 1, 1)
        self.conv2 = Conv(in_channels, mid_channels, 1, 1)
        self.conv3 = Conv(2 * mid_channels, out_channels, 1, 1)
        self.mod = ConvTransformer(mid_channels, mid_channels, n_heads, n_inner_blocks)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.mod(out1)

        out2 = self.conv2(x)

        out = torch.cat((out1, out2), dim=1)  # mid_channes -> 2 * mid_channels
        out = self.conv3(out)
        return out


@BLOCKS.register_class('spp')
class SPP(BaseModule):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(5, 9, 13)):
        super().__init__()

        mid_channels = in_channels // 2

        self.conv1 = Conv(in_channels, mid_channels, 1, 1)
        self.conv2 = Conv((len(kernel_size) + 1) * mid_channels, out_channels, 1, 1)
        self.mod = ModuleList([nn.MaxPool2d(k, stride=1, padding=k // 2) for k in kernel_size])

    def forward(self, x):
        out = self.conv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            out = torch.cat([out] + [b(out) for b in self.mod], dim=1)
            out = self.conv2(out)
            return out


@BLOCKS.register_class('sppf')
class SPPF(BaseModule):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5):  # equivalent to SPP(kernel_size=(5, 9, 13))
        super().__init__()

        mid_channels = in_channels // 2

        self.conv1 = Conv(in_channels, mid_channels, 1, 1)
        self.conv2 = Conv(4 * mid_channels, out_channels, 1, 1)
        self.mod = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        out = self.conv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            out2 = self.mod(out)
            out3 = self.mod(out2)
            out4 = self.mod(out3)
            out = torch.cat([out, out2, out3, out4], dim=1)
            out = self.conv2(out)
            return out


@BLOCKS.register_class('c3_spp')
class C3SPP(BaseModule):
    # CSP SPP with 3 conv blocks
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=0.5,
                 kernel_size=(5, 9, 13)):
        super().__init__()

        mid_channels = int(out_channels * expansion)

        self.conv1 = Conv(in_channels, mid_channels, 1, 1)
        self.conv2 = Conv(in_channels, mid_channels, 1, 1)
        self.conv3 = Conv(2 * mid_channels, out_channels, 1, 1)
        self.mod = SPP(mid_channels, mid_channels, kernel_size)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.mod(out1)

        out2 = self.conv2(x)

        out = torch.cat((out1, out2), dim=1)  # mid_channes -> 2 * mid_channels
        out = self.conv3(out)
        return out


@BLOCKS.register_class('ghost_conv')
class GhostConv(BaseModule):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 groups=1,
                 activation=True):
        super().__init__()

        mid_channels = out_channels // 2

        self.conv1 = Conv(in_channels, mid_channels, kernel_size, stride, 'auto', groups, activation)
        self.conv2 = Conv(mid_channels, mid_channels, 5, 1, 'auto', mid_channels, activation)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out = torch.cat((out1, out2), dim=1)
        return out


@BLOCKS.register_class('ghost_bottleneck')
class GhostBottleneck(BaseModule):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1):
        super().__init__()

        mid_channels = out_channels // 2

        conv1 = GhostConv(in_channels, mid_channels, 1, 1)
        if stride == 2:
            conv2 = DepthwiseConv(mid_channels, mid_channels, kernel_size, stride, activation=False)
        else:
            conv2 = nn.Identity()
        conv3 = GhostConv(mid_channels, out_channels, 1, 1, activation=False)

        self.conv = Sequential(conv1, conv2, conv3)

        conv1 = DepthwiseConv(in_channels, in_channels, kernel_size, stride, activation=False)
        if stride == 2:
            conv2 = Conv(in_channels, out_channels, 1, 1, activation=False)
        else:
            conv2 = nn.Identity()

        self.shortcut = Sequential(conv1, conv2)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


@BLOCKS.register_class('c3_ghost_bottleneck')
class C3GhostBottleneck(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 n_inner_blocks=1,
                 expansion=0.5):
        super().__init__()

        mid_channels = int(out_channels * expansion)

        self.conv1 = Conv(in_channels, mid_channels, 1, 1)
        self.conv2 = Conv(in_channels, mid_channels, 1, 1)
        self.conv3 = Conv(2 * mid_channels, out_channels, 1, 1)
        self.mod = Sequential(*[GhostBottleneck(mid_channels, mid_channels) for _ in range(n_inner_blocks)])

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.mod(out1)

        out2 = self.conv2(x)

        out = torch.cat((out1, out2), dim=1)  # mid_channes -> 2 * mid_channels
        out = self.conv3(out)
        return out


@BLOCKS.register_class('focus')
class Focus(BaseModule):
    # Focus width-height information into c-space
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding='auto',
                 groups=1,
                 activation=True):
        super().__init__()
        self.conv = Conv(4 * in_channels, out_channels, kernel_size, stride, padding, groups, activation)

    def forward(self, x):
        x1 = x[..., ::2, ::2]
        x2 = x[..., 1::2, ::2]
        x3 = x[..., ::2, 1::2]
        x4 = x[..., 1::2, 1::2]
        out = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.conv2(out)
        return out
