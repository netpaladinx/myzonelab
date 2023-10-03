import warnings

import torch.nn as nn
import torch.nn.functional as F

from ...registry import UPSAMPLE_LAYERS, BLOCKS
from ...utils import pair
from ..base_module import BaseModule
from ..initializers import xavier_init, kaiming_init, constant_init
from .norm import infer_norm_name, create_norm_layer
from .acti import create_acti_layer
from .pad import create_pad_layer
from .conv import create_conv_layer

UPSAMPLE_LAYERS.register_class('upsample', nn.Upsample)
UPSAMPLE_LAYERS.register_class('deconv', nn.ConvTranspose2d)
UPSAMPLE_LAYERS.register_class('deconv3d', nn.ConvTranspose3d)


def create_upsample_layer(upsample_cfg, *args, **kwargs):
    upsample_cfg = upsample_cfg.copy()
    upsample_cfg.update(kwargs)
    if args:
        upsample_layer = UPSAMPLE_LAYERS.create(upsample_cfg, args)
    else:
        upsample_layer = UPSAMPLE_LAYERS.create(upsample_cfg)
    return upsample_layer


def compute_output_padding(output_size, kernel_size, stride, padding, dilation=1):
    if output_size is None or stride == 1:
        return 0

    output_size = pair(output_size)  # (h,w)
    kernel_size = pair(kernel_size)
    stride = pair(stride)
    padding = pair(padding)
    output_padding = tuple(os - ((os + 2 * p - (dilation * (ks - 1) + 1)) // s * s - 2 * p + dilation * (ks - 1) + 1)
                           for os, ks, s, p in zip(output_size, kernel_size, stride, padding))
    return output_padding


def create_deconv_layer(deconv_cfg, *args, **kwargs):
    assert deconv_cfg['type'] in ('deconv', 'deconv3d')
    return create_upsample_layer(deconv_cfg, *args, **kwargs)


def makeup_deconv_cfg(deconv_cfg):
    deconv_cfg = deconv_cfg.copy()
    assert ('kernel_size' in deconv_cfg) and ('stride' in deconv_cfg)
    kernel_size = deconv_cfg['kernel_size']
    stride = deconv_cfg['stride']
    padding = deconv_cfg.get('padding', 0)
    output_padding = deconv_cfg.get('output_padding', 0)
    if stride == 2:
        if kernel_size == 3:
            padding = 1
            output_padding = 0  # w/h agnostic
        elif kernel_size == 2:
            padding = 0
            output_padding = 0  # w/h should be even
        elif kernel_size == 4:
            padding = 1
            output_padding = 0  # w/h should be even

    deconv_cfg['padding'] = padding
    deconv_cfg['output_padding'] = output_padding
    return deconv_cfg


@UPSAMPLE_LAYERS.register_class('pixel_shuffle')
class PixelShuffle(BaseModule):
    def init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(in_channels, out_channels * scale_factor * scale_factor, upsample_kernel,
                                       padding=(upsample_kernel - 1) // 2)

    def init_weights(self):
        xavier_init(self.upsample_conv, distribution='uniform')

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


@BLOCKS.register_class('deconv_block')
class DeconvBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding='auto',
                 dilation=1,
                 groups=1,
                 bias='auto',
                 output_size=None,  # determins output_padding
                 fully_deconv=False,
                 conv_cfg={'type': 'conv'},
                 deconv_cfg={'type': 'deconv'},
                 norm_cfg={'type': 'in'},
                 acti_cfg={'type': 'relu'},
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'acti')):
        super().__init__()
        self.conv_cfg = conv_cfg
        self.deconv_cfg = deconv_cfg
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
        self.output_size = output_size
        self.fully_deconv = fully_deconv

        if padding == 'auto':
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]

        if self.with_explicit_pad:
            pad_cfg = {'type': padding_mode}
            self.pad_layer = create_pad_layer(pad_cfg, padding)
            conv_padding = 0
        else:
            conv_padding = padding

        if fully_deconv or stride > 1:
            output_padding = compute_output_padding(output_size, kernel_size, stride, conv_padding, dilation=dilation)
            self.conv = create_deconv_layer(deconv_cfg, in_channels, out_channels, output_padding=output_padding,
                                            kernel_size=kernel_size, stride=stride, padding=conv_padding,
                                            dilation=dilation, groups=groups, bias=self.with_bias)
        else:
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
            if acti_cfg['type'] not in ('tanh', 'prelu', 'sigmoid', 'hardsigmoid', 'hsigmoid', 'swish', 'clamp', 'silu'):
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
