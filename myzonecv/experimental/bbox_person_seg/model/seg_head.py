import torch
import torch.nn as nn
import torch.nn.functional as F

from myzonecv.core.model import BaseModule, Sequential, ModuleList
from myzonecv.core.model.bricks import ConvBlock
from myzonecv.core.registry import BLOCKS
from ..registry import SEG_HEADS


class BaseSegHead(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 input_transform=None,  # default, 'resize_concat', 'multiple_select'
                 input_index=0,
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 acti_cfg=None,
                 init_cfg=None):
        """ 1. `input_transform` and `input_index` are used when inputs is multiple
            2. `align_corners` is used when multiple inputs of different sizes need to be concatenated,
               but default is False when resizing
        """
        super().__init__(init_cfg)
        self.out_channels = out_channels
        self.align_corners = align_corners
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.acti_cfg = acti_cfg

        self.in_channels, self.input_transform, self.input_index = \
            self.check_input_transform(in_channels, input_transform, input_index)

    def check_input_transform(self, in_channels, input_transform, input_index):
        if input_transform is not None:
            assert input_transform in ('resize_concat', 'multiple_select')
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(input_index, (list, tuple))
            assert len(in_channels) == len(input_index)
            if input_transform == 'resize_concat':
                in_channels = sum(in_channels)
        else:
            assert isinstance(in_channels, int)
            assert isinstance(input_index, int) or input_index is None
        return in_channels, input_transform, input_index

    def transform_inputs(self, x):
        if not isinstance(x, (list, tuple)):
            return x

        if self.input_transform == 'resize_concat':
            h, w = x[0].shape[2:]  # x[0] is the reference tensor
            xs = [F.interpolate(x[i], size=(h, w), mode='bilinear', align_corners=self.align_corners)
                  for i in self.input_index]
            x = torch.cat(xs, dim=1)  # bs x sum(c) x h x w
        elif self.input_transform == 'multipe_select':
            x = [x[i] for i in self.input_index]
        else:
            x = x[self.input_index]
        return x

    @property
    def n_classes(self):
        return self.out_channels


@BLOCKS.register_class('aspp')
class ASPP(ModuleList):
    """ Atrous Spatial Pyramid Pooling (ASPP)

        arch: [conv_1x1, dilated_conv_3x3, dilated_conv_3x3, dilated_conv_3x3] 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilations,
                 conv_cfg=None,
                 norm_cfg=None,
                 acti_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilations = dilations
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.acti_cfg = acti_cfg

        for dilation in self.dilations:
            kernel_size = 1 if dilation == 1 else 3
            padding = 0 if dilation == 1 else dilation  # (kernel_size - 1) * dilation / 2
            conv_block = ConvBlock(self.in_channels, self.out_channels, kernel_size, dilation=dilation, padding=padding,
                                   conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, acti_cfg=self.acti_cfg)
            self.append(conv_block)

    def forward(self, x):
        outs = []
        for mod in self:
            out = mod(x)
            outs.append(out)
        return outs


@SEG_HEADS.register_class('aspp_head')
class SegASPPHead(BaseSegHead):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=512,
                 input_transform=None,
                 input_index=-1,
                 align_corners=False,
                 dilations=(1, 6, 12, 18),
                 imagepool_out_size=1,
                 imagepool_kernel=1,
                 bottleneck_kernel=3,
                 final_dropout=0.1,
                 final_kernel=1,
                 conv_cfg={'type': 'conv'},
                 norm_cfg={'type': 'bn'},
                 acti_cfg={'type': 'relu'},
                 init_cfg={'type': 'normal', 'std': 0.01, 'child': 'conv_seg'}):
        super().__init__(in_channels, out_channels, input_transform, input_index, align_corners, conv_cfg, norm_cfg, acti_cfg, init_cfg)
        self.mid_channels = mid_channels
        self.dilations = dilations
        self.imagepool_out_size = imagepool_out_size
        self.imagepool_kernel = imagepool_kernel
        self.bottleneck_kernel = bottleneck_kernel
        self.final_dropout = final_dropout
        self.final_kernel = final_kernel

        self.aspp_modules = ASPP(self.in_channels, self.mid_channels, self.dilations,
                                 conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, acti_cfg=self.acti_cfg)

        self.image_pool = Sequential(
            nn.AdaptiveAvgPool2d(self.imagepool_out_size),
            ConvBlock(self.in_channels, self.mid_channels, self.imagepool_kernel,
                      conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, acti_cfg=self.acti_cfg)
        )

        all_mid_channels = self.mid_channels * (1 + len(self.dilations))  # from image_pool and assp_modules
        self.bottleneck = ConvBlock(all_mid_channels, self.mid_channels, self.bottleneck_kernel,
                                    conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, acti_cfg=self.acti_cfg)

        self.dropout = nn.Dropout2d(self.final_dropout) if self.final_dropout > 0 else None

        self.conv_seg = nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=self.final_kernel, padding=self.final_kernel // 2)

    def forward(self, x):
        x = self.transform_inputs(x)
        h, w = x.shape[2:]

        pool_out = self.image_pool(x)
        pool_out = F.interpolate(pool_out, size=(h, w), mode='bilinear', align_corners=self.align_corners)

        aspp_outs = self.aspp_modules(x)

        outs = [pool_out] + aspp_outs
        out = torch.cat(outs, dim=1)
        out = self.bottleneck(out)

        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv_seg(out)
        return out


@SEG_HEADS.register_class('fcn_head')
class SegFCNHead(BaseSegHead):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 input_transform=None,
                 input_index=-1,
                 align_corners=False,
                 kernel_size=3,
                 dilation=1,
                 n_convs=2,
                 input_shortcut=True,
                 final_dropout=0.1,
                 final_kernel=1,
                 conv_cfg={'type': 'conv'},
                 norm_cfg={'type': 'bn'},
                 acti_cfg={'type': 'relu'},
                 init_cfg={'type': 'normal', 'std': 0.01, 'child': 'conv_seg'}):
        super().__init__(in_channels, out_channels, input_transform, input_index, align_corners, conv_cfg, norm_cfg, acti_cfg, init_cfg)
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.n_convs = n_convs
        self.input_shortcut = input_shortcut
        self.final_dropout = final_dropout
        self.final_kernel = final_kernel

        if self.n_convs > 0:
            padding = (self.kernel_size // 2) * self.dilation
            convs = [ConvBlock(self.in_channels, self.mid_channels, self.kernel_size, padding=padding, dilation=self.dilation,
                               conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, acti_cfg=self.acti_cfg)]
            for _ in range(self.n_convs - 1):
                convs.append(ConvBlock(self.mid_channels, self.mid_channels, self.kernel_size, padding=padding, dilation=self.dilation,
                                       conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, acti_cfg=self.acti_cfg))
            self.convs = Sequential(*convs)
        else:
            self.convs = nn.Identity()

        if self.input_shortcut:
            padding = self.kernel_size // 2
            self.conv_cat = ConvBlock(self.in_channels + self.mid_channels, self.mid_channels, self.kernel_size, padding=padding,
                                      conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, acti_cfg=self.acti_cfg)

        self.dropout = nn.Dropout2d(self.final_dropout) if self.final_dropout > 0 else None

        self.conv_seg = nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=self.final_kernel, padding=self.final_kernel // 2)

    def forward(self, x):
        x = self.transform_inputs(x)
        out = self.convs(x)
        if self.input_shortcut:
            out = self.conv_cat(torch.cat([x, out], dim=1))

        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv_seg(out)
        return out
