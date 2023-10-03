import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

from ...registry import BACKBONES, BLOCKS
from ..base_module import BaseModule, Sequential
from ..initializers import kaiming_init, constant_init
from ..bricks import create_conv_layer, infer_norm_name, create_norm_layer, ConvBlock


@BLOCKS.register_class('resnet_basicblock')
class ResNetBasicBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 with_cp=False,
                 conv_cfg=dict(type='conv'),
                 norm_cfg=dict(type='bn'),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        assert expansion == 1 and out_channels % expansion == 0
        self.out_channels = out_channels
        self.expansion = expansion
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.conv1 = create_conv_layer(conv_cfg, in_channels, self.mid_channels,
                                       kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)

        self.norm1_name = infer_norm_name(norm_cfg) + '1'
        norm1 = create_norm_layer(norm_cfg, self.mid_channels)
        self.add_module(self.norm1_name, norm1)

        self.conv2 = create_conv_layer(conv_cfg, self.mid_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.norm2_name = infer_norm_name(norm_cfg) + '2'
        norm2 = create_norm_layer(norm_cfg, out_channels)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@BLOCKS.register_class('resnet_bottleneck')
class ResNetBottleneck(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 with_cp=False,
                 conv_cfg=dict(type='conv'),
                 norm_cfg=dict(type='bn'),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        assert out_channels % expansion == 0
        self.out_channels = out_channels
        self.expansion = expansion
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.conv1 = create_conv_layer(conv_cfg, in_channels, self.mid_channels, kernel_size=1, stride=1, bias=False)

        self.norm1_name = infer_norm_name(norm_cfg) + '1'
        norm1 = create_norm_layer(norm_cfg, self.mid_channels)
        self.add_module(self.norm1_name, norm1)

        self.conv2 = create_conv_layer(conv_cfg, self.mid_channels, self.mid_channels,
                                       kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)

        self.norm2_name = infer_norm_name(norm_cfg) + '2'
        norm2 = create_norm_layer(norm_cfg, self.mid_channels)
        self.add_module(self.norm2_name, norm2)

        self.conv3 = create_conv_layer(conv_cfg, self.mid_channels, out_channels, kernel_size=1, bias=False)

        self.norm3_name = infer_norm_name(norm_cfg) + '3'
        norm3 = create_norm_layer(norm_cfg, out_channels)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def get_expansion(block, expansion=None):
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, ResNetBasicBlock):
            expansion = 1
        elif issubclass(block, ResNetBottleneck):
            expansion = 4
        else:
            raise TypeError(f"expansion is not specified for {block.__name__}")
    else:
        raise TypeError('expansion must be an integer or None')
    return expansion


@BLOCKS.register_class('resnet_layer')
class ResNetLayer(Sequential):
    def __init__(self,
                 block,
                 n_blocks,
                 in_channels,
                 out_channles,
                 expansion=None,
                 stride=1,
                 dilation=1,
                 avg_downsample=False,
                 downsample_first=True,
                 multigrid_dilations=None,
                 contract_first_dilation=False,
                 conv_cfg=dict(type='conv'),
                 norm_cfg=dict(type='bn'),
                 **kwargs):
        self.block = block
        self.n_blocks = n_blocks
        self.in_channels = in_channels
        self.out_channels = out_channles
        self.expansion = get_expansion(block, expansion)
        self.stride = stride
        self.avg_downsample = avg_downsample
        self.downsample_first = downsample_first
        self.multigrid_dilations = multigrid_dilations
        self.contract_first_dilation = contract_first_dilation
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        downsample = None
        if stride != 1 or in_channels != out_channles:
            downsample = []
            conv_stride = stride
            if avg_downsample and stride != 1:
                conv_stride = 1
                downsample.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
            downsample.append(create_conv_layer(conv_cfg, in_channels, out_channles, kernel_size=1, stride=conv_stride, bias=False))
            downsample.append(create_norm_layer(norm_cfg, out_channles))
            downsample = Sequential(*downsample)

        grid_dilations = [dilation] * n_blocks
        if multigrid_dilations is None:
            if dilation > 1 and contract_first_dilation:
                grid_dilations[0] = dilation // 2
        else:
            grid_dilations = multigrid_dilations

        layers = []
        if downsample_first:
            layers.append(block(in_channels, out_channles, expansion=self.expansion,
                                stride=stride, dilation=grid_dilations[0], downsample=downsample,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
            for i in range(1, n_blocks):
                layers.append(block(out_channles, out_channles, expansion=self.expansion,
                                    stride=1, dilation=grid_dilations[i], conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
        else:
            for i in range(0, n_blocks - 1):
                layers.append(block(in_channels, in_channels, expansion=self.expansion,
                                    stride=1, dilation=grid_dilations[i], conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
            layers.append(block(in_channels, out_channles, expansion=self.expansion,
                                stride=stride, dilation=grid_dilations[-1], downsample=downsample,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))

        super().__init__(*layers, init_cfg=kwargs.get('init_cfg'))


@BACKBONES.register_class('resnet')
class ResNet(BaseModule):
    _allowed_archs = {
        18: (ResNetBasicBlock, (2, 2, 2, 2)),
        34: (ResNetBasicBlock, (3, 4, 6, 3)),
        50: (ResNetBottleneck, (3, 4, 6, 3)),
        101: (ResNetBottleneck, (3, 4, 23, 3)),
        152: (ResNetBottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 n_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_stages=(4,),  # starts at 1
                 deep_stem=False,
                 avg_downsample=False,
                 frozen_stages=-1,
                 with_cp=False,
                 norm_eval=False,
                 zero_init_residual=True,
                 multigrid_dilations=None,
                 contract_first_dilation=False,
                 conv_cfg=dict(type='conv'),
                 norm_cfg=dict(type='bn', requires_grad=True),
                 init_cfg=None):
        super().__init__(init_cfg)
        assert depth in self._allowed_archs, f"Invalid depth {depth} for resnet"
        self.depth = depth
        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.expansion = expansion
        assert 1 <= n_stages <= 4
        self.n_stages = n_stages
        assert len(strides) == len(dilations) == n_stages
        self.strides = strides
        self.dilations = dilations
        assert max(out_stages) <= n_stages
        self.out_stages = out_stages
        self.deep_stem = deep_stem
        self.avg_downsample = avg_downsample
        self.frozen_stages = frozen_stages
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.block, stage_blocks = self._allowed_archs[depth]
        self.stage_blocks = stage_blocks[:n_stages]
        self.expansion = get_expansion(self.block, expansion)

        # contributes 2 stride
        if self.deep_stem:
            self.stem = Sequential(
                ConvBlock(in_channels, stem_channels // 2,
                          kernel_size=3, stride=2, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, inplace=True),
                ConvBlock(stem_channels // 2, stem_channels // 2,
                          kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, inplace=True),
                ConvBlock(stem_channels // 2, stem_channels,
                          kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, inplace=True))
        else:
            self.conv1 = create_conv_layer(conv_cfg, in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.norm1_name = infer_norm_name(norm_cfg) + '1'
            norm1 = create_norm_layer(norm_cfg, stem_channels)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)

        # contributes 2 stride
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # contributes prod(strides) stride
        self.layer_names = []
        res_in_channels = stem_channels
        res_out_channels = base_channels * self.expansion
        for i, n_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            # one stage one resnet_layer
            res_layer = ResNetLayer(self.block, n_blocks, res_in_channels, res_out_channels, expansion=self.expansion,
                                    stride=stride, dilation=dilation, avg_downsample=avg_downsample,
                                    multigrid_dilations=multigrid_dilations, contract_first_dilation=contract_first_dilation,
                                    with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
            res_in_channels = res_out_channels
            res_out_channels *= 2
            layer_name = 'layer' + str(i + 1)
            self.add_module(layer_name, res_layer)
            self.layer_names.append(layer_name)

        self.out_channels = res_in_channels

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def freeze_stages(self, frozen_stages=-1):
        frozen_stages = self.frozen_stages if frozen_stages == -1 else frozen_stages
        if frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, frozen_stages + 1):
            m = getattr(self, 'layer' + str(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super().init_weights()
        if isinstance(self.init_cfg, dict) and self.init_cfg['type'] == 'pretrained':
            return

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetBasicBlock):
                    constant_init(m.norm2, 0)
                elif isinstance(m, ResNetBottleneck):
                    constant_init(m.norm3, 0)

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.layer_names):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i + 1 in self.out_stages:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self.freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


@BACKBONES.register_class('resnetv1c')
class ResNetV1c(ResNet):
    """ Compared with default ResNet (i.e. ResNetV1b), ResNetV1c replaces the 7x7 conv
        in the input stem with three 3x3 convs. (https://arxiv.org/abs/1812.01187) 
    """

    def __init__(self, *args, **kwargs):
        kwargs.update({'deep_stem': True, 'avg_downsample': False})
        super().__init__(*args, **kwargs)


@BACKBONES.register_class('resnetv1d')
class ResNetV1d(ResNet):
    """ Compared with default ResNet (i.e. ResNetV1b), ResNetV1d replaces the 7x7 conv
        in the input stem with three 3x3 convs. And in the downsampling block, a 2x2
        avg_pool with stride 2 is added before conv, whose stride is changed to 1. 
    """

    def __init__(self, *args, **kwargs):
        kwargs.update({'deep_stem': True, 'avg_downsample': True})
        super().__init__(*args, **kwargs)


@BACKBONES.register_class('resnets16')
class ResNetS16(ResNet):
    """ ResNetS16 changes stage 4's stride from 2 to 1 to make the global stride 16.
    """

    def __init__(self, *args, **kwargs):
        kwargs.update({'strides': (1, 2, 2, 1)})
        super().__init__(*args, **kwargs)
