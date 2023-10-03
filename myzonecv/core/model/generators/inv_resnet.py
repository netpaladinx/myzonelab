import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

from ...registry import GENERATORS, BLOCKS
from ...utils import pair
from ..base_module import BaseModule, Sequential
from ..initializers import kaiming_init, constant_init
from ..bricks import create_conv_layer, create_deconv_layer, infer_norm_name, create_norm_layer, compute_output_padding, DeconvBlock


@BLOCKS.register_class('inv_resnet_basicblock')
class InvResNetBasicBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,  # determines mid_channels
                 stride=1,
                 dilation=1,
                 output_size=None,  # determins output_padding
                 upsample=None,
                 fully_deconv=False,
                 with_cp=False,
                 conv_cfg=dict(type='conv'),
                 deconv_cfg=dict(type='deconv'),
                 norm_cfg=dict(type='in'),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        assert expansion == 1 and out_channels % expansion == 0
        self.out_channels = out_channels
        self.expansion = expansion
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.output_size = output_size
        self.fully_deconv = fully_deconv
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.deconv_cfg = deconv_cfg
        self.norm_cfg = norm_cfg

        if fully_deconv or stride > 1:
            output_padding = compute_output_padding(output_size, 3, stride, dilation, dilation=dilation)
            self.conv1 = create_deconv_layer(deconv_cfg, in_channels, self.mid_channels, output_padding=output_padding,
                                             kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        else:
            self.conv1 = create_conv_layer(conv_cfg, in_channels, self.mid_channels,
                                           kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)

        self.norm1_name = infer_norm_name(norm_cfg) + '1'
        norm1 = create_norm_layer(norm_cfg, self.mid_channels)
        self.add_module(self.norm1_name, norm1)

        if fully_deconv:
            self.conv2 = create_deconv_layer(deconv_cfg, self.mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        else:
            self.conv2 = create_conv_layer(conv_cfg, self.mid_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.norm2_name = infer_norm_name(norm_cfg) + '2'
        norm2 = create_norm_layer(norm_cfg, out_channels)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample

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

            if self.upsample is not None:
                identity = self.upsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@BLOCKS.register_class('inv_resnet_bottleneck')
class InvResNetBottleneck(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,    # determines mid_channels
                 stride=1,
                 dilation=1,
                 output_size=None,  # determins output_padding
                 upsample=None,
                 fully_deconv=False,
                 with_cp=False,
                 conv_cfg=dict(type='conv'),
                 deconv_cfg=dict(type='deconv'),
                 norm_cfg=dict(type='in'),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        assert out_channels % expansion == 0
        self.out_channels = out_channels
        self.expansion = expansion
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.output_size = output_size
        self.fully_deconv = fully_deconv
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.deconv_cfg = deconv_cfg
        self.norm_cfg = norm_cfg

        if fully_deconv:
            self.conv1 = create_deconv_layer(deconv_cfg, in_channels, self.mid_channels, kernel_size=1, stride=1, bias=False)
        else:
            self.conv1 = create_conv_layer(conv_cfg, in_channels, self.mid_channels, kernel_size=1, stride=1, bias=False)

        self.norm1_name = infer_norm_name(norm_cfg) + '1'
        norm1 = create_norm_layer(norm_cfg, self.mid_channels)
        self.add_module(self.norm1_name, norm1)

        if fully_deconv or stride > 1:
            output_padding = compute_output_padding(output_size, 3, stride, dilation, dilation=dilation)
            self.conv2 = create_deconv_layer(deconv_cfg, self.mid_channels, self.mid_channels, output_padding=output_padding,
                                             kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        else:
            self.conv2 = create_conv_layer(conv_cfg, self.mid_channels, self.mid_channels,
                                           kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)

        self.norm2_name = infer_norm_name(norm_cfg) + '2'
        norm2 = create_norm_layer(norm_cfg, self.mid_channels)
        self.add_module(self.norm2_name, norm2)

        if fully_deconv:
            self.conv3 = create_deconv_layer(deconv_cfg, self.mid_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.conv3 = create_conv_layer(conv_cfg, self.mid_channels, out_channels, kernel_size=1, bias=False)

        self.norm3_name = infer_norm_name(norm_cfg) + '3'
        norm3 = create_norm_layer(norm_cfg, out_channels)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample

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

            if self.upsample is not None:
                identity = self.upsample(x)

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
        elif issubclass(block, InvResNetBasicBlock):
            expansion = 1
        elif issubclass(block, InvResNetBottleneck):
            expansion = 4
        else:
            raise TypeError(f"expansion is not specified for {block.__name__}")
    else:
        raise TypeError('expansion must be an integer or None')
    return expansion


@BLOCKS.register_class('inv_resnet_layer')
class InvResNetLayer(Sequential):
    def __init__(self,
                 block,
                 n_blocks,
                 in_channels,
                 out_channles,
                 expansion=None,
                 stride=1,
                 dilation=1,
                 output_size=None,  # determins output_padding
                 direct_upsample=False,
                 upsample_first=True,
                 multigrid_dilations=None,
                 contract_first_dilation=False,
                 fully_deconv=False,
                 conv_cfg=dict(type='conv'),
                 deconv_cfg=dict(type='deconv'),
                 norm_cfg=dict(type='in'),
                 **kwargs):
        self.block = block
        self.n_blocks = n_blocks
        self.in_channels = in_channels
        self.out_channels = out_channles
        self.expansion = get_expansion(block, expansion)
        self.stride = stride
        self.output_size = output_size
        self.direct_upsample = direct_upsample
        self.upsample_first = upsample_first
        self.multigrid_dilations = multigrid_dilations
        self.contract_first_dilation = contract_first_dilation
        self.conv_cfg = conv_cfg
        self.deconv_cfg = deconv_cfg
        self.norm_cfg = norm_cfg

        upsample = None
        if stride != 1 or in_channels != out_channles:
            upsample = []
            conv_stride = stride
            if direct_upsample and stride != 1:
                conv_stride = 1
                upsample.append(nn.Upsample(scale_factor=stride, mode='nearest'))
            if fully_deconv or conv_stride > 1:
                output_padding = compute_output_padding(output_size, 1, conv_stride, 0)
                upsample.append(create_deconv_layer(deconv_cfg, in_channels, out_channles, output_padding=output_padding,
                                                    kernel_size=1, stride=conv_stride, padding=0, bias=False))
            else:
                upsample.append(create_conv_layer(conv_cfg, in_channels, out_channles, kernel_size=1, stride=conv_stride, bias=False))
            upsample.append(create_norm_layer(norm_cfg, out_channles))
            upsample = Sequential(*upsample)

        grid_dilations = [dilation] * n_blocks
        if multigrid_dilations is None:
            if dilation > 1 and contract_first_dilation:
                grid_dilations[0] = dilation // 2
        else:
            grid_dilations = multigrid_dilations

        layers = []
        if upsample_first:
            layers.append(block(in_channels, out_channles, expansion=self.expansion,
                                stride=stride, dilation=grid_dilations[0], output_size=output_size, upsample=upsample, fully_deconv=fully_deconv,
                                conv_cfg=conv_cfg, deconv_cfg=deconv_cfg, norm_cfg=norm_cfg, **kwargs))
            for i in range(1, n_blocks):
                layers.append(block(out_channles, out_channles, expansion=self.expansion,
                                    stride=1, dilation=grid_dilations[i], fully_deconv=fully_deconv,
                                    conv_cfg=conv_cfg, deconv_cfg=deconv_cfg, norm_cfg=norm_cfg, **kwargs))
        else:
            for i in range(0, n_blocks - 1):
                layers.append(block(in_channels, in_channels, expansion=self.expansion,
                                    stride=1, dilation=grid_dilations[i], fully_deconv=fully_deconv,
                                    conv_cfg=conv_cfg, deconv_cfg=deconv_cfg, norm_cfg=norm_cfg, **kwargs))
            layers.append(block(in_channels, out_channles, expansion=self.expansion,
                                stride=stride, dilation=grid_dilations[-1], output_size=output_size, upsample=upsample, fully_deconv=fully_deconv,
                                conv_cfg=conv_cfg, deconv_cfg=deconv_cfg, norm_cfg=norm_cfg, **kwargs))

        super().__init__(*layers, init_cfg=kwargs.get('init_cfg'))


def get_output_size(image_size, scale_factor=1):
    image_size = pair(image_size)
    scale_factor = pair(scale_factor)
    return (image_size[0] // scale_factor[0], image_size[1] // scale_factor[1])


@GENERATORS.register_class('inv_resnet')
class InvResNet(BaseModule):
    _allowed_archs = {
        18: (InvResNetBasicBlock, (2, 2, 2, 2)),
        34: (InvResNetBasicBlock, (3, 6, 4, 3)),
        50: (InvResNetBottleneck, (3, 6, 4, 3)),
        101: (InvResNetBottleneck, (3, 23, 4, 3)),
        152: (InvResNetBottleneck, (3, 36, 8, 3))
    }

    def __init__(self,
                 depth,
                 in_channels,
                 output_size,  # fixed image size (h, w)
                 out_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 n_stages=4,
                 strides=(2, 2, 2, 1),
                 dilations=(1, 1, 1, 1),
                 direct_upsample=False,
                 frozen_stages=-1,
                 with_cp=False,
                 norm_eval=False,
                 zero_init_residual=True,
                 multigrid_dilations=None,
                 contract_first_dilation=False,
                 fully_deconv=False,
                 conv_cfg=dict(type='conv'),
                 deconv_cfg=dict(type='deconv'),
                 norm_cfg=dict(type='in'),
                 out_acti_cfg=dict(type='tanh'),
                 out_autoscale=False,
                 out_autobias=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        assert depth in self._allowed_archs, f"Invalid depth {depth} for resnet"
        self.depth = depth
        self.in_channels = in_channels
        self.output_size = output_size
        self.out_channels = out_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.expansion = expansion
        assert 1 <= n_stages <= 4
        self.n_stages = n_stages
        assert len(strides) == len(dilations) == n_stages
        self.strides = strides
        self.dilations = dilations
        self.direct_upsample = direct_upsample
        self.frozen_stages = frozen_stages
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.fully_deconv = fully_deconv
        self.conv_cfg = conv_cfg
        self.deconv_cfg = deconv_cfg
        self.norm_cfg = norm_cfg
        self.out_acti_cfg = out_acti_cfg
        self.out_autoscale = out_autoscale
        self.out_autobias = out_autobias
        self.block, stage_blocks = self._allowed_archs[depth]
        self.stage_blocks = stage_blocks[:n_stages]
        self.expansion = get_expansion(self.block, expansion)

        # contributes prod(strides) stride
        self.layer_names = []
        res_in_channels = in_channels
        res_out_channels = base_channels * self.expansion * 2**(n_stages - 2)
        for i, n_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            # one stage one resnet_layer
            res_layer = InvResNetLayer(self.block, n_blocks, res_in_channels, res_out_channels, output_size=get_output_size(output_size, 2**(n_stages - i - 1)),
                                       expansion=self.expansion, stride=stride, dilation=dilation, direct_upsample=direct_upsample,
                                       multigrid_dilations=multigrid_dilations, contract_first_dilation=contract_first_dilation, fully_deconv=fully_deconv,
                                       with_cp=with_cp, conv_cfg=conv_cfg, deconv_cfg=deconv_cfg, norm_cfg=norm_cfg)
            res_in_channels = res_out_channels
            res_out_channels = res_out_channels // 2
            layer_name = 'layer' + str(i + 1)
            self.add_module(layer_name, res_layer)
            self.layer_names.append(layer_name)
        res_out_channels = res_in_channels

        self.stem = Sequential(
            DeconvBlock(res_out_channels, stem_channels,
                        fully_deconv=fully_deconv, kernel_size=3, stride=1, padding=1,
                        conv_cfg=conv_cfg, deconv_cfg=deconv_cfg, norm_cfg=norm_cfg),
            DeconvBlock(stem_channels, stem_channels,
                        fully_deconv=fully_deconv, kernel_size=3, stride=1, padding=1,
                        conv_cfg=conv_cfg, deconv_cfg=deconv_cfg, norm_cfg=norm_cfg),
            DeconvBlock(stem_channels, stem_channels, output_size=output_size,
                        fully_deconv=fully_deconv, kernel_size=3, stride=2, padding=1,
                        conv_cfg=conv_cfg, deconv_cfg=deconv_cfg, norm_cfg=norm_cfg),
            DeconvBlock(stem_channels, out_channels, output_size=output_size,
                        fully_deconv=fully_deconv, kernel_size=1,
                        conv_cfg=conv_cfg, deconv_cfg=deconv_cfg, norm_cfg=None, acti_cfg=out_acti_cfg))

        self.out_scale = nn.parameter.Parameter(torch.zeros(3, 1, 1)) if out_autoscale else None
        self.out_bias = nn.parameter.Parameter(torch.zeros(3, 1, 1)) if out_autobias else None

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def freeze_stages(self, frozen_stages=-1):
        frozen_stages = self.frozen_stages if frozen_stages == -1 else frozen_stages
        if frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
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
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                oc, ic = m.weight.shape[:2]
                kaiming_init(m, mode='fan_out' if oc >= ic else 'fan_in')
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, InvResNetBasicBlock):
                    constant_init(m.norm2, 0)
                elif isinstance(m, InvResNetBottleneck):
                    constant_init(m.norm3, 0)

        if self.out_acti_cfg is None:
            nonlinearity = 'linear'
        elif self.out_acti_cfg['type'] in ('sigmoid', 'sigmoid_ms'):
            nonlinearity = 'sigmoid'
        else:
            nonlinearity = self.out_acti_cfg['type']
        kaiming_init(self.stem[-1].conv, mode='fan_in', nonlinearity=nonlinearity)

    def forward(self, x):
        for i, layer_name in enumerate(self.layer_names):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

        x = self.stem(x)

        if self.out_scale is not None:
            x = x * torch.exp(self.out_scale)

        if self.out_bias is not None:
            x = x + self.out_bias

        return x

    def train(self, mode=True):
        super().train(mode)
        self.freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


@GENERATORS.register_class('inv_resnets16')
class InvResNetS16(InvResNet):
    """ InvResNetS16 changes stage 1's stride from 2 to 1 to make the global stride 16.
    """

    def __init__(self, *args, **kwargs):
        kwargs.update({'strides': (1, 2, 2, 1)})
        super().__init__(*args, **kwargs)
