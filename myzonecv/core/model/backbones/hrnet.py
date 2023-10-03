import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from ...registry import BACKBONES, BLOCKS
from ..base_module import BaseModule, Sequential, ModuleList
from ..initializers import constant_init, kaiming_init
from ..bricks import create_conv_layer, infer_norm_name, create_norm_layer, create_upsample_layer
from .resnet import get_expansion, ResNetBasicBlock, ResNetBottleneck


def create_hrnet_branch(block, n_blocks, in_channels, out_channels, stride, with_cp, conv_cfg, norm_cfg):
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = Sequential(
            create_conv_layer(conv_cfg, in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            create_norm_layer(norm_cfg, out_channels))

    layers = []
    layers.append(block(in_channels, out_channels, stride=stride, downsample=downsample, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
    for _ in range(1, n_blocks):
        layers.append(block(out_channels, out_channels, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg))

    return Sequential(*layers)


def create_hrnet_branch_fuse(out_branch, n_branches, in_channels, conv_cfg, norm_cfg, upsample_cfg):
    layers = []
    for i in range(n_branches):
        if i > out_branch:
            layers.append(Sequential(
                create_conv_layer(conv_cfg, in_channels[i], in_channels[out_branch], kernel_size=1, stride=1, padding=0, bias=False),
                create_norm_layer(norm_cfg, in_channels[out_branch]),
                create_upsample_layer(upsample_cfg, scale_factor=2**(i - out_branch))))
        elif i == out_branch:
            layers.append(None)
        else:
            downsamples = []  # conv downsample
            for k in range(out_branch - i):
                if k == out_branch - i - 1:
                    downsamples.append(Sequential(
                        create_conv_layer(conv_cfg, in_channels[i], in_channels[out_branch], kernel_size=3, stride=2, padding=1, bias=False),
                        create_norm_layer(norm_cfg, in_channels[out_branch])))
                else:
                    downsamples.append(Sequential(
                        create_conv_layer(conv_cfg, in_channels[i], in_channels[i], kernel_size=3, stride=2, padding=1, bias=False),
                        create_norm_layer(norm_cfg, in_channels[i]),
                        nn.ReLU(inplace=True)))
            layers.append(Sequential(*downsamples))
    return ModuleList(layers)


@BLOCKS.register_class('hrnet_layer')
class HRNetLayer(BaseModule):
    def __init__(self,
                 n_branches,
                 block,
                 n_blocks,
                 in_channels,
                 n_channels,
                 multiscale_output=False,
                 with_cp=False,
                 conv_cfg={'type': 'conv'},
                 norm_cfg={'type': 'bn'},
                 upsample_cfg={'type': 'upsample', 'mode': 'nearest', 'align_corners': None},
                 init_cfg=None):
        super().__init__(init_cfg)
        assert n_branches == len(n_blocks)
        assert n_branches == len(in_channels)
        assert n_branches == len(n_channels)
        self.n_branches = n_branches
        self.block = block
        self.n_blocks = n_blocks
        self.in_channels = in_channels.copy()
        self.n_channels = n_channels
        self.multiscale_output = multiscale_output
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.upsample_cfg = upsample_cfg

        branches = []
        for i in range(n_branches):
            br_n_blocks = n_blocks[i]
            br_in_channels = in_channels[i]
            br_out_channels = n_channels[i] * get_expansion(block)
            stride = 1
            branches.append(create_hrnet_branch(block, br_n_blocks, br_in_channels, br_out_channels, stride, with_cp, conv_cfg, norm_cfg))
            in_channels[i] = br_out_channels
        self.branches = ModuleList(branches)

        self.fuse_layers = None
        if n_branches > 1:
            fuse_layers = []
            out_branches = n_branches if multiscale_output else 1
            for i in range(out_branches):
                fuse_layers.append(create_hrnet_branch_fuse(i, n_branches, in_channels, conv_cfg, norm_cfg, upsample_cfg))
            self.fuse_layers = ModuleList(fuse_layers)
        self.out_channels = in_channels

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.n_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.n_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.n_branches):
                if i == j:
                    y += x[j]
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


def create_hrnet_stage1_layer(block, n_blocks, in_channels, out_channels, with_cp, conv_cfg, norm_cfg):
    downsample = None
    if in_channels != out_channels:
        downsample = Sequential(
            create_conv_layer(conv_cfg, in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            create_norm_layer(norm_cfg, out_channels))

    layers = []
    layers.append(block(in_channels, out_channels, stride=1, downsample=downsample, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
    for _ in range(1, n_blocks):
        layers.append(block(out_channels, out_channels, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
    return Sequential(*layers)


def create_hrnet_stage_transition(in_channels, out_channels, conv_cfg, norm_cfg):
    layers = []
    in_branches = len(in_channels)
    out_branches = len(out_channels)
    for i in range(out_branches):
        if i < in_branches:
            if out_channels[i] != in_channels[i]:
                layers.append(Sequential(
                    create_conv_layer(conv_cfg, in_channels[i], out_channels[i], kernel_size=3, stride=1, padding=1, bias=False),
                    create_norm_layer(norm_cfg, out_channels[i]),
                    nn.ReLU(inplace=True)))
            else:
                layers.append(None)
        else:
            downsamples = []  # conv downsample
            for j in range(i + 1 - in_branches):
                in_chs = in_channels[-1]
                out_chs = out_channels[i] if j == i - in_branches else in_chs
                downsamples.append(Sequential(
                    create_conv_layer(conv_cfg, in_chs, out_chs, kernel_size=3, stride=2, padding=1, bias=False),
                    create_norm_layer(norm_cfg, out_chs),
                    nn.ReLU(inplace=True)))
            layers.append(Sequential(*downsamples))
    return ModuleList(layers)


def create_hrnet_stage(n_layers, n_branches, block, n_blocks, in_channels, n_channels, with_cp, conv_cfg, norm_cfg, upsample_cfg,
                       multiscale_output=True):
    layers = []
    for i in range(n_layers):
        layer_multiscale_output = multiscale_output if i == n_layers - 1 else True
        layers.append(HRNetLayer(n_branches, block, n_blocks, in_channels, n_channels, multiscale_output=layer_multiscale_output,
                                 with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, upsample_cfg=upsample_cfg))
        in_channels = layers[-1].out_channels
    stage = Sequential(*layers)
    stage.out_channels = in_channels
    return stage


@BACKBONES.register_class('hrnet')
class HRNet(BaseModule):
    """
    Example:
        arch = {
            'stage1': {
                'n_layers': 1, 
                'n_branches': 1, 
                'block': 'resnet_bottleneck', 
                'n_blocks': (4,), 
                'n_channels': (64,)
            }, 
            'stage2': {
                'n_layers': 1, 
                'n_branches': 2, 
                'block': 'resnet_basicblock', 
                'n_blocks': (4, 4), 
                'n_channels': (32, 64)
            }, 
            'stage3': {
                'n_layers': 4, 
                'n_branches': 3, 
                'block': 'resnet_basicblock', 
                'n_blocks': (4, 4, 4), 
                'n_channels': (32, 64, 128)
            }, 
            'stage4': {
                'n_layers': 3, 
                'n_branches': 4, 
                'block': 'resnet_basicblock', 
                'n_blocks': (4, 4, 4, 4), 
                'n_channels': (32, 64, 128, 256)
            } 
        }
    """

    def __init__(self,
                 arch,
                 in_channels=3,
                 with_cp=False,
                 norm_eval=False,
                 zero_init_residual=False,
                 conv_cfg={'type': 'conv'},
                 norm_cfg={'type': 'bn'},
                 upsample_cfg={'type': 'upsample', 'mode': 'nearest', 'align_corners': None},
                 init_cfg=None):
        super().__init__(init_cfg)
        self.arch = arch
        self.in_channels = in_channels
        self.base_features = 64
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.upsample_cfg = upsample_cfg

        # stem
        self.conv1 = create_conv_layer(conv_cfg, in_channels, self.base_features, kernel_size=3, stride=2, padding=1, bias=False)

        self.norm1_name = infer_norm_name(norm_cfg) + '1'
        norm1 = create_norm_layer(norm_cfg, self.base_features)
        self.add_module(self.norm1_name, norm1)

        self.conv2 = create_conv_layer(conv_cfg, self.base_features, self.base_features, kernel_size=3, stride=2, padding=1, bias=False)

        self.norm2_name = infer_norm_name(norm_cfg) + '2'
        norm2 = create_norm_layer(norm_cfg, self.base_features)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)

        # one stage one or more hrnet_layer
        # stage 1
        self.stage1_cfg = arch['stage1']
        block_type = self.stage1_cfg['block']
        block = BLOCKS.get(block_type)
        n_blocks = self.stage1_cfg['n_blocks'][0]
        in_channels = self.base_features
        out_channels = self.stage1_cfg['n_channels'][0] * get_expansion(block)
        self.layer1 = create_hrnet_stage1_layer(block, n_blocks, in_channels, out_channels, with_cp, conv_cfg, norm_cfg)

        # stage 2
        self.stage2_cfg = arch['stage2']
        block_type = self.stage2_cfg['block']
        block = BLOCKS.get(block_type)
        in_channels = [out_channels]
        n_channels = self.stage2_cfg['n_channels']
        out_channels = [nc * get_expansion(block) for nc in n_channels]
        self.transition1 = create_hrnet_stage_transition(in_channels, out_channels, conv_cfg, norm_cfg)

        n_layers = self.stage2_cfg['n_layers']
        n_branches = self.stage2_cfg['n_branches']
        n_blocks = self.stage2_cfg['n_blocks']
        in_channels = out_channels
        self.stage2 = create_hrnet_stage(n_layers, n_branches, block, n_blocks, in_channels, n_channels,
                                         with_cp, conv_cfg, norm_cfg, self.upsample_cfg)

        # stage 3
        self.stage3_cfg = arch['stage3']
        block_type = self.stage3_cfg['block']
        block = BLOCKS.get(block_type)
        in_channels = self.stage2.out_channels
        n_channels = self.stage3_cfg['n_channels']
        out_channels = [nc * get_expansion(block) for nc in n_channels]
        self.transition2 = create_hrnet_stage_transition(in_channels, out_channels, conv_cfg, norm_cfg)

        n_layers = self.stage3_cfg['n_layers']
        n_branches = self.stage3_cfg['n_branches']
        n_blocks = self.stage3_cfg['n_blocks']
        in_channels = out_channels
        self.stage3 = create_hrnet_stage(n_layers, n_branches, block, n_blocks, in_channels, n_channels,
                                         with_cp, conv_cfg, norm_cfg, self.upsample_cfg)

        # stage 4
        self.stage4_cfg = arch['stage4']
        block_type = self.stage4_cfg['block']
        block = BLOCKS.get(block_type)
        in_channels = self.stage3.out_channels
        n_channels = self.stage4_cfg['n_channels']
        out_channels = [nc * get_expansion(block) for nc in n_channels]
        self.transition3 = create_hrnet_stage_transition(in_channels, out_channels, conv_cfg, norm_cfg)

        n_layers = self.stage4_cfg['n_layers']
        n_branches = self.stage4_cfg['n_branches']
        n_blocks = self.stage4_cfg['n_blocks']
        in_channels = out_channels
        multiscale_output = self.stage4_cfg.get('multiscale_output', False)
        self.stage4 = create_hrnet_stage(n_layers, n_branches, block, n_blocks, in_channels, n_channels,
                                         with_cp, conv_cfg, norm_cfg, self.upsample_cfg, multiscale_output=multiscale_output)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super().init_weights()
        if isinstance(self.init_cfg, dict) and self.init_cfg['type'] == 'pretrained':
            return

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
                # normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetBasicBlock):
                    constant_init(m.norm2, 0)
                elif isinstance(m, ResNetBottleneck):
                    constant_init(m.norm3, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        xs = []
        for i in range(self.stage2_cfg['n_branches']):
            transition = self.transition1[i]
            xs.append(transition(x) if transition else x)
        ys = self.stage2(xs)

        xs = []
        for i in range(self.stage3_cfg['n_branches']):
            transition = self.transition2[i]
            xs.append(transition(ys[i] if i < len(ys) else ys[-1]) if transition else ys[i])
        ys = self.stage3(xs)

        xs = []
        for i in range(self.stage4_cfg['n_branches']):
            transition = self.transition3[i]
            xs.append(transition(ys[i] if i < len(ys) else ys[-1]) if transition else ys[i])
        ys = self.stage4(xs)

        return ys

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
