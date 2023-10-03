import math

import torch
import torch.nn as nn

from ...registry import HEADS, BLOCKS
from ...utils import make_divisible
from ..initializers import normal_init
from ..base_module import BaseModule, Sequential, ModuleList


@BLOCKS.register_class('yolov5_detect6')
class Yolov5Detect6(BaseModule):
    """ bboxes (len: 5): x0, y0, w, h, conf
        classes (coco: 80): person, ...
        anchors: list(list), n_layers x (n_anchors*2)
        layers: 80x80, 40x40, 20x20
        strides: 8, 16, 32
    """
    default_bbox_dims = 5
    default_avg_objs = 8
    default_prob_all_cls = 0.6

    def __init__(self,
                 in_channels,
                 n_classes,
                 anchors,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.out_dims = n_classes + self.default_bbox_dims
        assert len(in_channels) == len(anchors)
        self.in_layers = len(in_channels)
        self.anchors = anchors
        self.n_anchors = len(anchors[0]) // 2
        self.out_channels = self.n_anchors * self.out_dims

        self.convs = ModuleList(nn.Conv2d(c, self.out_channels, 1) for c in in_channels)

    def forward(self, xs):
        outs = []
        for i in range(self.in_layers):
            out = self.convs[i](xs[i])
            # if input size is 640 x 640, then here:
            #   bs,out_channels,80,80 => bs,3,80,80,out_dims (e.g. out_channels: 255, out_dims: 85)
            #   bs,out_channels,40,40 => bs,3,40,40,out_dims
            #   bs,out_channels,20,20 => bs,3,20,20,out_dims
            bs, _, h, w = out.shape
            out = out.view(bs, self.n_anchors, self.out_dims, h, w).permute(0, 1, 3, 4, 2).contiguous()  # bs,n_anchors,h,w,out_dims
            outs.append(out)
        return outs

    def init_strides(self, strides):
        self.strides = strides

    def init_weights(self):
        init_cfg = {}
        if self.init_cfg and isinstance(self.init_cfg, (list, tuple)):
            init_cfg = self.init_cfg[0]
        input_size = init_cfg.pop('input_size', 640)
        input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        avg_objs = init_cfg.pop('avg_objs', self.default_avg_objs)
        prob_all_cls = init_cfg.pop('prob_all_cls', self.default_prob_all_cls)

        super().init_weights()

        # initialize biases (https://arxiv.org/abs/1708.02002 section 3.3)
        for conv, stride in zip(self.convs, self.strides):
            bias = conv.bias.view(self.n_anchors, -1)
            bias.data[:, 4] += math.log(avg_objs / ((input_size[0] / stride) * (input_size[1] / stride)))  # objectness
            bias.data[:, 5:] += math.log(prob_all_cls / (self.n_classes - .99))  # classes
            conv.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)


@HEADS.register_class('yolov5_head6')
class Yolov5Head6(BaseModule):
    detect_block_type = 'yolov5_detect6'

    def __init__(self,
                 backbone,
                 arch,
                 depth_multiple=1,
                 width_multiple=1,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.arch = arch
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.layer_offset = len(backbone.layers)

        layers = []
        branches = set()
        n_layers = len(arch)
        for i in range(n_layers):
            layer_cfg = arch[f'layer{i+1}'].copy()

            block = layer_cfg.pop('block')
            n_blocks = self.get_num_blocks(layer_cfg, depth_multiple)
            n_inner_blocks = self.get_num_inner_blocks(layer_cfg, depth_multiple)

            input_index = layer_cfg.pop('input_index', -1)
            in_channels = self.get_in_channels(layers, input_index, backbone.layers)
            out_channels = self.get_out_channels(layer_cfg, width_multiple, in_channels)

            kwargs = {'type': block}
            if block == self.detect_block_type:
                kwargs['in_channels'] = in_channels
            elif block not in ('upsample_layer.upsample', 'concat'):
                kwargs['in_channels'] = in_channels
                kwargs['out_channels'] = out_channels

            kwargs.update(layer_cfg)

            if n_inner_blocks > 0:
                kwargs['n_inner_blocks'] = n_inner_blocks

            if n_blocks == 1:
                layer = BLOCKS.create(kwargs)
            else:
                layer = Sequential(*[BLOCKS.create(kwargs) for _ in range(n_blocks)])

            layer.n_params = sum([p.numel() for p in layer.parameters()])
            layer.block = block
            layer.input_index = input_index
            layer.index = i
            layer.out_channels = out_channels
            layers.append(layer)

            input_index = [input_index] if isinstance(input_index, int) else input_index
            input_index = [idx for idx in input_index if idx != -1 and idx != i - 1 + self.layer_offset]
            branches.update(input_index)

        self.layers = ModuleList(layers)
        self.branches = branches

    @staticmethod
    def get_num_blocks(layer_cfg, depth_multiple, force_change=False):
        n_blocks = layer_cfg.pop('n_blocks', 1)
        if n_blocks == 1 and not force_change:
            return n_blocks
        else:
            return max(round(n_blocks * depth_multiple), 1)

    @staticmethod
    def get_num_inner_blocks(layer_cfg, depth_multiple, force_change=False):
        n_inner_blocks = layer_cfg.pop('n_inner_blocks', 0)
        if n_inner_blocks > 1 or force_change:
            n_inner_blocks = max(round(n_inner_blocks * depth_multiple), 1)
        return n_inner_blocks

    @staticmethod
    def get_in_channels(layers, input_index, backbone_layers):
        def _get_ch(i):
            m = len(backbone_layers)
            n = len(layers)
            if i < 0:
                i = m + n + i
            if i < m:
                return backbone_layers[i].out_channels
            else:
                return layers[i - m].out_channels

        if isinstance(input_index, (list, tuple)):
            return [_get_ch(idx) for idx in input_index]
        else:
            return _get_ch(input_index)

    @staticmethod
    def get_out_channels(layer_cfg, width_multiple, in_channels):
        if 'n_channels' in layer_cfg:
            n_channels = layer_cfg.pop('n_channels')
            return make_divisible(n_channels * width_multiple, 8)
        else:
            return sum(in_channels) if isinstance(in_channels, (list, tuple)) else in_channels

    def forward(self, x, **kwargs):
        branches = kwargs.get('backbone_outputs', {})
        out = x
        for i, layer in enumerate(self.layers):
            input_index = layer.input_index
            if isinstance(input_index, (list, tuple)):
                x = [out if idx == -1 or idx == i - 1 + self.layer_offset else branches[idx] for idx in input_index]
            else:
                x = out if input_index == -1 or input_index == i - 1 else branches[input_index]

            out = layer(x)

            if i + self.layer_offset in self.branches:
                branches[i + self.layer_offset] = out

        return out

    def init_strides(self, strides):
        assert self.layers[-1].block == self.detect_block_type
        self.layers[-1].init_strides(strides)
