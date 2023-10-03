import torch.nn as nn

from ...registry import BACKBONES, BLOCKS
from ...utils import make_divisible
from ..base_module import BaseModule, Sequential, ModuleList


@BACKBONES.register_class('yolov5_backbone6')
class Yolov5Backbone6(BaseModule):
    def __init__(self,
                 arch,
                 in_channels=3,
                 depth_multiple=1,
                 width_multiple=1,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.arch = arch
        self.in_channels = in_channels
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple

        layers = []
        branches = set()
        n_layers = len(arch)
        for i in range(n_layers):
            layer_cfg = arch[f'layer{i+1}'].copy()

            block = layer_cfg.pop('block')
            n_blocks = self.get_num_blocks(layer_cfg, depth_multiple)
            n_inner_blocks = self.get_num_inner_blocks(layer_cfg, depth_multiple)

            input_index = layer_cfg.pop('input_index', -1)
            in_channels = self.get_in_channels(layers, input_index, in_channels)
            out_channels = self.get_out_channels(layer_cfg, width_multiple, in_channels)

            kwargs = {
                'type': block,
                'in_channels': in_channels,
                'out_channels': out_channels,
                **layer_cfg
            }

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
            input_index = [idx for idx in input_index if idx != -1 and idx != i - 1]
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
    def get_in_channels(layers, input_index, in_channels):
        if len(layers) > 0:
            if isinstance(input_index, (list, tuple)):
                in_channels = sum([layers[idx].out_channels for idx in input_index])
            else:
                in_channels = layers[input_index].out_channels
        return in_channels

    @staticmethod
    def get_out_channels(layer_cfg, width_multiple, in_channels):
        if 'n_channels' in layer_cfg:
            n_channels = layer_cfg.pop('n_channels')
            return make_divisible(n_channels * width_multiple, 8)
        else:
            return in_channels

    def forward(self, img, output_indices=None):
        branches = {}
        outputs = {}
        out = img
        for i, layer in enumerate(self.layers):
            input_index = layer.input_index
            if isinstance(input_index, (list, tuple)):
                x = [out if idx == -1 or idx == i - 1 else branches[idx] for idx in input_index]
            else:
                x = out if input_index == -1 or input_index == i - 1 else branches[input_index]
            out = layer(x)

            if i in self.branches:
                branches[i] = out

            if output_indices and (i in output_indices):
                outputs[i] = out

        return out, outputs if outputs else out
