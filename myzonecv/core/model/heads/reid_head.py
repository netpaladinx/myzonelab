from collections import OrderedDict

import torch
import torch.nn as nn

from ...registry import Registry, BLOCKS, HEADS, GENERATORS
from ...utils import tolist
from ..base_module import BaseModule, Sequential

REID_BLOCKS = Registry('reid_block')


@REID_BLOCKS.register_class('straightthrough_feature_extractor')
class StraightthroughFeatureExtractor(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 out_activation=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_activation in ('relu', 'gelu', None)
        self.out_activation = out_activation

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_proj = nn.Linear(in_channels, out_channels)
        if out_activation:
            self.out_acti = nn.ReLU() if out_activation == 'relu' else nn.GELU()

    def forward(self, x):
        """ x: B x C(=2048) x H x W
        """
        out = self.pool(x).reshape(x.shape[0], -1)  # B x C

        out = self.out_proj(out)

        if self.out_activation:
            out = self.out_acti(out)

        return out  # B x C2


@REID_BLOCKS.register_class('transformer_feature_extractor')
class TransformerFeatureExtractor(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 n_heads=8,
                 n_feedforward_dims=2048,
                 layernorm=False,
                 dropout=0.1,
                 activation='relu',
                 proj_bias=True,
                 proj_activation='relu',
                 out_projection=False,
                 out_activation='relu',
                 use_output_token=False):
        super().__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.n_feedforward_dims = n_feedforward_dims
        self.layernorm = layernorm
        self.dropout = dropout
        self.activation = activation
        self.proj_bias = proj_bias
        assert proj_activation in ('relu', 'gelu', None)
        self.proj_activation = proj_activation
        self.out_projection = out_projection
        assert out_activation in ('relu', 'gelu', None)
        self.out_activation = out_activation
        self.use_output_token = use_output_token

        layers = []
        for out_c in tolist(out_channels):
            if in_channels != out_c:
                layers.append(nn.Linear(in_channels, out_c, bias=self.proj_bias))
                if proj_activation:
                    layers.append(nn.ReLU() if proj_activation == 'relu' else nn.GELU())
                in_channels = out_c
            layers.append(BLOCKS.create({'type': 'transformer_layer',
                                         'n_dims': in_channels,
                                         'n_heads': n_heads,
                                         'n_feedforward_dims': n_feedforward_dims,
                                         'layernorm': layernorm,
                                         'dropout': dropout,
                                         'activation': activation,
                                         'batch_first': True}))

        self.layers = Sequential(*layers)

        if self.use_output_token:
            self.token = nn.parameter.Parameter(torch.randn(1, 1, self.in_channels) * 0.02)

        if out_projection:
            self.out_proj = nn.Linear(in_channels, in_channels)

        if out_activation:
            self.out_acti = nn.ReLU() if out_activation == 'relu' else nn.GELU()

        self.out_channels = in_channels

    def forward(self, x):
        """ x: B x C(=2048) x H x W
        """
        x = x.reshape(x.shape[0], self.in_channels, -1).transpose(1, 2)  # B x (H*W) x C

        if self.use_output_token:
            x = torch.cat((self.token.expand(x.shape[0], -1, -1), x), dim=1)  # B x (1+H*W) x C
            out = self.layers(x)
            out = out[:, 0]
        else:
            out = self.layers(x)
            out = out.mean(dim=1)

        if self.out_projection:
            out = self.out_proj(out)

        if self.out_activation:
            out = self.out_acti(out)

        return out  # B x C2


@HEADS.register_class('reid_head')
class ReIDHead(BaseModule):
    def __init__(self,
                 in_channels,
                 feature_extractor=None,
                 mask_first_channels=0,
                 dropout=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.feature_extractor_cfg = feature_extractor
        self.mask_first_channels = mask_first_channels

        self.feature_extractor = REID_BLOCKS.create(self.feature_extractor_cfg, (in_channels,), _ignore=('name', 'abbr'))
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

        self.out_channels = self.feature_extractor.out_channels

    def forward(self, x, return_all_features=False):
        features = OrderedDict()

        if self.mask_first_channels > 0:
            mask = torch.ones(x.shape[:2]).to(x)
            mask[:, :self.mask_first_channels] = 0
            x = x * mask[..., None, None]

        if self.dropout is not None:
            x = self.dropout(x)
        out = self.feature_extractor(x)

        features['output'] = out
        return (out, features) if return_all_features else out


@HEADS.register_class('reid_recon_head')
class ReIDReconHead(BaseModule):
    def __init__(self,
                 in_channels,
                 output_size,  # input image size (h, w)
                 generator=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.output_size = output_size
        self.generator_cfg = generator

        self.generator = GENERATORS.create({'type': 'inv_resnet',
                                            'depth': 50,
                                            'in_channels': in_channels,
                                            'output_size': output_size,
                                            **(generator or {})})

    def forward(self, x):
        x = self.generator(x)
        return x
