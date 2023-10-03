from collections import OrderedDict

import torch
import torch.nn as nn

from ....registry import Registry, HEADS, BLOCKS, GENERATORS
from ...base_module import BaseModule, ModuleList, create_sequential_if_list, create_modulelist_if_list
from ...initializers import normal_init

REID_BLOCKS = Registry('reid_block')
REID_BLOCKS.init(create_modulelist_if_list)


@REID_BLOCKS.register_class('direct_pool')
class DirectPool(BaseModule):
    def __init__(self, output_size=(1, 1), use_avgpool=True, use_maxpool=False):
        super().__init__()
        self.output_size = output_size
        assert use_avgpool or use_maxpool
        self.use_avgpool = use_avgpool
        self.use_maxpool = use_maxpool

        if use_avgpool:
            self.avgpool = nn.AdaptiveAvgPool2d(output_size)
        if use_maxpool:
            self.maxpool = nn.AdaptiveMaxPool2d(output_size)
        if use_avgpool and use_maxpool:
            self.alpha = nn.parameter.Parameter(torch.tensor(0.))

    def forward(self, x):
        """ x: B x C x H x W
        """
        if self.use_avgpool and self.use_maxpool:
            a = torch.sigmoid(self.alpha)
            return self.avgpool(x) * a + self.maxpool(x) * (1 - a)  # B x C x output_size[0] x output_size[1]

        elif self.use_avgpool:
            return self.avgpool(x)

        elif self.use_maxpool:
            return self.maxpool(x)


@REID_BLOCKS.register_class('attention_pool')
class AttentionPool(BaseModule):
    _versions = {
        1: {'attn_conv_bias': True, 'attn_acti_cfg': {'type': 'sigmoid'}},
        2: {'attn_conv_bias': False, 'attn_acti_cfg': {'type': 'relu'}},
        3: {'attn_conv_bias': False, 'attn_acti_cfg': {'type': 'relu'}, 'attn_out_channels': 8},
        4: {'attn_conv_bias': True, 'attn_acti_cfg': {'type': 'sigmoid'}, 'attn_out_channels': 8},
    }

    def __init__(self, in_channels, mid_channels, depth=0, quantile_factor=None, version=1, aggr_method='sum', enable_temperature=False, temperature4channel=False):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.depth = depth
        self.quantile_factor = quantile_factor
        assert version in (1, 2, 3, 4)
        self.version = version
        assert aggr_method in ('sum', 'mean', 'max')
        self.aggr_method = aggr_method
        self.enable_temperature = enable_temperature
        self.temperature4channel = temperature4channel

        if version in (1, 2):
            self.projs = BLOCKS.create([{'type': 'conv_block', 'in_channels': in_channels if i == 0 else mid_channels, 'out_channels': mid_channels,
                                         'kernel_size': (1, 1), 'norm_cfg': None}
                                        for i in range(depth)]) if depth > 0 else None
            self.attn = BLOCKS.create({'type': 'conv_block', 'in_channels': mid_channels if depth > 0 else in_channels, 'out_channels': 1,
                                       'kernel_size': (1, 1), 'bias': self._versions[version]['attn_conv_bias'],
                                       'norm_cfg': None, 'acti_cfg': self._versions[version]['attn_acti_cfg']})
            self.attn_channels = 1
        elif version in (3, 4):
            self.depth = 0
            self.attn = BLOCKS.create({'type': 'conv_block', 'in_channels': in_channels, 'out_channels': self._versions[version]['attn_out_channels'],
                                       'kernel_size': (1, 1), 'bias': self._versions[version]['attn_conv_bias'],
                                       'norm_cfg': None, 'acti_cfg': self._versions[version]['attn_acti_cfg']})
            self.attn_channels = self._versions[version]['attn_out_channels']

        self.alpha = None
        if enable_temperature:
            if self.attn_channels > 1 and temperature4channel:
                self.alpha = nn.parameter.Parameter(torch.zeros(self.attn_channels))
            else:
                self.alpha = nn.parameter.Parameter(torch.tensor(0.))

    def aggregate_attn(self, x):
        """ x: B x attn_out_channels x H x W 
        """
        out = x
        if self.alpha is not None and self.alpha.dim == 1:
            out = out.pow(torch.exp(self.alpha)[:, None, None])

        if self.attn_channels > 1:
            if self.aggr_method == 'sum':
                out = torch.sum(out, 1, keepdim=True)
            elif self.aggr_method == 'mean':
                out = torch.mean(x, 1, keepdim=True)
            elif self.aggr_method == 'max':
                out = torch.max(x, 1, keepdim=True)[0]

        if self.alpha is not None and self.alpha.dim == 0:
            out = out.pow(torch.exp(self.alpha))
        return out

    def forward(self, x, return_attention=False):
        """ x: B x C x H x W
        """
        out = x
        for i in range(self.depth):
            out = self.projs[i](out)  # B x C2 x H x W
        out = self.attn(out)  # B x (1 or C_attn) x H x W
        attn = self.aggregate_attn(out)

        if self.quantile_factor is not None:
            attn_detached = attn.detach()
            quantile = torch.quantile(attn_detached.reshape(attn_detached.shape[0], 1, -1), self.quantile_factor, dim=2)  # B x 1
            attn = attn * (attn >= quantile[..., None, None])

        out = torch.sum(x * attn, (2, 3), keepdim=True) / torch.clamp(torch.sum(attn, (2, 3), keepdim=True), min=1e-8)  # B x C x 1 x 1
        return (out, attn) if return_attention else out

    def init_weights(self):
        super().init_weights()
        normal_init(self.attn.conv, std=0.02)


@REID_BLOCKS.register_class('single_route')
class SingleRoute(BaseModule):
    def __init__(self, in_channels, out_channels, depth=0, shortcut=False, pool_first=True, pool_cfg=None, init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.shortcut = shortcut
        self.pool_first = pool_first
        self.pool_cfg = pool_cfg or {}

        self.proj = BLOCKS.create({'type': 'conv_block', 'in_channels': in_channels, 'out_channels': out_channels,
                                   'kernel_size': (1, 1), 'norm_cfg': None, 'acti_cfg': None})
        self.projs = BLOCKS.create([{'type': 'conv_block', 'in_channels': out_channels, 'out_channels': out_channels,
                                     'kernel_size': (1, 1), 'norm_cfg': None, 'order': ('acti', 'conv', 'norm')}
                                    for _ in range(depth)]) if depth > 0 else None
        self.pool = REID_BLOCKS.create(self.pool_cfg)

    def forward(self, x):
        """ x: B x C x H x W
        """
        if self.pool_first:
            out = self.pool(x)  # B x C x 1 x 1
            out = self.proj(out)  # B x C2 x 1 x 1
            for i in range(self.depth):
                if self.shortcut:
                    out = self.projs[i](out) + out
                else:
                    out = self.projs[i](out)  # B x C2 x 1 x 1

        else:
            out = self.proj(x)  # B x C2 x H x W
            for i in range(self.depth):
                if self.shortcut:
                    out = self.projs[i](out) + out
                else:
                    out = self.projs[i](out)  # B x C2 x H x W
            out = self.pool(out)  # B x C2 x 1 x 1
        return out


@REID_BLOCKS.register_class('route_attention')
class RouteAttention(BaseModule):
    def __init__(self, n_dims, n_stops, pool_first=True, enable_temperature=True, pool_cfg=None):
        super().__init__()
        self.n_dims = n_dims
        self.n_stops = n_stops
        self.pool_first = pool_first
        self.enable_temperature = enable_temperature
        self.pool_cfg = pool_cfg or {}

        self.proj = BLOCKS.create({'type': 'conv_block', 'in_channels': n_dims, 'out_channels': n_dims,
                                   'kernel_size': (1, 1), 'norm_cfg': None, 'order': ('acti', 'conv', 'norm')})
        self.pool = DirectPool(**self.pool_cfg)
        self.weight = nn.parameter.Parameter(torch.randn(n_dims, n_stops) * 0.02)
        self.alpha = nn.parameter.Parameter(torch.tensor(0.)) if enable_temperature else None
        self.cossim = nn.CosineSimilarity()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """ x: B x C x H x W
        """
        if self.pool_first:
            out = self.pool(x)
            out = self.proj(out)
        else:
            out = self.proj(x)
            out = self.pool(out)

        out = out.reshape(out.shape[0], -1)  # B x C
        if self.alpha is None:
            out = self.cossim(out.unsqueeze(2), self.weight.unsqueeze(0))  # B x n_stops
        else:
            out = self.cossim(out.unsqueeze(2), self.weight.unsqueeze(0)) * torch.exp(self.alpha)  # B x n_stops
        out = self.softmax(out)
        return out  # B x n_stops


@REID_BLOCKS.register_class('multi_route')
class MultiRoute(BaseModule):
    def __init__(self, in_channels, out_channels, stops, shortcut=False, attn_pool_first=True,
                 pool_first=True, pool_index=None, pool_cfg=None, init_cfg=None):
        """ stops: list(int), i.e. [3, 3, 3]
        """
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stops = stops
        self.depth = len(stops)
        assert self.depth > 0
        self.shortcut = shortcut
        self.attn_pool_first = attn_pool_first
        self.pool_first = pool_first
        if pool_index is not None:
            assert 0 <= pool_index <= self.depth
        self.pool_index = pool_index
        self.pool_cfg = pool_cfg or {}

        self.proj = BLOCKS.create({'type': 'conv_block', 'in_channels': in_channels, 'out_channels': out_channels,
                                   'kernel_size': (1, 1), 'norm_cfg': None, 'acti_cfg': None})
        self.projs = BLOCKS.create([{'type': 'conv_block', 'in_channels': out_channels, 'out_channels': out_channels,
                                     'kernel_size': (1, 1), 'norm_cfg': None, 'order': ('acti', 'conv', 'norm'), 'inplace': False}
                                    for n_stops in self.stops for _ in range(n_stops)])
        self.attns = REID_BLOCKS.create([{'type': 'route_attention', 'n_dims': out_channels, 'n_stops': n_stops, 'pool_first': attn_pool_first}
                                         for n_stops in stops])
        self.pool = REID_BLOCKS.create(self.pool_cfg)

    def forward(self, x):
        """ x: B x C x H x W
        """
        if self.pool_index is None:
            if self.pool_first:
                out = self.pool(x)  # B x C x 1 x 1
                out = self.proj(out)  # B x C2 x 1 x 1
                base = 0
                for i, n_stops in enumerate(self.stops):
                    attn = self.attns[i](out)  # B x n_stops
                    if self.shortcut:
                        out = sum([self.projs[base + j](out) * attn[:, j][:, None, None, None] for j in range(n_stops)]) + out
                    else:
                        out = sum([self.projs[base + j](out) * attn[:, j][:, None, None, None] for j in range(n_stops)])
                    base += n_stops
            else:
                out = self.proj(x)  # B x C2 x H x W
                base = 0
                for i, n_stops in enumerate(self.stops):
                    attn = self.attns[i](out)  # B x n_stops
                    if self.shortcut:
                        out = sum([self.projs[base + j](out) * attn[:, j][:, None, None, None] for j in range(n_stops)]) + out
                    else:
                        out = sum([self.projs[base + j](out) * attn[:, j][:, None, None, None] for j in range(n_stops)])
                    base += n_stops
                out = self.pool(out)  # B x C2 x 1 x 1
        else:
            out = self.proj(x)  # B x C2 x H x W
            base = 0
            for i, n_stops in enumerate(self.stops):
                if self.pool_index == i:
                    out = self.pool(out)  # B x C2 x 1 x 1
                attn = self.attns[i](out)
                if self.shortcut:
                    out = sum([self.projs[base + j](out) * attn[:, j][:, None, None, None] for j in range(n_stops)]) + out
                else:
                    out = sum([self.projs[base + j](out) * attn[:, j][:, None, None, None] for j in range(n_stops)])
                base += n_stops
            if self.pool_index == self.depth:
                out = self.pool(out)  # B x C2 x 1 x 1
        return out  # B x C2 x 1 x 1


class BaseExtractor(BaseModule):
    def __init__(self, in_channels, out_channels, init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

    @property
    def n_features(self):
        return 1


@REID_BLOCKS.register_class('global_pool_extractor')
class GlobalPoolExtractor(BaseExtractor):
    """ direct_pool + single_route """

    def __init__(self,
                 in_channels,
                 out_channels,
                 depth=0,
                 shortcut=False,
                 pool_first=True,
                 use_avgpool=True,
                 use_maxpool=False,
                 concat_global=False,
                 init_cfg=None):
        super().__init__(in_channels, out_channels, init_cfg)
        self.depth = depth
        self.shortcut = shortcut
        self.pool_first = pool_first
        self.use_avgpool = use_avgpool
        self.use_maxpool = use_maxpool
        self.concat_global = concat_global
        if concat_global:
            self.in_channels *= 2

        self.pool = self.pool = DirectPool() if concat_global else None
        self.enc = SingleRoute(self.in_channels, out_channels, depth=depth, shortcut=shortcut, pool_first=pool_first,
                               pool_cfg={'type': 'direct_pool', 'use_avgpool': use_avgpool, 'use_maxpool': use_maxpool})

    def forward(self, x):
        if self.concat_global:
            x = torch.concat((x, self.pool(x).expand(*x.shape)), dim=1)
        out = self.enc(x)
        return out


@REID_BLOCKS.register_class('global_attention_extractor')
class GlobalAttentionExtractor(BaseExtractor):
    """ attention_pool + single_route """

    def __init__(self,
                 in_channels,
                 out_channels,
                 depth=0,
                 shortcut=False,
                 attn_depth=0,
                 quantile_factor=None,
                 pool_first=True,
                 concat_global=False,
                 attn_version=1,
                 init_cfg=None):
        super().__init__(in_channels, out_channels, init_cfg)
        self.depth = depth
        self.shortcut = shortcut
        self.attn_depth = attn_depth
        self.pool_first = pool_first
        self.concat_global = concat_global
        if concat_global:
            self.in_channels *= 2
        self.attn_version = attn_version

        self.pool = self.pool = DirectPool() if concat_global else None
        self.enc = SingleRoute(self.in_channels, out_channels, depth=depth, shortcut=shortcut, pool_first=pool_first,
                               pool_cfg={'type': 'attention_pool', 'in_channels': self.in_channels if pool_first else out_channels,
                                         'mid_channels': out_channels, 'depth': attn_depth, 'quantile_factor': quantile_factor, 'version': attn_version})

    def forward(self, x):
        if self.concat_global:
            x = torch.concat((x, self.pool(x).expand(*x.shape)), dim=1)
        out = self.enc(x)
        return out


@REID_BLOCKS.register_class('global_multiple_attention_extractor')
class GlobalMultipleAttentionExtractor(BaseExtractor):
    """ (attention_pool + single_route) x n """
    _versions = {
        1: {},
        2: {'quantile_factors': (0, 0.75, 0.8889, 0.8889, 0.9375, 0.9375, 0.9375), 'pool_first': True},
        3: {'quantile_factors': (0.75, 0.8889, 0.8889, 0.9375, 0.9375, 0.9375), 'pool_first': True},
        4: {'quantile_factors': (0, 0.75, 0.75, 0.8889, 0.8889, 0.8889, 0.9375, 0.9375, 0.9375, 0.9375), 'pool_first': True},
        5: {'quantile_factors': (0.75, 0.75, 0.8889, 0.8889, 0.8889, 0.9375, 0.9375, 0.9375, 0.9375), 'pool_first': True},
        6: {'multiple': 3, 'pool_first': True},
        7: {'multiple': 5, 'pool_first': True},
        8: {'multiple': 7, 'pool_first': True},
    }

    def __init__(self,
                 in_channels,
                 out_channels,
                 depth=0,
                 shortcut=False,
                 pool_first=True,
                 attn_depth=0,
                 quantile_factors=(0, 0.5, 0.75, 0.875, 0.9375),
                 multiple=None,
                 concat_global=False,
                 version=1,
                 attn_version=1,
                 attn_aggr_method='sum',
                 enable_temperature=False,
                 temperature4channel=False,
                 init_cfg=None):
        super().__init__(in_channels, out_channels, init_cfg)
        self.depth = depth
        self.shortcut = shortcut
        self.pool_first = pool_first
        self.attn_depth = attn_depth
        self.quantile_factors = quantile_factors
        self.multiple = len(quantile_factors) if quantile_factors else multiple
        self.concat_global = concat_global
        if concat_global:
            self.in_channels *= 2
        assert version in (1, 2, 3, 4, 5, 6, 7, 8)
        self.version = version
        if version in (2, 3, 4, 5):
            self.quantile_factors = self._versions[version]['quantile_factors']
            self.pool_first = self._versions[version]['pool_first']
            self.multiple = len(self.quantile_factors)
        elif version in (6, 7, 8):
            self.quantile_factors = ()
            self.pool_first = self._versions[version]['pool_first']
            self.multiple = self._versions[version]['multiple']
        self.attn_version = attn_version
        self.attn_aggr_method = attn_aggr_method
        self.enable_temperature = enable_temperature
        self.temperature4channel = temperature4channel

        self.pool = self.pool = DirectPool() if concat_global else None
        if self.quantile_factors:
            self.enc = REID_BLOCKS.create([{'type': 'single_route', 'in_channels': self.in_channels, 'out_channels': out_channels,
                                            'depth': depth, 'shortcut': shortcut, 'pool_first': self.pool_first,
                                            'pool_cfg': {'type': 'attention_pool', 'in_channels': self.in_channels if self.pool_first else out_channels,
                                                         'mid_channels': out_channels, 'depth': attn_depth, 'quantile_factor': quantile_factor,
                                                         'version': attn_version, 'aggr_method': attn_aggr_method,
                                                         'enable_temperature': enable_temperature, 'temperature4channel': temperature4channel}}
                                           for quantile_factor in self.quantile_factors])
        else:
            self.enc = REID_BLOCKS.create([{'type': 'single_route', 'in_channels': self.in_channels, 'out_channels': out_channels,
                                            'depth': depth, 'shortcut': shortcut, 'pool_first': self.pool_first,
                                            'pool_cfg': {'type': 'attention_pool', 'in_channels': self.in_channels if self.pool_first else out_channels,
                                                         'mid_channels': out_channels, 'depth': attn_depth, 'quantile_factor': None,
                                                         'version': attn_version, 'aggr_method': attn_aggr_method,
                                                         'enable_temperature': enable_temperature, 'temperature4channel': temperature4channel}}
                                           for _ in range(self.multiple)])

    def forward(self, x):
        if self.concat_global:
            x = torch.concat((x, self.pool(x).expand(*x.shape)), dim=1)
        outs = []
        for mod in self.enc:
            outs.append(mod(x))
        return outs

    @property
    def n_features(self):
        return len(self.quantile_factors) if self.quantile_factors else self.multiple


@REID_BLOCKS.register_class('global_multiroute_extractor')
class GlobalMultiRouteExtractor(BaseExtractor):
    """ direct_pool + multi_route """
    _versions = {
        1: {},
        2: {'stops': [5]},
        3: {'stops': [7]},
        4: {'stops': [5, 5]},
        5: {'stops': [7, 7]},
        6: {'stops': [3, 7]},
        7: {'stops': [7, 3]},
    }

    def __init__(self,
                 in_channels,
                 out_channels,
                 stops=None,
                 shortcut=False,
                 pool_first=True,
                 pool_index=None,
                 attn_pool_first=True,
                 concat_global=False,
                 version=1,
                 init_cfg=None):
        super().__init__(in_channels, out_channels, init_cfg)
        self.stops = stops
        self.shortcut = shortcut
        self.pool_first = pool_first
        self.pool_index = pool_index
        self.attn_pool_first = attn_pool_first
        self.concat_global = concat_global
        if concat_global:
            self.in_channels *= 2
        self.version = version
        if version == 1:
            assert self.stops is not None
        else:
            assert version in (2, 3, 4, 5, 6, 7)
            self.stops = self._versions[version]['stops']

        self.pool = self.pool = DirectPool() if concat_global else None
        self.enc = MultiRoute(self.in_channels, out_channels, self.stops, shortcut=shortcut, attn_pool_first=attn_pool_first,
                              pool_first=pool_first, pool_index=pool_index, pool_cfg={'type': 'direct_pool'})

    def forward(self, x):
        if self.concat_global:
            x = torch.concat((x, self.pool(x).expand(*x.shape)), dim=1)
        out = self.enc(x)
        return out


@REID_BLOCKS.register_class('global_attention_multiroute_extractor')
class GlobalAttentionMultiRouteExtractor(BaseExtractor):
    """ attention_pool + multi_route """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stops,
                 pool_first=True,
                 attn_depth=0,
                 quantile_factor=None,
                 concat_global=False,
                 init_cfg=None):
        super().__init__(in_channels, out_channels, init_cfg)
        self.stops = stops
        self.pool_first = pool_first
        self.attn_depth = attn_depth
        self.quantile_factor = quantile_factor
        self.concat_global = concat_global
        if concat_global:
            self.in_channels *= 2

        self.pool = self.pool = DirectPool() if concat_global else None
        self.enc = MultiRoute(self.in_channels, out_channels, stops, pool_first=pool_first,
                              pool_cfg={'type': 'attention_pool', 'in_channels': self.in_channels if pool_first else out_channels,
                                        'mid_channels': out_channels, 'depth': attn_depth, 'quantile_factor': quantile_factor})

    def forward(self, x):
        if self.concat_global:
            x = torch.concat((x, self.pool(x).expand(*x.shape)), dim=1)
        out = self.enc(x)
        return out


@REID_BLOCKS.register_class('global_multiple_attention_multiroute_extractor')
class GlobalMultipleAttentionMultiRouteExtractor(BaseExtractor):
    """ (attention_pool + multi_route) x n """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stops,
                 pool_first=True,
                 attn_depth=0,
                 quantile_factors=(0, 0.5, 0.75, 0.875, 0.9375),
                 concat_global=False,
                 init_cfg=None):
        super().__init__(in_channels, out_channels, init_cfg)
        self.stops = stops
        self.pool_first = pool_first
        self.attn_depth = attn_depth
        self.quantile_factors = quantile_factors
        self.concat_global = concat_global
        if concat_global:
            self.in_channels *= 2

        self.pool = self.pool = DirectPool() if concat_global else None
        self.enc = REID_BLOCKS.create([{'type': 'multi_route', 'in_channels': self.in_channels, 'out_channels': out_channels, 'stops': stops, 'pool_first': pool_first,
                                        'pool_cfg': {'type': 'attention_pool', 'in_channels': self.in_channels if pool_first else out_channels,
                                                     'mid_channels': out_channels, 'depth': attn_depth, 'quantile_factor': quantile_factor}}
                                       for quantile_factor in quantile_factors])

    def forward(self, x):
        if self.concat_global:
            x = torch.concat((x, self.pool(x).expand(*x.shape)), dim=1)
        outs = []
        for mod in self.enc:
            outs.append(mod(x))
        return outs

    @property
    def n_features(self):
        return len(self.quantile_factors)


@REID_BLOCKS.register_class('horizontal_strips_extractor')
class HorizontalStripsExtractor(BaseExtractor):
    def __init__(self,
                 in_channels,
                 out_channels,
                 n_parts=6,
                 use_hierarchy=False,
                 init_cfg=None):
        """
        AdaptiveAvgPool2d: 16x12 => 6x1
            kernel size: (16+6-1) // 6 = 3
            start indices: np.linspace(0, 16-3, 6).round = [0, 3, 5, 8, 10, 13]
            ranges: [0, 2], [3, 5], [5, 7], [8, 10], [10, 12], [13, 15] (overlap at 5,10)
        """
        super().__init__(in_channels, out_channels, init_cfg)
        self.n_parts = n_parts
        self.use_hierarchy = use_hierarchy

        if not use_hierarchy:
            self.ranges = [(i, i + 1) for i in range(n_parts)]
        else:
            self.ranges = [(i, i + stride) for stride in range(1, n_parts) for i in range(n_parts - stride + 1)]

        self.pool = nn.AdaptiveAvgPool2d((n_parts, 1))
        self.pool2 = None if not use_hierarchy else nn.AdaptiveAvgPool2d((1, 1))
        self.enc = BLOCKS.create([{'type': 'conv_block', 'in_channels': in_channels, 'out_channels': out_channels,
                                  'kernel_size': (1, 1), 'norm_cfg': None, 'acti_cfg': None} for _ in range(len(self.ranges))])

    def forward(self, x):
        x = self.pool(x)
        xs = [x[:, :, a:b]for a, b in self.ranges]
        if self.use_hierarchy:
            xs = [self.pool2(x) for x in xs]
        xs = [self.enc[i](x) for i, x in enumerate(xs)]
        return xs

    @property
    def n_features(self):
        return len(self.ranges)


class BaseFinalAggregator(BaseModule):
    def __init__(self, in_channels, n_features, out_channels=None, init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.n_features = n_features
        self.out_channels = out_channels


@REID_BLOCKS.register_class('final_concat_aggregator')
class FinalConcatAggregator(BaseFinalAggregator):
    def __init__(self,
                 in_channels,
                 n_features,
                 out_channels=None,
                 depth=0,
                 init_cfg=None):
        super().__init__(in_channels, n_features, out_channels, init_cfg)
        self.depth = 0 if out_channels is None else depth
        self.mid_channels = (512 if in_channels > 512 and out_channels < 512 else out_channels) if self.depth > 0 else None

        if out_channels is not None:
            blocks = [{'type': 'conv_block', 'in_channels': self.in_channels, 'out_channels': self.mid_channels if self.depth > 0 else out_channels,
                       'kernel_size': (1, 1), 'norm_cfg': None, 'acti_cfg': None}]
            for i in range(depth):
                blocks.append({'type': 'conv_block', 'in_channels': self.mid_channels, 'out_channels': out_channels if i == depth - 1 else self.mid_channels,
                               'kernel_size': (1, 1), 'norm_cfg': None, 'order': ('acti', 'conv', 'norm')})
            self.enc = BLOCKS.create(blocks, create_func=create_sequential_if_list)
        else:
            self.enc = None

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == self.n_features
        x = torch.cat(x, dim=1) if len(x) > 1 else x[0]
        if self.enc is not None:
            x = self.enc(x)
        return x


@HEADS.register_class('reid_head')
class ReIDHead(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 final_channels=None,
                 global_extractor={'type': 'global_pool_extractor', 'abbr': 'gpool'},
                 local_extractor=None,
                 final_aggregator={'type': 'final_concat_aggregator', 'abbr': 'fcat'},
                 init_cfg=None):
        super().__init__(init_cfg)
        self.global_extractor_cfg = global_extractor
        self.local_extractor_cfg = local_extractor
        self.final_aggregator_cfg = final_aggregator

        self.global_extractor = REID_BLOCKS.create(self.global_extractor_cfg,
                                                   (in_channels,) if out_channels is None else (in_channels, out_channels),
                                                   _ignore=('name', 'abbr'))
        self.local_extractor = REID_BLOCKS.create(self.local_extractor_cfg,
                                                  (in_channels, out_channels),
                                                  _ignore=('name', 'abbr'))
        self.final_aggregator = REID_BLOCKS.create({'out_channels': final_channels, **self.final_aggregator_cfg},
                                                   (self.feature_channels, self.n_features),
                                                   _ignore=('name', 'abbr'))

    @property
    def n_features(self):
        n = 0

        if isinstance(self.global_extractor, ModuleList):
            for mod in self.global_extractor:
                n += mod.n_features
        elif self.global_extractor is not None:
            n += self.global_extractor.n_features

        if isinstance(self.local_extractor, ModuleList):
            for mod in self.local_extractor:
                n += mod.n_features
        elif self.local_extractor is not None:
            n += self.local_extractor.n_features

        return n

    @property
    def feature_channels(self):
        n = 0

        if isinstance(self.global_extractor, ModuleList):
            for mod in self.global_extractor:
                n += mod.out_channels * mod.n_features
        elif self.global_extractor is not None:
            n += self.global_extractor.out_channels * mod.n_features

        if isinstance(self.local_extractor, ModuleList):
            for mod in self.local_extractor:
                n += mod.out_channels * mod.n_features
        elif self.local_extractor is not None:
            n += self.local_extractor.out_channels * mod.n_features

        return n

    def get_global_extractor(self):
        if isinstance(self.global_extractor, ModuleList):
            for cfg, mod in zip(self.global_extractor_cfg, self.global_extractor):
                yield cfg.get('name', cfg['type']), mod, cfg.get('abbr', cfg['type'])
        elif self.global_extractor is not None:
            cfg, mod = self.global_extractor_cfg, self.global_extractor
            yield cfg.get('name', cfg['type']), mod, cfg.get('abbr', cfg['type'])

    def get_local_extractor(self):
        if isinstance(self.local_extractor, ModuleList):
            for cfg, mod in zip(self.local_extractor_cfg, self.local_extractor):
                yield cfg.get('name', cfg['type']), mod, cfg.get('abbr', cfg['type'])
        elif self.local_extractor is not None:
            cfg, mod = self.local_extractor_cfg, self.local_extractor
            yield cfg.get('name', cfg['type']), mod, cfg.get('abbr', cfg['type'])

    def get_final_aggregator(self):
        cfg, mod = self.final_aggregator_cfg, self.final_aggregator
        return cfg.get('name', cfg['type']), mod, cfg.get('abbr', cfg['type'])

    def forward(self, x, return_all_features=False):
        """ x:  B x C (2048) x H (16) x W (12)
        """
        features = OrderedDict()

        for name, mod, _ in self.get_global_extractor():
            out = mod(x)
            if not isinstance(out, (list, tuple)):
                features[name] = out
            else:
                for i, o in enumerate(out):
                    features[f'{name}_{i}'] = o

        for name, mod, _ in self.get_local_extractor():
            out = mod(x)
            if not isinstance(out, (list, tuple)):
                features[name] = out
            else:
                for i, o in enumerate(out):
                    features[f'{name}_{i}'] = o

        name, mod, _ = self.get_final_aggregator()
        out = mod(list(features.values()))
        features[name] = out

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
