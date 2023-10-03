from collections import OrderedDict

import torch
import torch.nn as nn

from ....registry import Registry, BLOCKS, HEADS
from ...base_module import BaseModule, ModuleList, create_sequential_if_list, create_modulelist_if_list
from ...initializers import kaiming_init

REID_BLOCKS_V2 = Registry('reid_block_v2')  # non-negative modeling
REID_BLOCKS_V2.init(create_modulelist_if_list)


@REID_BLOCKS_V2.register_class('avg_pool')
class AvgPool(BaseModule):
    def __init__(self, output_size=(1, 1)):
        super().__init__()
        self.output_size = output_size

        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        return self.pool(x)


@REID_BLOCKS_V2.register_class('attn_pool')
class AttnPool(BaseModule):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.attn = BLOCKS.create({'type': 'conv_block',
                                   'in_channels': in_channels,
                                   'out_channels': 1,
                                   'kernel_size': (1, 1),
                                   'bias': True,
                                   'norm_cfg': None,
                                   'acti_cfg': {'type': 'sigmoid'}})

    def forward(self, x, return_attention=False):
        a = self.attn(x)
        out = torch.sum(x * a, (2, 3), keepdim=True).div(torch.sum(a, (2, 3), keepdim=True).clamp(min=1e-10))
        return (out, a) if return_attention else out

    def init_weights(self):
        kaiming_init(self.attn.conv, a=0, mode='fan_in', nonlinearity='sigmoid', bias=0)


@REID_BLOCKS_V2.register_class('single_route')
class SingleRoute(BaseModule):
    def __init__(self, in_channels, out_channels, depth=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        self.trans = BLOCKS.create([{'type': 'conv_block',
                                     'in_channels': in_channels if i == 0 else out_channels,
                                     'out_channels': out_channels,
                                     'kernel_size': (1, 1),
                                     'bias': True,
                                     'norm_cfg': None}
                                    for i in range(depth + 1)],
                                   create_func=create_sequential_if_list)

    def forward(self, x):
        return self.trans(x)


@REID_BLOCKS_V2.register_class('route_attention')
class RouteAttention(BaseModule):
    def __init__(self, in_channels, n_routes):
        super().__init__()
        self.in_channels = in_channels
        self.n_routes = n_routes

        self.pool = AvgPool()
        self.attn = BLOCKS.create({'type': 'conv_block',
                                   'in_channels': in_channels,
                                   'out_channels': n_routes,
                                   'kernel_size': (1, 1),
                                   'bias': True,
                                   'norm_cfg': None,
                                   'acti_cfg': {'type': 'softmax', 'dim': 1}})

    def forward(self, x):
        x = self.pool(x)
        return self.attn(x)  # B x n_routes x 1 x 1

    def init_weights(self):
        kaiming_init(self.attn.conv, a=0, mode='fan_in', nonlinearity='sigmoid', bias=0)


@REID_BLOCKS_V2.register_class('multi_route')
class MultiRoute(BaseModule):
    def __init__(self, in_channels, out_channels, n_routes, route_depth):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if isinstance(out_channels, (list, tuple)) else [out_channels]
        self.n_routes = n_routes if isinstance(n_routes, (list, tuple)) else [n_routes]
        self.route_depth = route_depth if isinstance(route_depth, (list, tuple)) else [route_depth]
        self.length = len(self.out_channels)

        self.routes = REID_BLOCKS_V2.create([{'type': 'single_route',
                                              'in_channels': in_channels if i == 0 else self.out_channels[i - 1],
                                              'out_channels': self.out_channels[i],
                                              'depth': self.route_depth[i]}
                                             for i in range(self.length) for _ in range(self.n_routes[i])])
        self.attns = REID_BLOCKS_V2.create([{'type': 'route_attention',
                                             'in_channels': in_channels if i == 0 else self.out_channels[i - 1],
                                             'n_routes': self.n_routes[i]}
                                            for i in range(self.length)])

    def forward(self, x):
        out = x
        base = 0
        for i in range(self.length):
            n_routes = self.n_routes[i]
            a = self.attns[i](out)
            out = sum([self.routes[base + j](out) * a[:, j][:, None] for j in range(n_routes)])
            base += n_routes
        return out


@REID_BLOCKS_V2.register_class('feature_extractor')
class FeatureExtractor(BaseModule):
    _versions = {
        0: {'prepool_proj': None,
            'postpool_proj': None},
        1: {'prepool_proj': None,
            'postpool_proj': {'type': 'single_route', 'out_channels': 128, 'depth': 0}},
        2: {'prepool_proj': {'type': 'single_route', 'out_channels': 128, 'depth': 0},
            'postpool_proj': None},
        3: {'prepool_proj': {'type': 'single_route', 'out_channels': 512, 'depth': 0},
            'postpool_proj': {'type': 'single_route', 'out_channels': 128, 'depth': 0}},
        4: {'prepool_proj': None,
            'postpool_proj': {'type': 'multi_route', 'out_channels': 128, 'n_routes': 3, 'route_depth': 0}},
        5: {'prepool_proj': {'type': 'multi_route', 'out_channels': 128, 'n_routes': 3, 'route_depth': 0},
            'postpool_proj': None},
        6: {'prepool_proj': {'type': 'multi_route', 'out_channels': 512, 'n_routes': 3, 'route_depth': 0},
            'postpool_proj': {'type': 'multi_route', 'out_channels': 128, 'n_routes': 3, 'route_depth': 0}}
    }

    def __init__(self,
                 in_channels,
                 pool_type='avg_pool',
                 prepool_proj=None,
                 postpool_proj=None,
                 version=0,
                 depth=None,
                 n_routes=None,
                 route_depth=None):
        super().__init__()
        self.in_channels = in_channels
        assert pool_type in ('avg_pool', 'attn_pool')
        self.pool_type = pool_type
        self.version = version
        self.prepool_proj_cfg = {**(self._versions[version]['prepool_proj'] or {}), **(prepool_proj or {})}
        self.postpool_proj_cfg = {**(self._versions[version]['postpool_proj'] or {}), **(postpool_proj or {})}

        self.prepool_proj_cfg = self._replace_cfg(self.prepool_proj_cfg, 'depth', depth)
        self.prepool_proj_cfg = self._replace_cfg(self.prepool_proj_cfg, 'n_routes', n_routes)
        self.prepool_proj_cfg = self._replace_cfg(self.prepool_proj_cfg, 'route_depth', route_depth)
        self.postpool_proj_cfg = self._replace_cfg(self.postpool_proj_cfg, 'depth', depth)
        self.postpool_proj_cfg = self._replace_cfg(self.postpool_proj_cfg, 'n_routes', n_routes)
        self.postpool_proj_cfg = self._replace_cfg(self.postpool_proj_cfg, 'route_depth', route_depth)

        self.prepool_proj = REID_BLOCKS_V2.create(self.prepool_proj_cfg, (in_channels,))
        out_channels = in_channels
        if isinstance(self.prepool_proj, SingleRoute):
            out_channels = self.prepool_proj.out_channels
        elif isinstance(self.prepool_proj, MultiRoute):
            out_channels = self.prepool_proj.out_channels[-1]

        self.pool = AvgPool() if pool_type == 'avg_pool' else AttnPool(out_channels)

        self.postpool_proj = REID_BLOCKS_V2.create(self.postpool_proj_cfg, (out_channels,))
        if isinstance(self.postpool_proj, SingleRoute):
            out_channels = self.postpool_proj.out_channels
        elif isinstance(self.postpool_proj, MultiRoute):
            out_channels = self.postpool_proj.out_channels[-1]

        self.out_channels = out_channels

    @staticmethod
    def _replace_cfg(cfg, key, value):
        if not cfg or value is None or key not in cfg:
            return cfg
        cfg[key] = value
        return cfg

    def forward(self, x):
        out = x
        if self.prepool_proj:
            out = self.prepool_proj(out)
        out = self.pool(out)
        if self.postpool_proj:
            out = self.postpool_proj(out)
        return out


@REID_BLOCKS_V2.register_class('feature_aggregator')
class FeatureAggregator(BaseModule):
    def __init__(self,
                 in_channels,
                 n_features,
                 out_channels,
                 depth=0):
        super().__init__()
        self.in_channels = in_channels
        self.n_features = n_features
        self.out_channels = out_channels
        self.depth = depth

        self.proj = SingleRoute(in_channels, out_channels, depth)

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == self.n_features
        out = torch.cat(x, dim=1) if len(x) > 1 else x[0]
        out = self.proj(out)
        return out


@HEADS.register_class('reid_head_v2')
class ReIDHeadV2(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 feature_extractor=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_extractor_cfg = feature_extractor

        self.feature_extractor = REID_BLOCKS_V2.create(self.feature_extractor_cfg, (in_channels,), _ignore=('name', 'abbr'))
        self.feature_aggregator = FeatureAggregator(self.feature_channels, self.n_features, out_channels)

    @property
    def n_features(self):
        if isinstance(self.feature_extractor, ModuleList):
            return len(self.feature_extractor)
        else:
            return 1

    @property
    def feature_channels(self):
        if isinstance(self.feature_extractor, ModuleList):
            return sum([mod.out_channels for mod in self.feature_extractor])
        else:
            return self.feature_extractor.out_channels

    def get_extractor(self):
        if isinstance(self.feature_extractor, ModuleList):
            for cfg, mod in zip(self.feature_extractor_cfg, self.feature_extractor):
                yield cfg['name'], mod, cfg.get('abbr', cfg['name'])
        else:
            cfg, mod = self.feature_extractor_cfg, self.feature_extractor
            yield cfg['name'], mod, cfg.get('abbr', cfg['name'])

    def get_aggregator(self):
        return 'feature_aggregator', self.feature_aggregator, 'faggr'

    def forward(self, x, return_all_features=False):
        features = OrderedDict()

        for name, mod, _ in self.get_extractor():
            out = mod(x)
            assert name not in features, f"Duplicated feature name: {name}"
            features[name] = out

        name, mod, _ = self.get_aggregator()
        out = mod(list(features.values()))
        features[name] = out

        return (out, features) if return_all_features else out
