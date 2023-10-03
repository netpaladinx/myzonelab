import warnings

import torch.nn as nn

from ...registry import BACKBONES, BLOCKS
from ...utils import to_numpy
from ..base_module import BaseModule, Sequential
from ..initializers import kaiming_init, constant_init
from ..bricks import infer_norm_name, create_norm_layer, create_acti_layer


@BLOCKS.register_class('dnn_layer')
class DNNLayer(BaseModule):
    def __init__(self, in_dims, out_dims, bias='auto', norm_cfg=None, acti_cfg=None, inplace=True, order=('linear', 'norm', 'acti')):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        if bias == 'auto':
            bias = norm_cfg is None
        self.bias = bias
        self.norm_cfg = norm_cfg
        self.acti_cfg = acti_cfg
        self.inplace = inplace
        assert isinstance(order, (list, tuple)) and len(set(order) & set(['linear', 'norm', 'acti'])) == 3
        self.order = order

        self.linear = nn.Linear(in_dims, out_dims, bias)

        if norm_cfg is not None:
            if order.index('norm') > order.index('linear'):
                norm_dims = out_dims
            else:
                norm_dims = in_dims
            self.norm_name = infer_norm_name(norm_cfg)
            norm = create_norm_layer(norm_cfg, norm_dims)
            self.add_module(self.norm_name, norm)
            if self.bias:
                if isinstance(norm_cfg['type'], ('bn', 'bn1d', 'bn2d', 'bn3d', 'in', 'in1d', 'in2d', 'in3d')):
                    warnings.warn("Unnecessary conv bias before batch/instance norm")

        if acti_cfg is not None:
            acti_cfg = acti_cfg.copy()
            if acti_cfg['type'] not in ('tanh', 'prelu', 'sigmoid', 'hardsigmoid', 'hsigmoid', 'swish', 'clamp', 'silu'):
                acti_cfg.setdefault('inplace', inplace)
            self.acti = create_acti_layer(acti_cfg)

    @property
    def norm(self):
        if self.norm_cfg is not None:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        if self.acti_cfg is not None:
            if self.acti_cfg['type'] == 'leaky_relu':
                nonlinearity = 'leaky_relu'
                a = self.acti_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
        else:
            nonlinearity = 'linear'
            a = 0
        kaiming_init(self.linear, a=a, nonlinearity=nonlinearity, bias=0, distribution='normal')

        if self.norm_cfg is not None:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x):
        for layer in self.order:
            if layer == 'linear':
                x = self.linear(x)
            elif layer == 'norm' and self.norm_cfg is not None:
                x = self.norm(x)
            elif layer == 'acti' and self.acti_cfg is not None:
                x = self.acti(x)
        return x

    def export_to_numpy(self):
        layers = []
        for layer in self.order:
            if layer == 'linear':
                layers.append(('linear', to_numpy(self.linear.weight), to_numpy(self.linear.bias) if self.bias else None))
            elif layer == 'norm' and self.norm_cfg is not None:
                layers.append((self.norm_cfg['type'], to_numpy(self.norm.weight), to_numpy(self.norm.bias)))
            elif layer == 'acti' and self.acti_cfg is not None:
                layers.append((self.acti_cfg['type'],))
        return layers


@BACKBONES.register_class('dnn')
class DNN(BaseModule):
    def __init__(self,
                 in_dims,
                 out_dims,
                 hidden_dims=None,
                 depth=0,
                 hidden_acti_cfg={'type': 'relu'},
                 out_acti_cfg=None,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        self.hidden_acti_cfg = hidden_acti_cfg
        self.out_acti_cfg = out_acti_cfg
        self.norm_cfg = norm_cfg

        layers = [DNNLayer(in_dims, out_dims if depth == 0 else hidden_dims,
                           norm_cfg=norm_cfg, acti_cfg=out_acti_cfg if depth == 0 else hidden_acti_cfg)]
        for i in range(depth):
            layers.append(DNNLayer(hidden_dims, out_dims if i == depth - 1 else hidden_dims,
                                   norm_cfg=norm_cfg, acti_cfg=out_acti_cfg if i == depth - 1 else hidden_acti_cfg))
        self.layers = Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

    def export_to_numpy(self):
        layers = []
        for dnn_layer in self.layers:
            layers += dnn_layer.export_to_numpy()
        return layers
