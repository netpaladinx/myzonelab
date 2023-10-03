import torch
import torch.nn as nn

from ...registry import ACTI_LAYERS

ACTI_LAYERS.register_class('relu', nn.ReLU)
ACTI_LAYERS.register_class('leaky_relu', nn.LeakyReLU)
ACTI_LAYERS.register_class('prelu', nn.PReLU)
ACTI_LAYERS.register_class('rrelu', nn.RReLU)
ACTI_LAYERS.register_class('relu6', nn.ReLU6)
ACTI_LAYERS.register_class('elu', nn.ELU)
ACTI_LAYERS.register_class('gelu', nn.GELU)
ACTI_LAYERS.register_class('silu', nn.SiLU)
ACTI_LAYERS.register_class('sigmoid', nn.Sigmoid)
ACTI_LAYERS.register_class('hardsigmoid', nn.Hardsigmoid)
ACTI_LAYERS.register_class('tanh', nn.Tanh)


def create_acti_layer(acti_cfg, *args, **kwargs):
    acti_cfg = acti_cfg.copy()
    acti_cfg.update(kwargs)
    if args:
        acti_layer = ACTI_LAYERS.create(acti_cfg, args)
    else:
        acti_layer = ACTI_LAYERS.create(acti_cfg)
    return acti_layer


@ACTI_LAYERS.register_class('clamp')
class Clamp(nn.Module):
    def __init__(self, min=-1., max=1., **kwargs):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=self.min, max=self.max)


@ACTI_LAYERS.register_class('hsigmoid')
class HSigmoid(nn.Module):
    def __init__(self, bias=1., divisor=2., min=0., max=1., **kwargs):
        super().__init__()
        self.bias = bias
        assert divisor != 0
        self.divisor = divisor
        self.min = min
        self.max = max

    def forward(self, x):
        x = (x + self.bias) / self.divisor
        return x.clamp_(self.min, self.max)


@ACTI_LAYERS.register_class('sigmoid_ms')
class SigmoidMS(nn.Module):
    def __init__(self, mean=0., std=1., dim=1, gain=0., **kwargs):
        super().__init__()
        self.mean = mean if isinstance(mean, (list, tuple)) else [mean]
        self.std = std if isinstance(std, (list, tuple)) else [std]
        self.dim = dim
        self.gain = gain

    def forward(self, x):
        sz_mean = [len(self.mean) if i == self.dim else 1 for i in range(x.ndim)]
        sz_std = [len(self.std) if i == self.dim else 1 for i in range(x.ndim)]
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device).reshape(sz_mean)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device).reshape(sz_std)

        x = torch.sigmoid(x) * (1 + self.gain) - 0.5 * self.gain
        x = (x - mean) / std
        return x


@ACTI_LAYERS.register_class('swish')
class Swish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


@ACTI_LAYERS.register_class('hswish')
class HSwish(nn.Module):
    def __init__(self, inplace=False, **kwargs):
        super().__init__()
        self.act = nn.ReLU6(inplace)

    def forward(self, x):
        x = x * self.act(x + 3) / 6
        return x


@ACTI_LAYERS.register_class('arsinh')
class Arsinh(nn.Module):
    def __init__(self, alpha=1., **kwargs):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        x = torch.asinh(self.alpha * x)
        return x


@ACTI_LAYERS.register_class('softmax')
class Softmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.act = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.act(x)
