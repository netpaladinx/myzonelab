import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from ...registry import HEADS
from ..base_module import BaseModule, Sequential
from ..initializers import normal_init, constant_init, build_module_from_init
from ..bricks import create_upsample_layer, makeup_deconv_cfg, create_norm_layer, create_conv_layer
from ..postprocessors import PoseProcess


def create_heatmap_deconv(n_layers, n_filters, n_kernels, in_channels, norm_cfg=dict(type='bn')):
    assert n_layers == len(n_filters)
    assert n_layers == len(n_kernels)
    layers = []
    deconv_cfg = dict(type='deconv', stride=2, bias=False)
    for i in range(n_layers):
        out_channels = n_filters[i]
        kernel_size = n_kernels[i]
        deconv_cfg.update(dict(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size))
        deconv_cfg = makeup_deconv_cfg(deconv_cfg)
        layers.append(create_upsample_layer(deconv_cfg))
        layers.append(create_norm_layer(norm_cfg, out_channels))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    deconv = Sequential(*layers)
    deconv.out_channels = in_channels
    return deconv


def create_heatmap_conv(n_layers, n_filters, n_kernels, final_kernel, final_padding, in_channels, out_channels, norm_cfg=dict(type='bn')):
    layers = []
    conv_cfg = dict(type='conv', stride=1)
    final_out_channels = out_channels
    for i in range(n_layers):
        out_channels = n_filters[i]
        kernel_size = n_kernels[i]
        padding = (kernel_size - 1) // 2
        conv_cfg.update(dict(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding))
        layers.append(create_conv_layer(conv_cfg))
        layers.append(create_norm_layer(norm_cfg, out_channels))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    conv_cfg.update(dict(in_channels=in_channels, out_channels=final_out_channels, kernel_size=final_kernel, padding=final_padding))
    layers.append(create_conv_layer(conv_cfg))
    if len(layers) > 1:
        return Sequential(*layers)
    else:
        return layers[0]


def create_feature_layers(n_hidden_layers, in_features, hidden_features, final_features,
                          flatten_first=True, final_activation=True):
    layers = [nn.Flatten()] if flatten_first else []
    for i in range(n_hidden_layers):
        out_features = hidden_features[i]
        layers.append(nn.Linear(in_features, out_features, bias=True))
        layers.append(nn.ReLU(inplace=True))
        in_features = out_features
    layers.append(nn.Linear(in_features, final_features, bias=True))
    if final_activation:
        layers.append(nn.ReLU(inplace=True))
    if len(layers) > 1:
        return Sequential(*layers)
    else:
        return layers[0]


class BasePoseHead(BaseModule):
    def __init__(self,
                 in_channels=None,
                 input_transform=None,
                 input_index=0,
                 align_corners=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels, self.input_transform, self.input_index = \
            self.check_input_transform(in_channels, input_transform, input_index)
        self.align_corners = align_corners

    def check_input_transform(self, in_channels, input_transform, input_index):
        if in_channels is not None:
            if input_transform is not None:
                assert input_transform in ('resize_concat', 'multiple_select')
                assert isinstance(in_channels, (list, tuple))
                assert isinstance(input_index, (list, tuple))
                assert len(in_channels) == len(input_index)
                if input_transform == 'resize_concat':
                    in_channels = sum(in_channels)
            else:
                assert isinstance(in_channels, int)
                assert isinstance(input_index, int) or input_index is None
        return in_channels, input_transform, input_index

    def transform_inputs(self, x):
        if not isinstance(x, (list, tuple)):
            return x

        if self.input_transform == 'resize_concat':
            h, w = x[0].shape[2:]
            xs = [F.interpolate(x[i], size=(h, w), mode='bilinear', align_corners=self.align_corners)
                  for i in self.input_index]
            x = torch.cat(xs, dim=1)
        elif self.input_transform == 'multipe_select':
            x = [x[i] for i in self.input_index]
        else:
            x = x[self.input_index]
        return x


@HEADS.register_class('pose_keypoint')
class PoseKeypoint(BasePoseHead):
    def __init__(self,
                 in_channels,
                 out_channels,
                 input_transform=None,  # default, 'resize_concat', 'multiple_select'
                 input_index=0,
                 align_corners=False,
                 n_deconvs=0,
                 deconv_filters=(),
                 deconv_kernels=(),
                 n_convs=0,
                 conv_filters=(),
                 conv_kernels=(),
                 final_conv_kernel=1,
                 use_heatmap_reg=False,
                 heatmap_reg_betas=(0.001, 0.001),
                 init_cfg=None):
        super().__init__(in_channels, input_transform, input_index, align_corners, init_cfg)
        self.out_channels = out_channels
        self.n_deconvs = n_deconvs
        self.deconv_filters = deconv_filters
        self.deconv_kernels = deconv_kernels
        self.n_convs = n_convs
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.final_conv_kernel = final_conv_kernel
        self.use_heatmap_reg = use_heatmap_reg

        if self.n_deconvs > 0:
            self.deconv_layers = create_heatmap_deconv(self.n_deconvs, self.deconv_filters, self.deconv_kernels, in_channels)
            in_channels = self.deconv_layers.out_channels
        else:
            self.deconv_layers = nn.Identity()

        final_conv_kernel = self.final_conv_kernel
        if final_conv_kernel > 0:
            assert final_conv_kernel in (1, 3)
            final_conv_padding = 1 if final_conv_kernel == 3 else 0
            self.final_layer = create_heatmap_conv(self.n_convs, self.conv_filters, self.conv_kernels,
                                                   final_conv_kernel, final_conv_padding, in_channels, out_channels)
        else:
            self.final_layer = nn.Identity()

        if self.use_heatmap_reg:
            beta1, beta2 = heatmap_reg_betas
            self.register_buffer('_heatmap_reg', torch.tensor([0., 0., 0., beta1, beta2]))  # (sum, square_sum, count, beta1, beta2)

        self.eval_process = PoseProcess()

    def _update_heatmap_reg(self, x):
        """ x: bs x kpts x h x w 
        """
        with torch.no_grad():
            bs, c = x.size()[:2]
            x = x.view(bs, c, -1)
            reg1 = x.mean(-1).mean()
            reg2 = (x**2).mean(-1).sqrt().mean()

            betas = torch.maximum(self._heatmap_reg[3:5], bs / (self._heatmap_reg[2] + bs))
            self._heatmap_reg[:2] = self._heatmap_reg[:2] * (1 - betas) + torch.stack((reg1, reg2)) * betas
            self._heatmap_reg[2] += bs

    @property
    def heatmap_reg(self):
        return self._heatmap_reg[:2] if self.use_heatmap_reg else None

    def init_weights(self):
        super().init_weights()
        if isinstance(self.init_cfg, dict) and self.init_cfg['type'] == 'pretrained':
            return

        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x, skip_heatmap_reg=True):
        """ x: list(bs x c x h x w)
        """
        x = self.transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        if self.use_heatmap_reg and not skip_heatmap_reg:
            self._update_heatmap_reg(x)

        return x

    def forward_predict(self, x, flip_pairs=None, shift_heatmap=False, **kwargs):
        output = self(x)
        output = output.detach().cpu().numpy()
        if flip_pairs is not None:
            output = self.eval_process.flip_heatmap_back(output, flip_pairs)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if shift_heatmap:
                output[:, :, :, 1:] = output[:, :, :, :-1]
        return output


@HEADS.register_class('pose_feature')
class PoseFeature(BasePoseHead):
    """ Example:
        {
            in_channels: 32,
            in_features: 98304, # 32 * 64 * 48
            n_hidden_layers: 1,
            hidden_features: [1024],
            final_features: 128
        } 
    """

    def __init__(self,
                 in_channels=None,
                 in_features=None,
                 n_hidden_layers=None,
                 hidden_features=None,
                 final_features=None,
                 input_transform=None,
                 input_index=0,
                 align_corners=False,
                 build_from_init=False,
                 init_cfg=None):
        super().__init__(in_channels, input_transform, input_index, align_corners, init_cfg)
        self.in_features = in_features
        self.n_hidden_layers = n_hidden_layers
        self.hidden_features = hidden_features
        self.final_features = final_features

        if build_from_init:
            self.layers = build_module_from_init(self.init_cfg)
            self.init_cfg = None
            self.init_from_module()

        else:
            self.layers = create_feature_layers(n_hidden_layers, in_features, hidden_features, final_features)

    def forward(self, x):
        """ x: list(bs x c x h x w)
        """
        x = self.transform_inputs(x)  # bs x c x h x w
        x = self.layers(x)
        return x

    def init_from_module(self):
        self.n_hidden_layers = 0
        self.hidden_features = []

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.n_hidden_layers += 1
                self.hidden_features.append(layer.out_features)

                if self.in_features is None:
                    self.in_features = layer.in_features

        self.n_hidden_layers -= 1
        self.final_features = self.hidden_features[-1]
        self.hidden_features = self.hidden_features[:-1]
