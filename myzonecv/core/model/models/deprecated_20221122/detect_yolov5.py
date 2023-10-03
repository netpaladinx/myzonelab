import torch
import torch.nn as nn

from ...registry import DETECT_MODELS, BACKBONES, HEADS, LOSSES, POSTPROCESSORS
from ..base_model import BaseModel
from ...utils import auto_fp16, check_divisible, profile


@DETECT_MODELS.register_class('yolov5')
class DetectYolov5(BaseModel):
    """
    Example:
        backbone = {
            'type': 'yolov5_backbone6',
            'arch': {
                'layer1': {'block': 'conv', 'n_channels': 64, 'kernel_size': 6, 'stride': 2, 'padding': 2},
                'layer2': {'block': 'conv', 'n_channels': 128, 'kernel_size': 3, 'stride': 2},
                'layer3': {'block': 'c3_bottleneck', 'n_channels': 128, 'n_inner_blocks': 3},
                'layer4': {'block': 'conv', 'n_channels': 256, 'kernel_size': 3, 'stride': 2},
                'layer5': {'block': 'c3_bottleneck', 'n_channels': 256, 'n_inner_blocks': 6},
                'layer6': {'block': 'conv', 'n_channels': 512, 'kernel_size': 3, 'stride': 2},
                'layer7': {'block': 'c3_bottleneck', 'n_channels': 512, 'n_inner_blocks': 9},
                'layer8': {'block': 'conv', 'n_channels': 1024, 'kernel_size': 3, 'stride': 2},
                'layer9': {'block': 'c3_bottleneck', 'n_channels': 1024, 'n_inner_blocks': 3},
                'layer10': {'block': 'sppf', 'n_channels': 1024, 'kernel_size': 5},
            },
            'in_channels': 3,
            'depth_multiple': 0.33,
            'width_multiple': 0.50
        },
        head = {
            'type': 'yolov5_head6',
            'arch': {
                'layer1': {'block': 'conv', 'n_channels': 512, 'kernel_size': 1, 'stride': 1},
                'layer2': {'block': 'upsample_layer.upsample', 'scale_factor': 2, 'mode': 'nearest'},
                'layer3': {'block': 'concat', 'input_index': [-1, 6], 'dim': 1},
                'layer4': {'block': 'c3_bottleneck', 'n_channels': 512, 'n_inner_blocks': 3, 'shortcut': False},
                'layer5': {'block': 'conv', 'n_channels': 256, 'kernel_size': 1, 'stride': 1},
                'layer6': {'block': 'upsample_layer.upsample', 'scale_factor': 2, 'mode': 'nearest'},
                'layer7': {'block': 'concat', 'input_index': [-1, 4], 'dim': 1},
                'layer8': {'block': 'c3_bottleneck', 'n_channels': 256, 'n_inner_blocks': 3, 'shortcut': False},
                'layer9': {'block': 'conv', 'n_channels': 256, 'kernel_size': 3, 'stride': 2},
                'layer10': {'block': 'concat', 'input_index': [-1, 14], 'dim': 1},
                'layer11': {'block': 'c3_bottleneck', 'n_channels': 512, 'n_inner_blocks': 3, 'shortcut': False},
                'layer12': {'block': 'conv', 'n_channels': 512, 'kernel_size': 3, 'stride': 2},
                'layer13': {'block': 'concat', 'input_index': [-1, 10], 'dim': 1},
                'layer14': {'block': 'c3_bottleneck', 'n_channels': 1024, 'n_inner_blocks': 3, 'shortcut': False},
                'layer15': {
                    'block': 'yolov5_detec6',
                    'input_index': [17, 20, 23],
                    'anchors': [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]],
                    'num_classes': 80
                },
            },
            'depth_multiple': 0.33,
            'width_multiple': 0.50
        }
    """

    def __init__(self,
                 backbone,
                 head,
                 loss=None,
                 predict=None,
                 fp16_enabled=False,
                 ema_enabled=False,
                 train_cfg=None,
                 eval_cfg=None,
                 infer_cfg=None,
                 init_cfg=None):
        super().__init__(fp16_enabled, ema_enabled, train_cfg, eval_cfg, infer_cfg, init_cfg)
        self.backbone_cfg = backbone
        self.head_cfg = head
        self.loss_cfg = loss
        self.predict_cfg = predict

        self.backbone = BACKBONES.create(self.backbone_cfg)
        self.head = HEADS.create(self.head_cfg, self.backbone)
        self.loss = LOSSES.create(self.loss_cfg)
        self.predict = POSTPROCESSORS.create(self.predict_cfg)
        self.augment_process = POSTPROCESSORS.create({'type': 'detect_postprocessor.process'})

        self.init_strides()
        self.init_weights()

    def compute_loss(self, output, targets):
        """ output: list(bs x n_anchors(3) x out_h x out_w x out_dims) 
        """
        target_cxywh = targets['target_cxywh']
        target_cij = targets['target_cij']
        target_cls = targets['target_cls']
        target_anc_idx = targets['target_anc_idx']
        target_anc = targets['target_anc']
        target_cnt = targets['target_cnt']

        return self.loss(output, target_cxywh, target_cij, target_cls, target_anc_idx, target_anc, target_cnt)

    @auto_fp16(apply_to=('inputs',))
    def forward_train(self, inputs, targets, output_preds=False, **kwargs):
        """ inputs (dict):
                'img': bs x channels x img_h x img_w
            targets (dict):
                'target_cxywh': list(n_pred x 4), (cx, cy, w, h)
                'target_cij': list(n_pred x 2), (ci, cj)
                'target_cls': list(n_pred)
                'target_anc_idx': list(n_pred)
                'target_anc': list(n_pred x 2), (anc_w, anc_h)
                'target_cnt': list(int)
        """
        img = inputs['img']
        results = {}

        output, outputs = self.backbone(img, output_indices=self.head.branches)
        output = self.head(output, backbone_outputs=outputs)
        if output_preds:
            with torch.no_grad():
                preds = self.predict(output)
                preds = torch.cat(preds, 1)
                res = self.predict.get_direct_results(preds)
                results.update(res)

        summary_keys = []

        res = self.compute_loss(output, targets)
        summary_keys += self.collect_summary(res)
        results.update(res)

        results = self.merge_losses(results)
        results['summary_keys'] = summary_keys
        return results

    @auto_fp16(apply_to=('inputs',))
    def forward_predict(self,
                        inputs,
                        batch_dict,
                        flip_augment=None,
                        scale_augment=None,
                        output_loss=True,
                        conf_thr=None,
                        iou_thr=None,
                        topk=None,
                        **kwargs):
        img = inputs['img']
        targets = batch_dict.get('targets')
        results = {}

        output, outputs = self.backbone(img, output_indices=self.head.branches)
        output = self.head(output, backbone_outputs=outputs)
        preds_list = self.predict(output)

        if targets and output_loss:
            res = self.compute_loss(output, targets)
            results.update(res)

        if flip_augment and scale_augment:
            flip_ops = flip_augment if isinstance(flip_augment, (list, tuple)) else (flip_augment,)
            scale_ops = scale_augment if isinstance(scale_augment, (list, tuple)) else (scale_augment,)

            for f, s in zip(flip_ops, scale_ops):
                aug_img = self.augment_process.flip_scale_img(img, flip=f, scale=s)
                output, outputs = self.backbone(aug_img, output_indices=self.head.branches)
                output = self.head(output, backbone_outputs=outputs)
                aug_preds_list = self.predict(output)
                rev_preds_list = self.augment_process.scale_fip_preds_back(aug_preds_list, img.shape[3], flip=f, scale=s)
                preds_list += rev_preds_list

        preds = torch.cat(preds_list, 1)  # bs x n_pred x out_dims
        res = self.predict.get_final_results(preds, batch_dict, conf_thr, iou_thr, topk)
        results.update(res)
        return results

    @auto_fp16(apply_to=('img',))
    def forward_dummy(self, img, return_preds=True):
        output, outputs = self.backbone(img, output_indices=self.head.branches)
        output = self.head(output, backbone_outputs=outputs)
        if return_preds:
            output = self.predict(output)
        return output

    def init_strides(self):
        dummy_size = 2**8
        dummy_input = torch.zeros((1, self.in_channels, dummy_size, dummy_size))
        outs = self(dummy_input, is_dummy=True, return_preds=False)
        strides = [dummy_size / out.shape[-2] for out in outs]

        self.head.init_strides(strides)
        self.predict.init_strides(strides)

    def init_weights(self):
        super().init_weights()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
            elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True

        if isinstance(self.init_cfg, dict) and self.init_cfg['type'] in ('pretrained', 'checkpoint'):
            return

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def info(self, input_size=640, pretty_print=False):
        input_size = input_size if isinstance(input_size, list) else [input_size, input_size]
        self.check_input_size(input_size)

        model_info = super().info(input_size)

        min_size = int(max(max(self.strides), 32)) if self.strides else 32
        min_img = torch.zeros((1, self.in_channels, min_size, min_size)).to(self.device)

        flops = profile(min_img, self, flops_only=True)[0]
        gflops = flops * input_size[0] / min_size * input_size[1] / min_size
        model_info['GFLOPs'] = round(gflops, 2)

        if pretty_print:
            model_info = '\n'.join([f'{k}: {v}' for k, v in model_info.items()])
        return model_info

    def check_input_size(self, input_size):
        if not check_divisible(input_size, max(self.strides)):
            raise ValueError(f"input_size {input_size} is not a multiple of stride {max(self.strides)}")

    @property
    def in_channels(self):
        return self.backbone.in_channels

    @property
    def strides(self):
        return self.head.layers[-1].strides

    @property
    def anchors(self):
        return self.head.layers[-1].anchors


@DETECT_MODELS.register_class('yolov5_myzone')
class DetectYolov5MyZone(DetectYolov5):
    @auto_fp16(apply_to=('inputs',))
    def forward_predict(self,
                        inputs,
                        batch_dict,
                        output_loss=True,
                        conf_thr=None,
                        iou_thr=None,
                        **kwargs):
        img = inputs['img']
        targets = batch_dict.get('targets')
        results = {}

        output, outputs = self.backbone(img, output_indices=self.head.branches)
        output = self.head(output, backbone_outputs=outputs)
        preds_list = self.predict(output)

        if targets and output_loss:
            res = self.compute_loss(output, targets)
            results.update(res)

        preds = torch.cat(preds_list, 1)  # bs x n_pred x out_dims
        res = self.predict.get_final_results(preds, batch_dict, conf_thr, iou_thr)
        results.update(res)
        return results

    @staticmethod
    def preprocess_loaded_state_dict(model, state_dict):
        detect_layer_idx = 14
        detect_duplicates = 2
        pretrained_detect_out_dims = 255
        pretrained_anchor_out_dims = 85
        detect_out_dims = 84
        anchor_out_dims = 14
        anchor_bbox_dims = 5
        pretrained_out_mask = torch.arange(pretrained_detect_out_dims) % pretrained_anchor_out_dims < anchor_bbox_dims
        detect_out_mask = torch.arange(detect_out_dims) % anchor_out_dims < anchor_bbox_dims

        for i in (0, 1, 2):
            weight_name = f'head.layers.{detect_layer_idx}.convs.{i}.weight'
            pretrained_conv_weight = state_dict[weight_name]
            conv_weight = model.get_parameter(weight_name).detach().clone()
            conv_weight[detect_out_mask] = pretrained_conv_weight[pretrained_out_mask].repeat(detect_duplicates, 1, 1, 1)
            state_dict[weight_name] = conv_weight

            bias_name = f'head.layers.{detect_layer_idx}.convs.{i}.bias'
            pretrained_conv_bias = state_dict[bias_name]
            conv_bias = model.get_parameter(bias_name).detach().clone()
            conv_bias[detect_out_mask] = pretrained_conv_bias[pretrained_out_mask].repeat(detect_duplicates)
            state_dict[bias_name] = conv_bias

        return state_dict


@DETECT_MODELS.register_class('yolov5_myzone_v2')
class DetectYolov5MyZoneV2(DetectYolov5MyZone):
    @staticmethod
    def preprocess_loaded_state_dict(model, state_dict):
        detect_layer_idx = 14
        pretrained_anchor_out_dims = 85
        pretrained_num_anchors = 3
        anchor_out_dims = model.head.layers[-1].out_dims
        anchor_bbox_dims = 5
        num_anchors = model.head.layers[-1].n_anchors

        for i in (0, 1, 2):
            weight_name = f'head.layers.{detect_layer_idx}.convs.{i}.weight'
            pretrained_conv_weight = state_dict[weight_name]
            conv_weight = model.get_parameter(weight_name).detach().clone()
            for di in range(num_anchors):
                si = int(di / num_anchors * pretrained_num_anchors)
                conv_weight[di * anchor_out_dims:anchor_bbox_dims + di * anchor_out_dims] = \
                    pretrained_conv_weight[si * pretrained_anchor_out_dims:anchor_bbox_dims + si * pretrained_anchor_out_dims]
            state_dict[weight_name] = conv_weight

            bias_name = f'head.layers.{detect_layer_idx}.convs.{i}.bias'
            pretrained_conv_bias = state_dict[bias_name]
            conv_bias = model.get_parameter(bias_name).detach().clone()
            for di in range(num_anchors):
                si = int(di / num_anchors * pretrained_num_anchors)
                conv_bias[di * anchor_out_dims:anchor_bbox_dims + di * anchor_out_dims] = \
                    pretrained_conv_bias[si * pretrained_anchor_out_dims:anchor_bbox_dims + si * pretrained_anchor_out_dims]
            state_dict[bias_name] = conv_bias

        return state_dict


@DETECT_MODELS.register_class('yolov5_myzone_v3')
class DetectYolov5MyZoneV3(DetectYolov5MyZone):
    @staticmethod
    def preprocess_loaded_state_dict(model, state_dict):
        detect_layer_idx = 14
        pretrained_anchor_out_dims = 85
        pretrained_num_anchors = 3
        anchor_out_dims = model.head.layers[-1].out_dims
        anchor_num_classes = model.head.layers[-1].n_classes
        anchor_bbox_dims = 5
        num_anchors = model.head.layers[-1].n_anchors

        for i in (0, 1, 2):
            weight_name = f'head.layers.{detect_layer_idx}.convs.{i}.weight'
            pretrained_conv_weight = state_dict[weight_name]
            conv_weight = model.get_parameter(weight_name).detach().clone()
            for di in range(num_anchors):
                si = int(di / num_anchors * pretrained_num_anchors)
                for c in range(anchor_num_classes):
                    conv_weight[di * anchor_out_dims + c * anchor_bbox_dims:di * anchor_out_dims + (c + 1) * anchor_bbox_dims - 1] = \
                        pretrained_conv_weight[si * pretrained_anchor_out_dims:si * pretrained_anchor_out_dims + anchor_bbox_dims - 1]
            state_dict[weight_name] = conv_weight

            bias_name = f'head.layers.{detect_layer_idx}.convs.{i}.bias'
            pretrained_conv_bias = state_dict[bias_name]
            conv_bias = model.get_parameter(bias_name).detach().clone()
            for di in range(num_anchors):
                si = int(di / num_anchors * pretrained_num_anchors)
                for c in range(anchor_num_classes):
                    conv_bias[di * anchor_out_dims + c * anchor_bbox_dims:di * anchor_out_dims + (c + 1) * anchor_bbox_dims - 1] = \
                        pretrained_conv_bias[si * pretrained_anchor_out_dims:si * pretrained_anchor_out_dims + anchor_bbox_dims - 1]
            state_dict[bias_name] = conv_bias

        return state_dict


@DETECT_MODELS.register_class('yolov5_myzone_v4')
class DetectYolov5MyZoneV4(DetectYolov5MyZone):
    @staticmethod
    def preprocess_loaded_state_dict(model, state_dict):
        detect_layer_idx = 14
        pretrained_anchor_out_dims = 85
        pretrained_num_anchors = 3
        anchor_out_dims = model.head.layers[-1].out_dims
        anchor_num_classes = model.head.layers[-1].n_classes
        anchor_bbox_dims = 5
        num_anchors = model.head.layers[-1].n_anchors

        for i in (0, 1, 2):
            weight_name = f'head.layers.{detect_layer_idx}.convs.{i}.weight'
            pretrained_conv_weight = state_dict[weight_name]
            conv_weight = model.get_parameter(weight_name).detach().clone()
            for di in range(num_anchors):
                si = int(di / num_anchors * pretrained_num_anchors)
                for c in range(anchor_num_classes):
                    conv_weight[di * anchor_out_dims + c * (anchor_bbox_dims + 1):di * anchor_out_dims + (c + 1) * (anchor_bbox_dims + 1)] = \
                        pretrained_conv_weight[si * pretrained_anchor_out_dims:anchor_bbox_dims + si * pretrained_anchor_out_dims]
            state_dict[weight_name] = conv_weight

            bias_name = f'head.layers.{detect_layer_idx}.convs.{i}.bias'
            pretrained_conv_bias = state_dict[bias_name]
            conv_bias = model.get_parameter(bias_name).detach().clone()
            for di in range(num_anchors):
                si = int(di / num_anchors * pretrained_num_anchors)
                for c in range(anchor_num_classes):
                    conv_bias[di * anchor_out_dims + c * (anchor_bbox_dims + 1):di * anchor_out_dims + (c + 1) * (anchor_bbox_dims + 1)] = \
                        pretrained_conv_bias[si * pretrained_anchor_out_dims:anchor_bbox_dims + si * pretrained_anchor_out_dims]
            state_dict[bias_name] = conv_bias

        return state_dict
