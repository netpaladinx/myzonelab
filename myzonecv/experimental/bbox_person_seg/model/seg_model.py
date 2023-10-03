import os.path as osp

import torch
import torch.nn as nn

from myzonecv.core.registry import BACKBONES, HEADS, LOSSES, POSTPROCESSORS
from myzonecv.core.model import BaseModel
from myzonecv.core.utils import auto_fp16, profile, mktempdir, rmdir, mv
from ..registry import SEG_MODELS


@SEG_MODELS.register_class('encoder_decoder')
class SegEncoderDecoder(BaseModel):
    def __init__(self,
                 backbone,
                 decode_head,
                 auxiliary_head=None,
                 decode_loss=None,
                 auxiliary_loss=None,
                 accuracy=None,
                 predict=None,
                 fp16_enabled=False,
                 ema_enabled=False,
                 train_cfg=None,
                 eval_cfg=None,
                 infer_cfg=None,
                 init_cfg=None):
        super().__init__(fp16_enabled, ema_enabled, train_cfg, eval_cfg, infer_cfg, init_cfg)
        self.backbone_cfg = backbone
        self.decode_head_cfg = decode_head
        self.auxiliary_head_cfg = auxiliary_head
        self.decode_loss_cfg = decode_loss
        self.auxiliary_loss_cfg = auxiliary_loss
        self.accuracy_cfg = accuracy
        self.predict_cfg = predict

        self.backbone = BACKBONES.create(self.backbone_cfg)
        self.decode_head = HEADS.create(self.decode_head_cfg)
        self.auxiliary_head = HEADS.create(self.auxiliary_head_cfg)
        self.decode_loss = LOSSES.create(self.decode_loss_cfg, loss_name='dec_loss')
        self.auxiliary_loss = LOSSES.create(self.auxiliary_loss_cfg, loss_name='aux_loss')
        self.decode_accuracy = POSTPROCESSORS.create(self.accuracy_cfg, accuracy_name='dec_acc')
        self.auxiliary_accuracy = POSTPROCESSORS.create(self.accuracy_cfg, accuracy_name='aux_acc')
        self.predict = POSTPROCESSORS.create(self.predict_cfg, align_corners=self.decode_loss.align_corners)

        self.auxiliary_loss.ignore_index = self.decode_loss.ignore_index
        self.accuracy = self.decode_loss.ignore_index

        self.init_weights()
        self.pred_cache_dir = mktempdir()

    @property
    def align_corners(self):
        return self.decode_head.align_corners

    @property
    def n_classes(self):
        return self.decode_head.n_classes

    @property
    def in_channels(self):
        return self.backbone.in_channels

    def compute_loss(self, outputs, targets):
        """ outputs (tuple):
                decode_output: bs x n_classes x out_h x out_w
                auxiliary_output: bs x n_classes x out_h x out_w
            targets (dict):
                'target_seg': bs x out_h x out_w (dtype: torch.int64)
                'target_weight': bs x out_h x out_w
        """
        decode_output, auxiliary_output = outputs
        target_seg = targets['target_seg']
        target_weight = targets.get('target_weight')
        assert target_seg.dim() == 3 and (target_weight is None or target_weight.dim() == 3)
        assert target_seg.shape[-2:] == decode_output.shape[-2:] == auxiliary_output.shape[-2:]

        losses = {}
        losses.update(self.decode_loss(decode_output, target_seg, target_weight))
        if self.auxiliary_loss:
            losses.update(self.auxiliary_loss(auxiliary_output, target_seg, target_weight))
        return losses

    def compute_accuracy(self, outputs, targets):
        decode_output, auxiliary_output = outputs
        target_seg = targets['target_seg']

        accs = {}
        accs.update(self.decode_accuracy(decode_output, target_seg))
        if self.auxiliary_accuracy:
            accs.update(self.auxiliary_accuracy(auxiliary_output, target_seg))
        return accs

    @auto_fp16(apply_to=('inputs',))
    def forward_train(self, inputs, targets, **kwargs):
        """ inputs (dict)
                'img': bs x img_c x img_h x img_w
        """
        img = inputs['img']

        output = self.backbone(img)
        decode_output = self.decode_head(output)
        auxiliary_output = self.auxiliary_head(output)
        outputs = (decode_output, auxiliary_output)

        results = {}
        summary_keys = []

        res = self.compute_loss(outputs, targets)
        summary_keys += self.collect_summary(res)
        results.update(res)

        res = self.compute_accuracy(outputs, targets)
        summary_keys += self.collect_summary(res)
        results.update(res)

        results = self.merge_losses(results)
        results['summary_keys'] = summary_keys
        return results

    @auto_fp16(apply_to=('inputs',))
    def forward_predict(self, inputs, batch_dict, return_output=False, **kwargs):
        img = inputs['img']

        output = self.backbone(img)
        output = self.decode_head(output)

        results = {}
        results.update(self.predict(output, batch_dict, pred_cache_dir=self.pred_cache_dir))
        if return_output:
            results['logit_output'] = output
        return results

    @auto_fp16(apply_to=('img',))
    def forward_dummy(self, img):
        output = self.backbone(img)
        output = self.decode_head(output)
        return output

    def info(self, input_size=512, pretty_print=False):
        input_size = input_size if isinstance(input_size, list) else [input_size, input_size]

        model_info = super().info(input_size, self.in_channels)

        input_img = torch.zeros((1, self.in_channels, input_size[0], input_size[0])).to(self.device)

        gflops = profile(input_img, self, flops_only=True)[0]
        model_info['GFLOPs'] = round(gflops, 2)

        if pretty_print:
            model_info = '\n'.join([f'{k}: {v}' for k, v in model_info.items()])
        return model_info

    def call_after_eval(self, ctx):
        if ctx.is_('save_cache', True):
            cache_dir = osp.join(ctx.work_dir, '.pred_cache')
            mv(self.pred_cache_dir, cache_dir)
        rmdir(self.pred_cache_dir)

    def call_after_infer(self, ctx):
        self.call_after_eval(ctx)


@SEG_MODELS.register_class('bbox_person')
class SegBBoxPerson(SegEncoderDecoder):
    def compute_loss(self, outputs, targets):
        """ outputs (tuple):
                decode_output: bs x 1 x out_h x out_w
                auxiliary_output: bs x 1 x out_h x out_w
            targets (dict):
                'target_mask': bs x 1 x out_h x out_w (dtype: torch.float32)
        """
        decode_output, auxiliary_output = outputs
        target_mask = targets['target_mask']
        assert target_mask.shape == decode_output.shape == auxiliary_output.shape

        losses = {}
        losses.update(self.decode_loss(decode_output, target_mask))
        if self.auxiliary_loss:
            losses.update(self.auxiliary_loss(auxiliary_output, target_mask))
        return losses

    def compute_accuracy(self, outputs, targets):
        decode_output, auxiliary_output = outputs
        target_mask = targets['target_mask']

        accs = {}
        accs.update(self.decode_accuracy(decode_output, target_mask))
        if self.auxiliary_accuracy:
            accs.update(self.auxiliary_accuracy(auxiliary_output, target_mask))
        return accs


@SEG_MODELS.register_class('pointguided_bbox_person')
class SegPointGuidedBBoxPerson(SegBBoxPerson):
    def revise_state_dict(self, state_dict):
        weight_name = 'backbone.stem.0.conv.weight'
        weight = state_dict.get(weight_name)
        if weight is not None and weight.shape[1] == 3:
            new_shape = list(weight.shape)
            new_shape[1] = 4
            new_weight = weight.new_zeros(new_shape)
            nn.init.normal_(new_weight, mean=0, std=0.01)
            new_weight[:, :3, ...] = weight
            state_dict[weight_name] = new_weight
        return state_dict
