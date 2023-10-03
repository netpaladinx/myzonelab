from collections import OrderedDict

import torch

from ...registry import REID_MODELS, BACKBONES, HEADS, LOSSES, POSTPROCESSORS
from ...utils import auto_fp16, no_grad
from ..base_model import BaseModel


@REID_MODELS.register_class('topdown')
class ReIDTopDown(BaseModel):
    """
    Example of base ReID model:
        backbone = {
            'type': 'resnets16',
            'depth': 50,
            'in_channels': 3
        }
        head = {
            'type': 'reid_head',
            'in_channels': 2048,
            'out_channels': 128
        }
        loss = {
            'type': 'reid_loss',
            'cls_loss': {
                'type': 'classification',
                'n_classes': NUM_PERSON_IDS
            },
            'ml_loss': {
                'type': 'metriclearning'
            }
        }
        accuracy = {
            'type': 'reid_postprocessor.accuracy'
        }
    """

    def __init__(self,
                 backbone,
                 head,
                 recon_head=None,
                 loss=None,
                 recon_loss=None,
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
        self.head_cfg = head
        self.recon_head_cfg = recon_head
        self.loss_cfg = loss
        self.recon_loss_cfg = recon_loss
        self.accuracy_cfg = accuracy
        self.predict_cfg = predict

        self.backbone = BACKBONES.create(self.backbone_cfg)
        self.head = HEADS.create(self.head_cfg, in_channels=self.backbone.out_channels)
        self.recon_head = HEADS.create(self.recon_head_cfg, in_channels=self.backbone.out_channels)
        self.loss = LOSSES.create(self.loss_cfg, self.head)
        self.recon_loss = LOSSES.create(self.recon_loss_cfg)
        self.accuracy = POSTPROCESSORS.create(self.accuracy_cfg, self.head)
        self.predict = POSTPROCESSORS.create(self.predict_cfg)

        self.init_weights()

    def compute_loss(self, features, targets, batch_dict, recon_pred=None, recon_gt=None,
                     apply_l1_recon=False, apply_gan_recon=False,
                     train_discriminator=False, train_recon_only=False):
        """ features (dict):
                'global_average_extractor': B x D or B x D x 1 x 1
                'final_concat_aggregator': B x D or B x D x 1 x 1
        """
        cls_target = targets['cls_target']
        mask_pos = batch_dict['mask_pos']
        mask_neg = batch_dict['mask_neg']
        mask_tri = batch_dict['mask_tri']

        x_dict = OrderedDict()
        for k, v in features.items():
            x_dict[k] = v.reshape(v.shape[0], -1) if v.ndim > 2 else v

        loss_res = {}

        if not (train_discriminator or train_recon_only):
            loss_res.update(self.loss(x_dict, cls_target, mask_pos, mask_neg, mask_tri))

        if recon_pred is not None:
            mask_img = targets.get('mask')
            loss_res.update(self.recon_loss(recon_pred, recon_gt, mask_img,
                                            apply_l1_recon=apply_l1_recon,
                                            apply_gan_recon=apply_gan_recon,
                                            train_discriminator=train_discriminator,
                                            train_recon_only=train_recon_only))

        return loss_res

    def compute_accuracy(self, features, batch_dict):
        groups, members = batch_dict['size_params']

        x_dict = OrderedDict()
        for k, v in features.items():
            v_np = v.detach().cpu().numpy()
            x_dict[k] = v_np.reshape(v_np.shape[0], -1) if v_np.ndim > 2 else v_np

        acc_res = self.accuracy(x_dict, groups, members)
        return acc_res

    @auto_fp16(apply_to=('inputs',))
    def forward_train(self, inputs, targets, batch_dict=None,
                      apply_l1_recon=False, apply_gan_recon=False,
                      train_discriminator=False, train_recon_only=False, **kwargs):
        """ inputs (dict)
                'img': B x C x H x W
            targets (dict)
                'cls_target': B (dtype: torch.int64)
                'mask': B x H x W (dtype: torch.bool, optional)
            batch_dict (dict)
                'mask_pos': B x B
                'mask_neg': B x B
                'mask_tri': #Trues-in-mask_pos x #Trues-in-mask_neg
                'size_params': (groups, members)
        """
        img = inputs['img']
        img_gt = inputs.get('orig_img')
        if img_gt is None:
            img_gt = img

        with no_grad(enable=train_discriminator or train_recon_only):
            bb_out = self.backbone(img)
            _, features = self.head(bb_out, return_all_features=True)

        with no_grad(enable=train_discriminator):
            recon_out = None
            if apply_l1_recon or apply_gan_recon or train_discriminator:
                recon_out = self.recon_head(bb_out)

        results = {}
        summary_keys = []

        loss_res = self.compute_loss(features, targets, batch_dict,
                                     recon_pred=recon_out, recon_gt=img_gt,
                                     apply_l1_recon=apply_l1_recon, apply_gan_recon=apply_gan_recon,
                                     train_discriminator=train_discriminator, train_recon_only=train_recon_only)
        summary_keys += self.collect_summary(loss_res)
        results.update(loss_res)

        acc_res = self.compute_accuracy(features, batch_dict)
        summary_keys += self.collect_summary(acc_res)
        results.update(acc_res)

        results = self.merge_losses(results)
        results['summary_keys'] = summary_keys
        return results

    @auto_fp16(apply_to=('inputs',))
    def forward_predict(self, inputs, batch_dict=None, return_all_features=False, return_recon=False, **kwargs):
        img = inputs['img']

        bb_out = self.backbone(img)
        if return_all_features:
            output, features = self.head(bb_out, return_all_features=True)
        else:
            output, features = self.head(bb_out), None

        recon_out = None
        recon_mask = None
        if return_recon:
            recon_out = self.recon_head(bb_out)
            recon_mask = batch_dict.get('mask')

        results = self.predict(output, batch_dict, features=features,
                               recon_pred=recon_out, recon_gt=img, recon_mask=recon_mask)
        return results

    @auto_fp16(apply_to=('img',))
    def forward_dummy(self, img, **kwargs):
        output = self.backbone(img)
        output = self.head(output)
        return output
