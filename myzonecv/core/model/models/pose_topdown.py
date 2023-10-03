import torch

from ...registry import POSE_MODELS, BACKBONES, HEADS, LOSSES, POSTPROCESSORS
from ...utils import auto_fp16
from ..base_model import BaseModel
from ..losses import PoseStableMSE, PoseStableV2MSE


@POSE_MODELS.register_class('topdown')
class PoseTopDown(BaseModel):
    """ 
    Example:
        backbone = {
            'type': 'hrnet',
            'arch': {
                'stage1': {'n_layers': 1, 'n_branches': 1, 'block': 'resnet_bottleneck', 'n_blocks': (4,), 'n_channels': (64,)}
                'stage2': {'n_layers': 1, 'n_branches': 2, 'block': 'resnet_basicblock', 'n_blocks': (4, 4), 'n_channels': (32, 64)}
                'stage3': {'n_layers': 4, 'n_branches': 3, 'block': 'resnet_basicblock', 'n_blocks': (4, 4, 4), 'n_channels': (32, 64, 128)}
                'stage4': {'n_layers': 3, 'n_branches': 4, 'block': 'resnet_basicblock', 'n_blocks': (4, 4, 4, 4), 'n_channels': (32, 64, 128, 256)}
            }
            'in_channels': 3
        }
        head = {
            'type': 'pose_keypoint',
            'arch': {
                'n_deconv_layers': 0, 'n_deconv_filters': (), 'n_deconv_kernels': (),
                'n_conv_layers': 0, 'n_conv_filters': (), 'n_conv_kernels': (), 'final_conv_kernel': 1
            },
            'in_channels': 32,
            'out_channels': 17
        }
        loss = {
            'type': 'pose_loss.mse',
            'use_target_weights': True
        }
        eval_cfg = {
            'flip_test': True,
            'postprocess': 'default',
            'modulate_kernel': 11,
            'shift_heatmap': True
        }
        init_cfg = {
            'type': pretrained,
            'path': ...
        }
    """

    def __init__(self,
                 backbone,
                 keypoint_head,
                 # feature_head=None,  # deprecated
                 loss=None,
                 stable_loss=None,
                 smooth_loss=None,
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
        self.keypoint_head_cfg = keypoint_head
        # self.feature_head_cfg = feature_head
        self.loss_cfg = loss
        self.stable_loss = stable_loss
        self.smooth_loss = smooth_loss
        self.accuracy_cfg = accuracy
        self.predict_cfg = predict

        self.backbone = BACKBONES.create(self.backbone_cfg)
        self.keypoint_head = HEADS.create(self.keypoint_head_cfg)
        # self.feature_head = HEADS.create(self.feature_head_cfg)
        self.loss = LOSSES.create(self.loss_cfg)
        self.stable_loss = LOSSES.create(self.stable_loss)
        self.smooth_loss = LOSSES.create(self.smooth_loss)
        self.accuracy = POSTPROCESSORS.create(self.accuracy_cfg)
        self.predict = POSTPROCESSORS.create(self.predict_cfg)

        self.init_weights()

    def compute_loss(self, output, targets, batch_dict=None, accuracy_results=None, target_loss=None):
        """ output: bs x kpts x heatmap_h x heatmap_w
        """
        loss_res = {}

        if self.match_loss('loss', target_loss):
            target_heatmaps = targets['target_heatmaps']
            target_weights = targets.get('target_weights')
            assert target_heatmaps.dim() == 4 and (target_weights is None or target_weights.dim() == 3)
            assert target_heatmaps.shape == output.shape

            dist_weights = None
            if accuracy_results:
                dist_weights = (accuracy_results['pixel_dist'].T + 1).clip(0)  # bs x kpts
                dist_weights = torch.from_numpy(dist_weights).to(output)

            loss = self.loss(output, target_heatmaps, target_weights, dist_weights=dist_weights)
            loss_res.update(loss)

        if self.stable_loss and self.match_loss('stable_loss', target_loss):
            if isinstance(self.stable_loss, PoseStableMSE):
                assert batch_dict is not None
                center = batch_dict['center']
                scale = batch_dict['scale']
                flipped = batch_dict['flipped']
                rotate = batch_dict['rotate']
                params = batch_dict['params']
                flip_pairs = batch_dict['flip_pairs']
                heatmap_reg = self.keypoint_head.heatmap_reg
                stable_loss = self.stable_loss(output, center, scale, flipped, rotate, params, flip_pairs, heatmap_reg)
                loss_res.update(stable_loss)

            elif isinstance(self.stable_loss, PoseStableV2MSE):
                assert batch_dict is not None
                flipped = batch_dict['flipped']
                warp_mat_inv = batch_dict['mat_inv']
                warp_mat = batch_dict['mat']
                params = batch_dict['params']
                flip_pairs = batch_dict['flip_pairs']
                stable_loss = self.stable_loss(output, flipped, warp_mat_inv, warp_mat, params, flip_pairs)
                loss_res.update(stable_loss)

        return loss_res

    def compute_accuracy(self, output, targets):
        acc_res = {}

        if 'target_heatmaps' in targets:
            target_heatmaps = targets['target_heatmaps']
            target_weights = targets.get('target_weights')
            output = output.detach().cpu().numpy()
            target_heatmaps = target_heatmaps.detach().cpu().numpy()
            mask = target_weights.detach().cpu().numpy().squeeze(-1) > 0 if target_weights is not None else target_weights

            acc = self.accuracy(output, target_heatmaps, mask=mask)
            acc_res.update(acc)

        return acc_res

    @auto_fp16(apply_to=('inputs',))
    def forward_train(self, inputs, targets, batch_dict=None, target_loss=None, **kwargs):
        """ inputs (dict)
                'img': bs x img_c x img_h x img_w
            targets (dict)
                'target_heatmaps': bs x kpts x heatmap_h x heatmap_w
                'target_weights': bs x kpts x 1
        """
        img = inputs['img']

        output = self.backbone(img)
        output = self.keypoint_head(output, skip_heatmap_reg=not self.match_loss('loss', target_loss))

        results = {}
        summary_keys = []

        acc_res = self.compute_accuracy(output, targets)
        summary_keys += self.collect_summary(acc_res)
        results.update(acc_res)

        loss_res = self.compute_loss(output, targets, batch_dict=batch_dict, accuracy_results=acc_res, target_loss=target_loss)
        summary_keys += self.collect_summary(loss_res)
        results.update(loss_res)

        results = self.merge_losses(results)
        results['summary_keys'] = summary_keys
        return results

    @auto_fp16(apply_to=('inputs',))
    def forward_predict(self,
                        inputs,
                        batch_dict,
                        return_heatmaps=False,
                        flip_test=False,
                        postprocess='default',
                        modulate_kernel=11,
                        shift_heatmap=False,
                        # return_feature=True,
                        **kwargs):
        img = inputs['img']

        output = self.backbone(img)

        results = {}

        # if return_feature and self.feature_head is not None:
        #     feature_output = self.feature_head(output)
        #     results['pred_feature'] = feature_output

        output = self.keypoint_head.forward_predict(output)

        if flip_test:
            flip_pairs = batch_dict['flip_pairs']
            img_flipped = img.flip(3)
            output_flipped = self.backbone(img_flipped)
            output_flipped_back = self.keypoint_head.forward_predict(output_flipped, flip_pairs, shift_heatmap)
            output = (output + output_flipped_back) / 2.

        results.update(self.predict(output, batch_dict, postprocess, modulate_kernel))
        if return_heatmaps:
            results['pred_heatmaps'] = output
        return results

    @auto_fp16(apply_to=('img',))
    def forward_dummy(self,
                      img,
                      # return_feature=False
                      ):
        output = self.backbone(img)

        # feature_output = None
        # if return_feature and self.feature_head is not None:
        #     feature_output = self.feature_head(output)

        output = self.keypoint_head(output)

        return output  # if feature_output is None else (feature_output, output)
