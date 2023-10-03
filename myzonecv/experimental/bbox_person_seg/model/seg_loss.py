import torch
import torch.nn.functional as F

from myzonecv.core.model import BaseModule
from myzonecv.core.model.losses import CrossEntropy, BinaryCrossEntropy
from ..registry import SEG_LOSSES


@SEG_LOSSES.register_class('cross_entropy')
class SegCrossEntropy(BaseModule):
    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 ignore_index=-100,
                 avg_non_ignore=True,
                 avg_factor=None,
                 class_weight=None,
                 pos_weight=None,
                 loss_weight=1.,
                 align_corners=False,
                 use_hard_sampling=False,
                 sample_threshold=None,
                 max_sample_pixels=100000,  # out of H*W
                 loss_name='ce_loss'):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        self.avg_factor = avg_factor
        self.class_weight = class_weight
        self.pos_weight = pos_weight
        self.loss_weight = loss_weight
        self.align_corners = align_corners
        self.use_hard_sampling = use_hard_sampling
        self.sample_threshold = sample_threshold
        self.max_sample_pixels = max_sample_pixels
        self.loss_name = loss_name

        if use_sigmoid:
            self.criterion = BinaryCrossEntropy(reduction=reduction,
                                                ignore_index=ignore_index,
                                                avg_non_ignore=avg_non_ignore,
                                                avg_factor=avg_factor,
                                                pos_weight=pos_weight)
        else:
            self.criterion = CrossEntropy(reduction=reduction,
                                          ignore_index=ignore_index,
                                          avg_non_ignore=avg_non_ignore,
                                          avg_factor=avg_factor,
                                          class_weight=class_weight)

    def forward(self, seg_pred, seg_target, target_weight=None, mask_index=None, reduction=None):
        """ seg_pred (logits): N x C x H x W
            seg_target: N x H x W (dtype: torch.int64) or N x C x H x W (dtype: torch.float32)
            target_weight: shape as seg_target
            mask_index: N (dtype: torch.int64) (contains selected class indices)
        """
        if reduction is None:
            reduction = self.reduction

        pred_h, pred_w = seg_pred.shape[-2:]
        tar_h, tar_w = seg_target.shape[-2:]
        if pred_h != tar_h or pred_w != tar_w:
            seg_pred = F.interpolate(seg_pred, size=(tar_h, tar_w), mode='bilinear', align_corners=self.align_corners)

        if self.use_hard_sampling:
            seg_weight = self.sample(seg_pred, seg_target)
            if target_weight is None:
                target_weight = seg_weight
            else:
                target_weight *= seg_weight

        if self.use_sigmoid:
            loss = self.criterion(seg_pred, seg_target, target_weight, self.pos_weight, mask_index if self.use_mask else None, reduction)
        else:
            loss = self.criterion(seg_pred, seg_target, target_weight, self.class_weight, reduction)

        loss = loss * self.loss_weight
        return {self.loss_name: loss}

    def sample(self, seg_pred, seg_target):
        """ Online hard example sampling: sample pixels that have high loss or with low prediction confidence

            seg_pred (logits): N x C x H x W
            seg_target: N x H x W (dtype: torch.int64)
        """
        with torch.no_grad():
            assert seg_pred.shape[-2:] == seg_target.shape[-2:]

            max_pixels = self.max_sample_pixels * seg_target.shape[0]
            valid_mask = (seg_target != self.ignore_index)

            seg_weight = seg_pred.new_zeros(seg_target.shape)
            valid_seg_weight = seg_weight[valid_mask]

            if self.sample_threshold is not None:
                seg_prob = F.softmax(seg_pred, dim=1)  # N x C x H x W
                seg_prob = seg_prob.gather(1, seg_target[:, None]).squeeze(1)
                sort_prob, _ = seg_prob[valid_mask].sort()

                if sort_prob.numel() > 0:
                    min_thr = sort_prob[min(max_pixels, sort_prob.numel() - 1)]
                else:
                    min_thr = 0.0
                sample_thr = max(min_thr, self.sample_threshold)
                valid_seg_weight[seg_prob[valid_mask] < sample_thr] = 1.  # low prediction confidence
            else:
                seg_loss = self(seg_pred, seg_target, reduction='none')
                _, sort_index = seg_loss[valid_mask].sort(descending=True)
                valid_seg_weight[sort_index[:max_pixels]] = 1.  # high loss

            seg_weight[valid_mask] = valid_seg_weight
            return seg_weight  # N x H x W
