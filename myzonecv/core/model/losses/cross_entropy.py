import torch
import torch.nn.functional as F

from ...registry import LOSSES
from ...utils import weight_reduce_loss
from ..base_module import BaseModule


@LOSSES.register_class('cross_entropy')
class CrossEntropy(BaseModule):
    def __init__(self, reduction='mean', ignore_index=-100, avg_non_ignore=True, avg_factor=None, class_weight=None):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        self.avg_factor = avg_factor
        self.class_weight = class_weight

    def forward(self, pred, target, target_weight=None, class_weight=None, reduction=None):
        """ pred: N x C or N x C x H x W (C: number of classes, pred: logits)
            target: N or N x H x W (dtype: torch.int64)
            target_weight: shape as target
            class_weight: C
        """
        if class_weight is None:
            class_weight = self.class_weight
        if reduction is None:
            reduction = self.reduction

        loss = F.cross_entropy(pred, target, weight=class_weight, reduction='none', ignore_index=self.ignore_index)

        avg_factor = self.avg_factor
        if self.avg_non_ignore and reduction == 'mean' and avg_factor is None:
            avg_factor = target.numel() - (target == self.ignore_index).sum()

        loss = weight_reduce_loss(loss, target_weight, reduction, avg_factor)
        return loss


@LOSSES.register_class('binary_cross_entropy')
class BinaryCrossEntropy(BaseModule):
    def __init__(self, reduction='mean', ignore_index=-100, avg_non_ignore=True, avg_factor=None, pos_weight=None):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        self.avg_factor = avg_factor
        self.pos_weight = pos_weight

    def forward(self, pred, target, target_weight=None, pos_weight=None, mask_index=None, reduction=None):
        """ pred: N x C or N x C x H x W (pred: logits)
            target: 1) N or N x H x W (dtype: torch.int64)
                    2) N x C or N x C x H x W (dtype: torch.float32)
            target_weight: shape as target
            pos_weight: broadcastable with target
            mask_index: N (dtype: torch.int64) (contains selected class indices)
        """
        if pred.ndim != target.ndim:
            assert (pred.ndim == 2 and target.ndim == 1) or (pred.ndim == 4 and target.ndim == 3)
            target, target_weight, valid_mask = self._expand_target_as_pred(target, target_weight, pred.shape, self.ignore_index)
        else:
            if self.ignore_index >= 0:
                valid_mask = target[:, self.ignore_index, ...] > 0
                valid_mask = valid_mask[:, None].expand(target.shape).float()
            else:
                valid_mask = target.new_ones(target.shape)

            if target_weight is None:
                target_weight = valid_mask
            else:
                target_weight *= valid_mask

        if pos_weight is None:
            pos_weight = self.pos_weight
        if reduction is None:
            reduction = self.reduction

        if mask_index is not None:
            n_target = target.shape[0]
            indices = torch.arange(0, n_target, device=target.device)
            if pos_weight is not None:
                pos_weight = pos_weight.expand(target.shape)[indices, mask_index]
            pred = pred[indices, mask_index]
            target = target[indices, mask_index]
            valid_mask = valid_mask[indices, mask_index]

        loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight, reduction='none')

        avg_factor = self.avg_factor
        if self.avg_non_ignore and reduction == 'mean' and avg_factor is None:
            avg_factor = valid_mask.sum()

        loss = weight_reduce_loss(loss, target_weight, reduction, avg_factor)

        return loss

    @staticmethod
    def _expand_target_as_pred(target, target_weight, pred_shape, ignore_index):
        """ target: N or N x H x W (dtype is torch.int64)
            target_weight: shape as target
        """
        bin_target = target.new_zeros(pred_shape).float()  # N x C or N x C x H x W
        valid_mask = (target >= 0) & (target != ignore_index)
        indices = torch.nonzero(valid_mask, as_tuple=True)  # Todo: avoid unnecessary CPU-GPU synchronization

        if indices[0].numel() > 0:
            if target.ndim == 1:  # N
                bin_target[indices[0], target[valid_mask]] = 1
            elif target.ndim == 3:  # N x H x W
                bin_target[indices[0], target[valid_mask], indices[1], indices[2]] = 1
            else:
                raise ValueError(f"target.ndim should be 1 or 3")

        valid_mask = valid_mask[:, None].expand(pred_shape).float()

        if target_weight is None:
            bin_target_weight = valid_mask
        else:
            bin_target_weight = target_weight[:, None].expand(pred_shape)
            bin_target_weight *= valid_mask

        return bin_target, bin_target_weight, valid_mask
