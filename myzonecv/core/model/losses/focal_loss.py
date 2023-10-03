import torch

from ...registry import LOSSES
from ..base_module import BaseModule


@LOSSES.register_class('focal_loss')
class FocalLoss(BaseModule):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, target):
        loss = self.loss_fcn(pred, target)
        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        prob_pred = torch.sigmoid(pred)
        prob_true = target * prob_pred + (1 - target) * (1 - prob_pred)
        modulating_factor = (1. - prob_true) ** self.gamma
        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
        loss *= modulating_factor * alpha_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


@LOSSES.register_class('qfocal_loss')
class QFocalLoss(BaseModule):
    """ Quality focal loss """

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, target):
        loss = self.loss_fcn(pred, target)
        prob_pred = torch.sigmoid(pred)
        modulating_factor = torch.abs(target - prob_pred) ** self.gamma
        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
        loss *= modulating_factor * alpha_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
