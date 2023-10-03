import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registry import LOSSES
from ..base_module import BaseModule


def wgan_loss(input, target):
    """ target: same shape as input, with values in {0, 1} or [0, 1] 
    """
    real_mask = target > 0.5
    fake_mask = torch.logical_not(real_mask)
    loss = - input[real_mask].mean() + input[fake_mask].mean()
    return loss


def wgan_logistic_ns_loss(input, target):
    """ WGAN loss in logistically non-saturating mode. Widely used in StyleGANv2. 
    """
    real_mask = target > 0.5
    fake_mask = torch.logical_not(real_mask)
    loss = F.softplus(-input[real_mask]).mean() + F.softplus(input[fake_mask]).mean()
    return loss


def hinge_loss(input, target):
    real_mask = target > 0.5
    fake_mask = torch.logical_not(real_mask)
    loss = F.relu(1 - input[real_mask]).mean() + F.relu(1 + input[fake_mask]).mean()
    return loss


@LOSSES.register_class('gan_loss')
class GANLoss(BaseModule):
    def __init__(self, gan_type='vanilla'):
        super().__init__()
        assert gan_type in ('vanilla', 'lsgan', 'wgan', 'wgan-logistic-ns', 'hinge')
        self.gan_type = gan_type

        if gan_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif gan_type == 'wgan':
            self.criterion = wgan_loss
        elif gan_type == 'wgan-logistic-ns':
            self.criterion = wgan_logistic_ns_loss
        elif gan_type == 'hinge':
            self.criterion = hinge_loss

    def forward(self, input, target):
        """ input: logits
            target (same shape as input): {0, 1} or [0, 1] 
        """
        return self.criterion(input, target)
