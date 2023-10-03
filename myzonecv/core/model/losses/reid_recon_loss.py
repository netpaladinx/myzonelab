import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registry import LOSSES, REID_LOSSES
from ..base_module import BaseModule
from ..backbones import ResNet
from .gan_loss import GANLoss


@REID_LOSSES.register_class('l1_recon')
class ReIDL1Recon(BaseModule):
    def __init__(self, use_salience_map=False, salience_alpha=1., epsilon=1e-16, loss_weight=1., loss_name='l1_recon_loss', init_cfg=None):
        super().__init__(init_cfg)
        self.use_salience_map = use_salience_map
        self.salience_alpha = salience_alpha
        self.epsilon = epsilon
        self.loss_weight = loss_weight
        assert loss_name.endswith('_loss')
        self.loss_name = loss_name

        self.criterion = nn.L1Loss(reduction='none')

    def get_salience_map(self, gt):
        bs, _, input_h, input_w = gt.shape
        Vmax = torch.max(gt, dim=1, keepdims=True).values  # bs x 1 x input_h x input_w
        Vmin = torch.min(gt, dim=1, keepdims=True).values
        Vdiff = Vmax - Vmin
        S = Vdiff / torch.clamp(torch.max(Vdiff.reshape(bs, 1, -1), dim=-1).values[..., None, None], self.epsilon)
        S = torch.exp(S * self.salience_alpha)
        return S  # bs x 1 x input_h x input_w

    def forward(self, pred, gt, mask=None, train_recon_only=False):
        """ pred: bs x 3 x input_h x input_w
            gt: bs x 3 x input_h x input_w
            mask: bs x input_h x input_w
        """
        recon_loss = self.criterion(pred, gt)  # bs x 3 x input_h x input_w

        salience_map = self.get_salience_map(gt) if self.use_salience_map else None

        if mask is not None:
            salience_map = mask[:, None] if salience_map is None else salience_map * mask[:, None]

        if salience_map is not None:
            recon_loss = recon_loss * salience_map
            recon_loss = recon_loss.sum((2, 3)) / torch.clamp(salience_map.sum((2, 3)), self.epsilon)  # bs x 3

        recon_loss = recon_loss.mean()

        loss = recon_loss * self.loss_weight if not train_recon_only else recon_loss
        return {self.loss_name: loss,
                'll1recon': recon_loss.detach()}


@REID_LOSSES.register_class('gan_recon')
class ReIDGANRecon(BaseModule):
    def __init__(self, out_activation=None, out_scaling=False, dropout=0., dropout_ref=0., use_dynamic_dropout_ref=False,
                 gan_loss_type='vanilla', loss_weight=1., loss_name='gan_recon_loss', init_cfg=None):
        super().__init__(init_cfg)
        assert out_activation in ('tanh', 'relu', None)
        self.out_activation = out_activation
        self.out_scaling = out_scaling
        self.dropout_prob = dropout
        self.dropout_ref_prob = dropout_ref
        self.use_dynamic_dropout_ref = use_dynamic_dropout_ref
        self.gan_loss_type = gan_loss_type
        self.loss_weight = loss_weight
        assert loss_name.endswith('_loss')
        self.loss_name = loss_name

        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.dropout_ref = nn.Dropout(p=self.dropout_ref_prob) if not use_dynamic_dropout_ref else F.dropout

        # Input:
        #   1) positive input: (dropout_ref(input_img), dropout(input_img))
        #   2) negative input: (dropout_ref(input_img), dropout(recon_img))
        self.discriminator = ResNet(50, in_channels=6)
        self.discriminator.preprocess_loaded_state_dict = self._preprocess_loaded_state_dict

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_proj = nn.Linear(self.discriminator.out_channels, 1)
        if out_activation:
            self.out_acti = nn.Tanh() if out_activation == 'tanh' else nn.ReLU()
        if out_scaling:
            self.alpha = nn.parameter.Parameter(torch.tensor(0.))

        self.criterion = GANLoss(gan_loss_type)

    def forward(self, pred, gt, mask=None, train_discriminator=False, train_recon_only=False):
        if mask is not None:
            mask = mask[:, None]
            pred = pred * mask
            gt = gt * mask

        if self.dropout_ref_prob > 0:
            if not self.use_dynamic_dropout_ref:
                ref = self.dropout_ref(gt)
            else:
                ref = self.dropout_ref(gt, p=random.random() * self.dropout_ref_prob, training=self.training)
        else:
            ref = gt

        if train_discriminator:
            return self._forward_for_discriminator(ref, pred, gt)
        else:
            return self._forward_for_generator(ref, pred, train_recon_only=train_recon_only)

    def _forward_for_discriminator(self, ref, pred, gt):
        if self.dropout_prob > 0:
            pred = self.dropout(pred)
            gt = self.dropout(gt)

        bs = ref.shape[0]
        real_input = torch.cat((ref, gt), dim=1)  # bs x 6 x input_h x input_w
        fake_input = torch.cat((ref, pred), dim=1)
        inp = torch.cat((real_input, fake_input), dim=0)  # (bs*2) x 6 x input_h x input_w
        out = self.discriminator(inp)  # (bs*2) x out_channels x out_h x out_w
        out = self.pool(out).reshape(bs * 2, -1)
        out = self.out_proj(out).squeeze(-1)  # (bs*2)
        if self.out_activation:
            out = self.out_acti(out)
        if self.out_scaling:
            out = out * torch.exp(self.alpha)

        target = torch.zeros_like(out)
        target[:bs] = 1

        disc_loss = self.criterion(out, target)
        return {'gan_disc_loss': disc_loss}

    def _forward_for_generator(self, ref, pred, train_recon_only=False):
        if self.dropout_prob > 0:
            pred = self.dropout(pred)

        bs = ref.shape[0]
        fake_input = torch.cat((ref, pred), dim=1)  # bs x 6 x input_h x input_w
        out = self.discriminator(fake_input)  # bs x out_channels x out_h x out_w
        out = self.pool(out).reshape(bs, -1)
        out = self.out_proj(out).squeeze(-1)  # bs
        if self.out_activation:
            out = self.out_acti(out)
        if self.out_scaling:
            out = out * torch.exp(self.alpha)

        target = torch.zeros_like(out)

        recon_loss = -self.criterion(out, target)
        loss = recon_loss * self.loss_weight if not train_recon_only else recon_loss
        return {self.loss_name: loss,
                'lganrecon': recon_loss.detach()}

    @staticmethod
    def _preprocess_loaded_state_dict(discriminator, state_dict):
        if 'conv1.weight' in state_dict and state_dict['conv1.weight'].shape[1] == 3:
            conv1_weight = discriminator.conv1.weight.clone()
            conv1_weight[:, :3].mul_(0.1)                     # for ref
            conv1_weight[:, 3:] = state_dict['conv1.weight']  # for pred or gt
            state_dict['conv1.weight'] = conv1_weight
        return state_dict


@LOSSES.register_class('reid_recon_loss')
class ReIDReconLoss(BaseModule):
    def __init__(self,
                 l1_recon_loss=None,
                 gan_recon_loss=None,
                 loss_weight=1.):
        super().__init__()
        self.l1_recon_loss_cfg = l1_recon_loss or {}
        self.gan_recon_loss_cfg = gan_recon_loss or {}
        self.loss_weight = loss_weight

        loss_weight = self.l1_recon_loss_cfg.get('loss_weight', 1.0) * loss_weight
        self.l1_recon_loss = REID_LOSSES.create(self.l1_recon_loss_cfg, loss_weight=loss_weight)

        loss_weight = self.gan_recon_loss_cfg.get('loss_weight', 1.0) * loss_weight
        self.gan_recon_loss = REID_LOSSES.create(self.gan_recon_loss_cfg, loss_weight=loss_weight)

    def forward(self, pred, gt, mask=None,
                apply_l1_recon=True, apply_gan_recon=True, train_discriminator=False, train_recon_only=False):
        """ pred: bs x 3 x input_h x input_w
            gt: bs x 3 x input_h x input_w
            mask: bs x input_h x input_w
        """
        loss_res = {}

        if self.gan_recon_loss and train_discriminator:
            res = self.gan_recon_loss(pred, gt, mask, train_discriminator=True)
            loss_res.update(res)
        else:
            if self.l1_recon_loss and apply_l1_recon:
                res = self.l1_recon_loss(pred, gt, mask, train_recon_only=train_recon_only)
                loss_res.update(res)

            if self.gan_recon_loss and apply_gan_recon:
                res = self.gan_recon_loss(pred, gt, mask, train_recon_only=train_recon_only)
                loss_res.update(res)

        return loss_res
