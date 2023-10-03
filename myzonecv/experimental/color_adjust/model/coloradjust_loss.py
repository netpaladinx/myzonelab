import torch
import torch.nn as nn

from myzonecv.core.model import BaseModule
from ..registry import COLORADJUST_LOSSES
from ..data.coloradjust_consts import (RED_MEAN, RED_STD, GREEN_MEAN, GREEN_STD, BLUE_MEAN, BLUE_STD,
                                       BRIGHTNESS, CONTRAST, SATURATION_MEAN, SATURATION_STD,
                                       RED_MEAN_MARGIN, RED_STD_MARGIN, GREEN_MEAN_MARGIN, GREEN_STD_MARGIN, BLUE_MEAN_MARGIN, BLUE_STD_MARGIN,
                                       BRIGHTNESS_MARGIN, CONTRAST_MARGIN, SATURATION_MEAN_MARGIN, SATURATION_STD_MARGIN)


@COLORADJUST_LOSSES.register_class('stats_distance')
class StatsDistance(BaseModule):
    def __init__(self,
                 red_mean_weight=1.,
                 red_std_weight=1.,
                 green_mean_weight=1.,
                 green_std_weight=1.,
                 blue_mean_weight=1.,
                 blue_std_weight=1.,
                 brightness_weight=1.,
                 contrast_weight=1.,
                 saturation_mean_weight=1.,
                 saturation_std_weight=1.,
                 loss_name='stats_loss'):
        super().__init__()
        self.red_mean = RED_MEAN
        self.red_std = RED_STD
        self.green_mean = GREEN_MEAN
        self.green_std = GREEN_STD
        self.blue_mean = BLUE_MEAN
        self.blue_std = BLUE_STD
        self.brightness = BRIGHTNESS
        self.contrast = CONTRAST
        self.saturation_mean = SATURATION_MEAN
        self.saturation_std = SATURATION_STD
        self.red_mean_margin = RED_MEAN_MARGIN
        self.red_std_margin = RED_STD_MARGIN
        self.green_mean_margin = GREEN_MEAN_MARGIN
        self.green_std_margin = GREEN_STD_MARGIN
        self.blue_mean_margin = BLUE_MEAN_MARGIN
        self.blue_std_margin = BLUE_STD_MARGIN
        self.brightness_margin = BRIGHTNESS_MARGIN
        self.contrast_margin = CONTRAST_MARGIN
        self.saturation_mean_margin = SATURATION_MEAN_MARGIN
        self.saturation_std_margin = SATURATION_STD_MARGIN
        self.red_mean_weight = red_mean_weight
        self.red_std_weight = red_std_weight
        self.green_mean_weight = green_mean_weight
        self.green_std_weight = green_std_weight
        self.blue_mean_weight = blue_mean_weight
        self.blue_std_weight = blue_std_weight
        self.brightness_weight = brightness_weight
        self.contrast_weight = contrast_weight
        self.saturation_mean_weight = saturation_mean_weight
        self.saturation_std_weight = saturation_std_weight
        self.loss_name = loss_name

        self.criterion = nn.MSELoss(reduction='none')

    def _calc_loss(self, pred, target, margin):
        target = torch.full_like(pred, target)
        loss = torch.clamp(self.criterion(pred, target) - margin, 0).mean()
        return loss

    def forward(self, r_mean, r_std, g_mean, g_std, b_mean, b_std, brt, cont, s_mean, s_std):
        r_mean_loss = self._calc_loss(r_mean, self.red_mean, self.red_mean_margin)
        r_std_loss = self._calc_loss(r_std, self.red_std, self.red_std_margin)
        g_mean_loss = self._calc_loss(g_mean, self.green_mean, self.green_mean_margin)
        g_std_loss = self._calc_loss(g_std, self.green_std, self.green_std_margin)
        b_mean_loss = self._calc_loss(b_mean, self.blue_mean, self.blue_mean_margin)
        b_std_loss = self._calc_loss(b_std, self.blue_std, self.blue_std_margin)
        brt_loss = self._calc_loss(brt, self.brightness, self.brightness_margin)
        cont_loss = self._calc_loss(cont, self.contrast, self.contrast_margin)
        s_mean_loss = self._calc_loss(s_mean, self.saturation_mean, self.saturation_mean_margin)
        s_std_loss = self._calc_loss(s_std, self.saturation_std, self.saturation_std_margin)
        loss = r_mean_loss * self.red_mean_weight \
            + r_std_loss * self.red_std_weight \
            + g_mean_loss * self.green_mean_weight \
            + g_std_loss * self.green_std_weight \
            + b_mean_loss * self.blue_mean_weight \
            + b_std_loss * self.blue_std_weight \
            + brt_loss * self.brightness_weight \
            + cont_loss * self.contrast_weight \
            + s_mean_loss * self.saturation_mean_weight \
            + s_std_loss * self.saturation_std_weight
        return {self.loss_name: loss,
                'lrm': r_mean_loss.detach(),
                'lrs': r_std_loss.detach(),
                'lgm': g_mean_loss.detach(),
                'lgs': g_std_loss.detach(),
                'lbm': b_mean_loss.detach(),
                'lbs': b_std_loss.detach(),
                'lbrt': brt_loss.detach(),
                'lcont': cont_loss.detach(),
                'lsm': s_mean_loss.detach(),
                'lsd': s_std_loss.detach()}
