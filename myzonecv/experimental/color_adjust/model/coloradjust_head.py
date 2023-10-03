import numpy as np
import torch

from myzonecv.core.model import BaseModule
from ..registry import COLORADJUST_HEADS
from ..data.coloradjust_consts import SIDE_TRUNCATION, CONTRAST_RANGE, BIAS_RANGE, GAMMA_RANGE, SATURATION_RANGE


class ColorAdjustOps:
    @staticmethod
    def _split(img, squeeze=False):
        sp = torch.split(img, 1, 0) if img.ndim == 3 else torch.split(img, 1, 1)  # 1HW or N1HW
        if squeeze:
            sp = [s.squeeze(0) if s.ndim == 3 else s.squeeze(1) for s in sp]
        return sp

    @staticmethod
    def _cat(C1, C2, C3):
        return torch.cat((C1, C2, C3), 0) if C1.ndim == 3 else torch.cat((C1, C2, C3), 1)  # CHW or NCHW

    @staticmethod
    def _cdim(img):
        return 0 if img.ndim == 3 else 1

    @staticmethod
    def rgb2xyz(img):
        """ img: CHW or NCHW, (RGB, 0 ~ 1) 
        """
        assert img.ndim in (3, 4)
        # to linear RGB
        a = img.detach() <= 0.04045
        img = img / 12.92 * a + ((img + 0.055) / 1.055)**2.4 * torch.logical_not(a)
        R, G, B = ColorAdjustOps._split(img)
        # to XYZ
        X = (R * 0.4124564 + G * 0.3575761 + B * 0.1804375) / 0.95047
        Y = R * 0.2126729 + G * 0.7151522 + B * 0.0721750
        Z = (R * 0.0193339 + G * 0.1191920 + B * 0.9503041) / 1.08883
        img = ColorAdjustOps._cat(X, Y, Z)
        return img  # 0 ~ 1

    @staticmethod
    def xyz2rgb(img):
        """ img: CHW or NCHW, (XYZ, 0 ~ 1) 
        """
        assert img.ndim in (3, 4)
        # to linear RGB
        X, Y, Z = ColorAdjustOps._split(img)
        X = X * 0.95047
        Z = Z * 1.08883
        R = X * 3.2404542 - Y * 1.5371385 - Z * 0.4985314
        G = - X * 0.9692660 + Y * 1.8760108 + Z * 0.0415560
        B = X * 0.0556434 - Y * 0.2040259 + Z * 1.0572252
        img = ColorAdjustOps._cat(R, G, B)
        # to RGB
        img = torch.clamp(img, 0)
        a = img.detach() <= 0.0031308
        img = img * 12.92 * a + (img**(1 / 2.4) * 1.055 - 0.055) * torch.logical_not(a)
        img = torch.clamp(img, 0, 1)
        return img  # 0 ~ 1

    @staticmethod
    def xyz2lab(img):
        """ img: CHW or NCHW, (XYZ, 0 ~ 1) 
        """
        assert img.ndim in (3, 4)
        a = img.detach() > (6. / 29)**3
        img = img**(1. / 3) * a + (img * ((29 / 6.)**2 * 1. / 3) + 4. / 29) * torch.logical_not(a)
        X, Y, Z = ColorAdjustOps._split(img)
        L = Y * 116 - 16
        A = torch.clamp((X - Y) * 500, -127, 127)
        B = torch.clamp((Y - Z) * 200, -127, 127)
        img = ColorAdjustOps._cat(L, A, B)
        return img  # LAB, 0 <= L <= 100, -127 <= A,B <= 127

    @staticmethod
    def lab2xyz(img):
        """ img: CHW or NCHW (LAB, 0 <= L <= 100, -127 <= A,B <= 127)
        """
        assert img.ndim in (3, 4)
        L, A, B = ColorAdjustOps._split(img)
        Y = (L + 16) / 116
        X = Y + A / 500
        Z = Y - B / 200
        img = ColorAdjustOps._cat(X, Y, Z)
        img = torch.clamp(img, 0)
        a = img.detach() > 6.0 / 29
        img = img**3 * a + (img - 4. / 29) * 3 * (6. / 29)**2 * torch.logical_not(a)
        return img  # 0 ~ 1

    @staticmethod
    def rgb2gray(img):
        """ img: CHW or NCHW, (RGB, 0 ~ 1) 
        """
        assert img.ndim in (3, 4)
        R, G, B = ColorAdjustOps._split(img)
        img = R * 0.299 + G * 0.587 + B * 0.114
        img = img.squeeze(0) if img.ndim == 3 else img.squeeze(1)
        return img  # HW or NHW, 0 ~ 1

    @staticmethod
    def rgb2hsv(img, epsilon=1e-10):
        """ img: CHW or NCHW, (RGB, 0 ~ 1) 
        """
        assert img.ndim in (3, 4)
        R, G, B = ColorAdjustOps._split(img)
        V, VI = torch.max(img, dim=ColorAdjustOps._cdim(img), keepdims=True)
        Vmin, _ = torch.min(img, dim=ColorAdjustOps._cdim(img), keepdims=True)
        a = V.detach() > epsilon
        S = ((V - Vmin) / V) * a + torch.zeros_like(V) * torch.logical_not(a)
        denominator = torch.clamp(V - Vmin, epsilon)
        HR = 60 * (G - B) / denominator
        HG = 120 + 60 * (B - R) / denominator
        HB = 240 + 60 * (R - G) / denominator
        H = HR * (VI == 0) + HG * (VI == 1) + HB * (VI == 2)
        H = (H + 360) * (H < 0) + H * (H >= 0)
        img = ColorAdjustOps._cat(H, S, V)
        return img  # HSV, 0 <= V,S <=1, 0 <= H <= 360

    @staticmethod
    def _asimg(x, img):
        if img.ndim == 3:
            return x.reshape(1, 1, 1)
        elif img.ndim == 4:
            return x.reshape(img.shape[0], 1, 1, 1)
        else:
            raise ValueError(f"Invalid img.ndim: {img.ndim}")

    @staticmethod
    def apply_sigmoidal(img, contrast, bias, epsilon=1e-16):
        """ img: CHW or NCHW, (RGB, 0 ~ 1)
            contrast, bias: bs
        """
        assert img.ndim in (3, 4)
        img = torch.clamp(img, 1. / 255)
        contrast = torch.clamp(ColorAdjustOps._asimg(contrast, img), epsilon)  # bs x 1 x 1 x 1 if img is NCHW
        bias = torch.clamp(ColorAdjustOps._asimg(bias, img), epsilon)  # bs x 1 x 1 x 1 if img is NCHW
        numerator = torch.sigmoid((img - bias) * contrast) - torch.sigmoid((- bias) * contrast)
        denominator = torch.sigmoid((1 - bias) * contrast) - torch.sigmoid((- bias) * contrast)
        img = numerator / denominator
        return img

    @staticmethod
    def apply_gamma(img, red, green, blue):
        """ img: CHW or NCHW, (RGB, 0 ~ 1)
            red, green, blue: bs
        """
        assert img.ndim in (3, 4)
        img = torch.clamp(img, 1. / 255)
        red = ColorAdjustOps._asimg(red, img)
        green = ColorAdjustOps._asimg(green, img)
        blue = ColorAdjustOps._asimg(blue, img)
        R, G, B = ColorAdjustOps._split(img)  # 1HW or N1HW
        R = R ** (1. / red)
        G = G ** (1. / green)
        B = B ** (1. / blue)
        img = ColorAdjustOps._cat(R, G, B)
        return img

    @staticmethod
    def apply_saturation(img, saturation, margin=5):
        """ img: CHW or NCHW, (RGB, 0 ~ 1)
            saturation: bs
        """
        assert img.ndim in (3, 4)
        img = torch.clamp(img, 1. / 255)
        saturation = ColorAdjustOps._asimg(saturation, img)
        img = ColorAdjustOps.rgb2xyz(img)
        img = ColorAdjustOps.xyz2lab(img)
        L, A, B = ColorAdjustOps._split(img)
        A = torch.clamp(A * saturation, -127 + margin, 127 - margin)
        B = torch.clamp(B * saturation, -127 + margin, 127 - margin)
        img = ColorAdjustOps._cat(L, A, B)
        img = ColorAdjustOps.lab2xyz(img)
        img = ColorAdjustOps.xyz2rgb(img)
        return img

    # @staticmethod
    # def _get_mask(x, side_truncation):
    #     if x.ndim == 2:  # HW
    #         vmin = torch.quantile(x, side_truncation)
    #         vmax = torch.quantile(x, 1 - side_truncation)
    #         return (x >= vmin) & (x <= vmax)
    #     elif x.ndim == 3:  # NHW
    #         N = x.shape[0]
    #         vmin = torch.quantile(x.reshape(N, -1), side_truncation, dim=1).reshape(N, 1, 1)
    #         vmax = torch.quantile(x.reshape(N, -1), 1 - side_truncation, dim=1).reshape(N, 1, 1)
    #         return (x >= vmin) & (x <= vmax)
    #     else:
    #         ValueError(f"Invalid x.ndim: {x.ndim}")

    @staticmethod
    def _get_mask(x, side_truncation):
        if x.ndim == 2:  # HW
            xx = x.reshape(-1)
            beg = np.ceil(xx.shape[0] * side_truncation).astype(int)
            end = xx.shape[0] - beg
            sorted_idx = torch.argsort(xx)[beg:end]
            mask = torch.zeros_like(xx)
            mask.scatter_(0, sorted_idx, 1)
            mask = mask.reshape(x.shape).to(dtype=bool)
            return mask
        elif x.ndim == 3:  # NHW
            xx = x.reshape(x.shape[0], -1)
            beg = np.ceil(xx.shape[1] * side_truncation).astype(int)
            end = xx.shape[1] - beg
            sorted_idx = torch.argsort(xx)[:, beg:end]
            mask = torch.zeros_like(xx)
            mask.scatter_(1, sorted_idx, 1)
            mask = mask.reshape(x.shape).to(dtype=bool)
            return mask
        else:
            ValueError(f"Invalid ndim: {x.ndim}")

    @staticmethod
    def _get_mean(x, mask=None):
        if x.ndim == 2:  # HW
            return x.mean() if mask is None else x[mask].mean()
        elif x.ndim == 3:  # NHW
            x_sp = [x[i].mean() if mask is None else x[i][mask[i]].mean() for i in range(x.shape[0])]
            return torch.stack(x_sp)
        else:
            ValueError(f"Invalid x.ndim: {x.ndim}")

    @staticmethod
    def _get_std(x, mask=None):
        if x.ndim == 2:  # HW
            return x.std(unbiased=False) if mask is None else x[mask].std(unbiased=False)
        elif x.ndim == 3:  # NHW
            x_sp = [x[i].std(unbiased=False) if mask is None else x[i][mask[i]].std(unbiased=False) for i in range(x.shape[0])]
            return torch.stack(x_sp)
        else:
            ValueError(f"Invalid x.ndim: {x.ndim}")

    @staticmethod
    def get_color_stats(img, side_truncation=0.05):
        """ img: CHW or NCHW, (RGB, 0 ~ 1)
        """
        R, G, B = ColorAdjustOps._split(img, squeeze=True)  # HW or NHW
        Gr = ColorAdjustOps.rgb2gray(img)
        img_hsv = ColorAdjustOps.rgb2hsv(img)
        S = img_hsv[1] if img_hsv.ndim == 3 else img_hsv[:, 1]

        R_mask, G_mask, B_mask, Gr_mask, S_mask = None, None, None, None, None
        if side_truncation > 0:
            R_mask = ColorAdjustOps._get_mask(R, side_truncation)
            G_mask = ColorAdjustOps._get_mask(G, side_truncation)
            B_mask = ColorAdjustOps._get_mask(B, side_truncation)
            Gr_mask = ColorAdjustOps._get_mask(Gr, side_truncation)
            S_mask = ColorAdjustOps._get_mask(S, side_truncation)

        return {'r_mean': ColorAdjustOps._get_mean(R, R_mask),
                'r_std': ColorAdjustOps._get_std(R, R_mask),
                'g_mean': ColorAdjustOps._get_mean(G, G_mask),
                'g_std': ColorAdjustOps._get_std(G, G_mask),
                'b_mean': ColorAdjustOps._get_mean(B, B_mask),
                'b_std': ColorAdjustOps._get_std(B, B_mask),
                'brightness': ColorAdjustOps._get_mean(Gr, Gr_mask),
                'contrast': ColorAdjustOps._get_std(Gr, Gr_mask),
                's_mean': ColorAdjustOps._get_mean(S, S_mask),
                's_std': ColorAdjustOps._get_std(S, S_mask)}


@COLORADJUST_HEADS.register_class('adjust_image')
class AdjustImage(BaseModule):
    def __init__(self, epsilon=1e-10):
        super().__init__()
        self.contrast_range = CONTRAST_RANGE
        self.bias_range = BIAS_RANGE
        self.gamma_range = GAMMA_RANGE
        self.saturation_range = SATURATION_RANGE
        self.side_truncation = SIDE_TRUNCATION
        self.epsilon = epsilon

    def forward(self, imgs, x):
        """ imgs: list(C x Hi x Wi), 0 ~ 1
            x: bs x n_dims (n_dims = 6)
        """
        out = torch.sigmoid(x)
        contrast = out[:, 0] * (self.contrast_range[1] - self.contrast_range[0]) + self.contrast_range[0]  # bs
        bias = out[:, 1] * (self.bias_range[1] - self.bias_range[0]) + self.bias_range[0]  # bs
        red = out[:, 2] * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]  # bs
        green = out[:, 3] * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]  # bs
        blue = out[:, 4] * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]  # bs
        saturation = out[:, 5] * (self.saturation_range[1] - self.saturation_range[0]) + self.saturation_range[0]  # bs

        out_imgs = []
        for i, img in enumerate(imgs):
            img = ColorAdjustOps.apply_sigmoidal(img, contrast[i], bias[i])
            img = ColorAdjustOps.apply_gamma(img, red[i], green[i], blue[i])
            img = ColorAdjustOps.apply_saturation(img, saturation[i])
            out_imgs.append(img)

        stats_arr = [ColorAdjustOps.get_color_stats(img, self.side_truncation) for img in out_imgs]
        out_stats = {}
        for key in stats_arr[0]:
            out_stats[key] = torch.stack([stats[key] for stats in stats_arr])

        op_params = {
            'contrast': contrast, 'bias': bias,
            'red': red, 'green': green, 'blue': blue,
            'saturation': saturation
        }
        return out_imgs, out_stats, op_params
