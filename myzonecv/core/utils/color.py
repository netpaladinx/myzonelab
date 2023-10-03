import numpy as np
import cv2

from .image import img_as_float, img_as_ubyte


def cv2_convert(img, code, is_gray=False):
    """ img: HWC or NHWC
    """
    if img.ndim == 4 or (is_gray and img.ndim == 3):
        img = np.stack([cv2.cvtColor(im.astype(np.float32), code) for im in img]).astype(img.dtype)
    else:
        img = cv2.cvtColor(img.astype(np.float32), code).astype(img.dtype)
    return img


def rgb2gray(img, use_cv2=False):
    """ img (np.ndarry, 0~1 float): HWC or NHWC
    """
    assert img.dtype in (np.float32, np.float64)
    assert img.min() >= 0 and img.max() <= 1
    if use_cv2:
        img = cv2_convert(img, cv2.COLOR_RGB2GRAY)
    else:
        R, G, B = np.split(img, 3, -1)  # HW1 or NHW1
        img = R * 0.299 + G * 0.587 + B * 0.114
        img = img.squeeze(-1)
    return img  # HW or NHW


def gray2rgb(img, use_cv2=False):
    """ img (np.ndarry, 0~1 float): HW, HW1, NHW, or NHW1
    """
    assert img.dtype in (np.float32, np.float64)
    assert img.min() >= 0 and img.max() <= 1
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
    if use_cv2:
        img = cv2_convert(img, cv2.COLOR_GRAY2RGB, is_gray=True)
    else:
        img = np.stack((img, img, img), -1)
    return img


def rgb2xyz(img, use_cv2=False):
    """ img (np.ndarry, 0~1 float): HWC or NHWC
    """
    assert img.dtype in (np.float32, np.float64)
    assert img.min() >= 0 and img.max() <= 1
    if use_cv2:
        img = cv2_convert(img, cv2.COLOR_RGB2XYZ)
    else:
        # to linear RGB
        a = img <= 0.04045
        img = img / 12.92 * a + ((img + 0.055) / 1.055)**2.4 * (1 - a)
        R, G, B = np.split(img, 3, -1)  # HW1 or NHW1
        # to XYZ
        X = (R * 0.4124564 + G * 0.3575761 + B * 0.1804375) / 0.95047
        Y = R * 0.2126729 + G * 0.7151522 + B * 0.0721750
        Z = (R * 0.0193339 + G * 0.1191920 + B * 0.9503041) / 1.08883
        img = np.concatenate((X, Y, Z), -1)
    return img


def xyz2rgb(img, use_cv2=False):
    """ img (np.ndarry): HWC or NHWC
    """
    if use_cv2:
        img = cv2_convert(img, cv2.COLOR_XYZ2RGB)
    else:
        # to linear RGB
        X, Y, Z = np.split(img, 3, -1)  # HW1 or NHW1
        X *= 0.95047
        Z *= 1.08883
        R = X * 3.2404542 - Y * 1.5371385 - Z * 0.4985314
        G = - X * 0.9692660 + Y * 1.8760108 + Z * 0.0415560
        B = X * 0.0556434 - Y * 0.2040259 + Z * 1.0572252
        img = np.concatenate((R, G, B), -1)
        # to RGB
        img = np.clip(img, 0, None)
        a = img <= 0.0031308
        img = img * 12.92 * a + (img**(1 / 2.4) * 1.055 - 0.055) * (1 - a)
        img = np.clip(img, 0, 1)
    return img


def xyz2lab(img):
    """ img (np.ndarry): HWC or NHWC

        return: LAB 0 <= L <= 100, -127 <= A,B <= 127
    """
    a = img > (6. / 29)**3
    img = img**(1. / 3) * a + (img * ((29 / 6.)**2 * 1. / 3) + 4. / 29) * (1 - a)
    X, Y, Z = np.split(img, 3, -1)  # HW1 or NHW1
    L = Y * 116 - 16
    A = np.clip((X - Y) * 500, -127, 127)
    B = np.clip((Y - Z) * 200, -127, 127)
    img = np.concatenate((L, A, B), -1)
    return img


def lab2xyz(img):
    """ img (np.ndarry): HWC or NHWC
    """
    L, A, B = np.split(img, 3, -1)
    Y = (L + 16) / 116
    X = Y + A / 500
    Z = Y - B / 200
    img = np.concatenate((X, Y, Z), -1)
    img = np.clip(img, 0, None)
    a = img > 6.0 / 29
    img = img**3 * a + (img - 4. / 29) * 3 * (6. / 29)**2 * (1 - a)
    return img


def lab2lch(img):
    """ img (np.ndarry): HWC or NHWC
    """
    L, A, B = np.split(img, 3, -1)
    C = (A**2 + B**2) ** 0.5
    H = np.arctan2(B, A)
    img = np.concatenate((L, C, H), -1)
    return img


def lch2lab(img):
    """ img (np.ndarry): HWC or NHWC
    """
    L, C, H = np.split(img, 3, -1)
    A = C * np.cos(H)
    B = C * np.sin(H)
    img = np.concatenate((L, A, B), -1)
    return img


def rgb2lab(img, use_cv2=False):
    if use_cv2:
        img = cv2_convert(img, cv2.COLOR_RGB2LAB)
    else:
        img = xyz2lab(rgb2xyz(img, use_cv2=False))
    return img


def lab2rgb(img, use_cv2=False):
    if use_cv2:
        img = cv2_convert(img, cv2.COLOR_LAB2RGB)
    else:
        img = xyz2rgb(lab2xyz(img), use_cv2=False)
    return img


def rgb2lch(img, use_cv2=False):
    return lab2lch(rgb2lab(img, use_cv2=use_cv2))


def lch2rgb(img, use_cv2=False):
    return lab2rgb(lch2lab(img), use_cv2=use_cv2)


def rgb2hsv(img, use_cv2=False, epsilon=np.finfo(float).eps):
    """ img (np.ndarry, 0~1 float): HWC or NHWC

        return: 0 <= V,S <=1, 0 <= H <= 360
    """
    if use_cv2:
        img = cv2_convert(img, cv2.COLOR_RGB2HSV)
    else:
        R, G, B = np.split(img, 3, -1)  # HW1 or NHW1
        V = np.max(img, -1, keepdims=True)
        VI = np.argmax(img, -1, keepdims=True)
        Vmin = np.min(img, -1, keepdims=True)
        S = ((V - Vmin) / V) * (V > epsilon) + np.zeros_like(V) * (V <= epsilon)
        denominator = np.clip(V - Vmin, epsilon, None)
        HR = 60 * (G - B) / denominator
        HG = 120 + 60 * (B - R) / denominator
        HB = 240 + 60 * (R - G) / denominator
        H = HR * (VI == 0) + HG * (VI == 1) + HB * (VI == 2)
        H = (H + 360) * (H < 0) + H * (H >= 0)
        img = np.concatenate((H, S, V), -1)
    return img


def apply_sigmoidal(img, contrast, bias, epsilon=np.finfo(float).eps):
    """ img (np.ndarry, 0~1 float): HWC, RGB
        contrast: float or 
    """
    assert contrast >= 0 and 0 - epsilon < bias < 1 + epsilon
    alpha, beta = bias, contrast
    if alpha == 0:
        alpha = epsilon
    if beta < epsilon:
        beta = epsilon

    if beta > 0:
        img = np.clip(img, 1. / 255, None)
        numerator = 1 / (1 + np.exp(beta * (alpha - img))) - 1 / (1 + np.exp(beta * alpha))
        denominator = 1 / (1 + np.exp(beta * (alpha - 1))) - 1 / (1 + np.exp(beta * alpha))
        img = numerator / denominator
    return img


def apply_gamma(img, gamma):
    """ img (np.ndarry, 0~1 float): HWC, RGB
    """
    if isinstance(gamma, (tuple, list)):
        assert len(gamma) == 3
        img = np.clip(img, 1. / 255, None)
        for i, g in enumerate(gamma):
            if g is not None and g > 0:
                img[..., i] = img[..., i] ** (1.0 / g)
    elif gamma is not None:
        assert gamma > 0
        img = np.clip(img, 1. / 255, None)
        img = img ** (1.0 / gamma)
    return img


def apply_saturation(img, saturation, use_cv2=False, margin=5):
    """ img (np.ndarry, 0~1 float): HWC, RGB
    """
    if saturation != 1:
        img = np.clip(img, 1. / 255, None)
        img = rgb2lab(img, use_cv2=use_cv2)
        img[..., 1] = np.clip(img[..., 1] * saturation, -127 + margin, 127 - margin)
        img[..., 2] = np.clip(img[..., 2] * saturation, -127 + margin, 127 - margin)
        img = lab2rgb(img, use_cv2=use_cv2)
    return img


def get_color_stats(img, use_cv2=False, side_truncation=0.05):
    """ img (np.ndarry, 0~1 float): HWC or NHWC, RGB
    """
    # def _get_mask(a):
    #     if a.ndim == 2:  # HW
    #         vmin = np.quantile(a, side_truncation)
    #         vmax = np.quantile(a, 1 - side_truncation)
    #         return (a >= vmin) & (a <= vmax)
    #     elif a.ndim == 3:  # NHW
    #         vmin = np.quantile(a, side_truncation, (1, 2), keepdims=True)
    #         vmax = np.quantile(a, 1 - side_truncation, (1, 2), keepdims=True)
    #         return (a >= vmin) & (a <= vmax)
    #     else:
    #         ValueError(f"Invalid ndim: {a.ndim}")

    def _get_mask(a):
        if a.ndim == 2:  # HW
            aa = a.reshape(-1)
            beg = np.ceil(aa.shape[0] * side_truncation).astype(int)
            end = aa.shape[0] - beg
            sorted_idx = np.argsort(aa)[beg:end]
            mask = np.zeros_like(aa)
            np.put_along_axis(mask, sorted_idx, 1, -1)
            mask = mask.reshape(a.shape).astype(bool)
            return mask
        elif a.ndim == 3:  # NHW
            aa = a.reshape(a.shape[0], -1)
            beg = np.ceil(aa.shape[1] * side_truncation).astype(int)
            end = aa.shape[1] - beg
            sorted_idx = np.argsort(aa)[:, beg:end]
            mask = np.zeros_like(aa)
            np.put_along_axis(mask, sorted_idx, 1, -1)
            mask = mask.reshape(a.shape).astype(bool)
            return mask
        else:
            ValueError(f"Invalid ndim: {a.ndim}")

    def _get_mean(a, mask=None):
        if a.ndim == 2:  # HW
            return a.mean() if mask is None else a[mask].mean()
        elif a.ndim == 3:  # NHW
            a_sp = [a[i].mean() if mask is None else a[i][mask[i]].mean() for i in range(a.shape[0])]
            return np.stack(a_sp)
        else:
            ValueError(f"Invalid a.ndim: {a.ndim}")

    def _get_std(a, mask=None):
        if a.ndim == 2:  # HW
            return a.std() if mask is None else a[mask].std()
        elif a.ndim == 3:  # NHW
            a_sp = [a[i].std() if mask is None else a[i][mask[i]].std() for i in range(a.shape[0])]
            return np.stack(a_sp)
        else:
            ValueError(f"Invalid a.ndim: {a.ndim}")

    R, G, B = img[..., 0], img[..., 1], img[..., 2]  # HW or NHW
    img_gray = rgb2gray(img, use_cv2=use_cv2)  # HW or NHW
    Gr = img_gray
    img_hsv = rgb2hsv(img, use_cv2=use_cv2)
    S = img_hsv[..., 1]  # HW or NHW

    if side_truncation > 0:
        R_mask = _get_mask(R)
        G_mask = _get_mask(G)
        B_mask = _get_mask(B)
        Gr_mask = _get_mask(Gr)
        S_mask = _get_mask(S)
    else:
        R_mask = None
        G_mask = None
        B_mask = None
        Gr_mask = None
        S_mask = None

    return {'r_mean': _get_mean(R, R_mask),
            'r_std': _get_std(R, R_mask),
            'g_mean': _get_mean(G, G_mask),
            'g_std': _get_std(G, G_mask),
            'b_mean': _get_mean(B, B_mask),
            'b_std': _get_std(B, B_mask),
            'brightness': _get_mean(Gr, Gr_mask),
            'contrast': _get_std(Gr, Gr_mask),
            's_mean': _get_mean(S, S_mask),
            's_std': _get_std(S, S_mask)}


def adjust_color(img, contrast=0, bias=0, red=None, green=None, blue=None, saturation=1):
    img = img_as_float(img)
    img = apply_sigmoidal(img, contrast, bias)
    img = apply_gamma(img, (red, green, blue))
    img = apply_saturation(img, saturation)
    img = img_as_ubyte(img)
    return img
