import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from ...registry import WARP_LAYERS
from ...utils import npf


def create_warp_layer(acti_cfg, *args, **kwargs):
    acti_cfg = acti_cfg.copy()
    acti_cfg.update(kwargs)
    if args:
        acti_layer = WARP_LAYERS.create(acti_cfg, args)
    else:
        acti_layer = WARP_LAYERS.create(acti_cfg)
    return acti_layer


def fliplr(map2d, flip_pairs=None, flip_dim=-1):
    """ map2d: ... x h x w, or ... x c x h x w if flip_pairs
        flip_paris (list): n_pairs x 2
    """
    if flip_pairs is not None:
        assert map2d.ndim >= 3 and isinstance(flip_pairs, (list, tuple))
        indices = list(range(map2d.size(-3)))
        for left, right in flip_pairs:
            temp = indices[left]
            indices[left] = indices[right]
            indices[right] = temp
        map2d = map2d[..., indices, :, :]

    map2d = torch.flip(map2d, (flip_dim,))
    return map2d


def affine(map2d, translate=(0, 0), rotate=0, scale=1, flip=False, flip_pairs=None, flip_first=True, dst_size=None,
           interpolation=TF.InterpolationMode.BILINEAR, fill=None, order_mode='TRS', use_safe_padding=True):
    """ map2d (torch.Tensor): ... x h x w
        translate: in pixels
        rotate: rotation angle in [-180, 180] (around the center)
        scale: scale ratio (around the center)
        flip (bool): left-right flip
        flip_paris (list): n_pairs x 2
    """
    def _get_padding(src_size, dst_size):
        pad_w, pad_h = int(dst_size[0] - src_size[0]), int(dst_size[1] - src_size[1])
        pad_l = pad_w // 2 if pad_w >= 0 else -(-pad_w // 2)
        pad_t = pad_h // 2 if pad_h >= 0 else -(-pad_h // 2)
        pad_r = pad_w - pad_l
        pad_b = pad_h - pad_t
        padding = (pad_l, pad_r, pad_t, pad_b)
        return padding

    def _get_safe_padding(src_size, dst_size):
        pad_w, pad_h = int(dst_size[0] - src_size[0]), int(dst_size[1] - src_size[1])
        safe_pad_w = max(0, pad_w)
        safe_pad_h = max(0, pad_h)
        pad_w = pad_w - safe_pad_w
        pad_h = pad_h - safe_pad_h
        safe_padding = (safe_pad_w // 2, safe_pad_w - safe_pad_w // 2, safe_pad_h // 2, safe_pad_h - safe_pad_h // 2)
        pad_l = pad_w // 2 if pad_w >= 0 else -(-pad_w // 2)
        pad_t = pad_h // 2 if pad_h >= 0 else -(-pad_h // 2)
        pad_r = pad_w - pad_l
        pad_b = pad_h - pad_t
        rest_padding = (pad_l, pad_r, pad_t, pad_b)
        return safe_padding, rest_padding

    if isinstance(translate, np.ndarray):
        translate = translate.tolist()
    if isinstance(scale, np.ndarray):
        scale = scale.tolist()
    if isinstance(scale, (list, tuple)):
        assert all(np.isclose(s, scale[0]) for s in scale)
        scale = scale[0]

    if flip and flip_first:
        map2d = fliplr(map2d, flip_pairs=flip_pairs)

    src_h, src_w = map2d.size()[-2:]
    if dst_size is None:
        dst_size = npf([src_w, src_h])
    src_size = npf([src_w, src_h])
    if use_safe_padding:
        safe_padding, dst_padding = _get_safe_padding(src_size, dst_size)
        if max(safe_padding) > 0:
            map2d = F.pad(map2d, safe_padding, value=0 if fill is None else fill)
    else:
        dst_padding = _get_padding(src_size, dst_size)

    if order_mode == 'TRS':
        if translate[0] != 0 or translate[1] != 0:
            map2d = TF.affine(map2d, 0, translate, 1, 0, interpolation=interpolation, fill=fill)
        if rotate != 0:
            map2d = TF.affine(map2d, -rotate, (0, 0), 1, 0, interpolation=interpolation, fill=fill)
        if scale != 1:
            map2d = TF.affine(map2d, 0, (0, 0), scale, 0, interpolation=interpolation, fill=fill)
    elif order_mode == 'SRT':
        if scale != 1:
            map2d = TF.affine(map2d, 0, (0, 0), scale, 0, interpolation=interpolation, fill=fill)
        if rotate != 0:
            map2d = TF.affine(map2d, -rotate, (0, 0), 1, 0, interpolation=interpolation, fill=fill)
        if translate[0] != 0 or translate[1] != 0:
            map2d = TF.affine(map2d, 0, translate, 1, 0, interpolation=interpolation, fill=fill)
    else:
        raise ValueError(f'Invalid order_mode: {order_mode}')

    if any(dp != 0 for dp in dst_padding):
        map2d = F.pad(map2d, dst_padding, value=0 if fill is None else fill)

    if flip and not flip_first:
        map2d = fliplr(map2d, flip_pairs=flip_pairs)

    return map2d


def get_affine_mask(map2d, translate=(0, 0), rotate=0, scale=1, flip=False, flip_first=True, dst_size=None,
                    interpolation=TF.InterpolationMode.BILINEAR, order_mode='TRS', use_safe_padding=True):
    mask = torch.ones(map2d.size()[-2:]).to(map2d)
    return affine(mask[None], translate, rotate, scale, flip, None, flip_first, dst_size,
                  interpolation=interpolation, order_mode=order_mode, use_safe_padding=use_safe_padding)[0]


def dual_affine(map2d, translate1=(0, 0), rotate1=0, scale1=1, flip1=False, flip_pairs1=None, flip_first1=True, dst_size1=None,
                scale2=1, rotate2=0, translate2=(0, 0), flip2=False, flip_pairs2=None, flip_first2=True, dst_size2=None,
                interpolation=TF.InterpolationMode.BILINEAR, fill=None, order_mode='TRSSRT', use_safe_padding=True):
    """ map2d (torch.Tensor): ... x h x w
        translate1: in pixels
        rotate1: rotation angle in [-180, 180]
        scale1: scale ratio
        scale2: scale ratio
        rotate2: rotation angle in [-180, 180]
        translate2: in pixels
    """
    if order_mode == 'TRSSRT':
        map2d = affine(map2d, translate1, rotate1, scale1, flip1, flip_pairs1, flip_first1, dst_size1,
                       interpolation=interpolation, fill=fill, order_mode='TRS', use_safe_padding=use_safe_padding)
        map2d = affine(map2d, translate2, rotate2, scale2, flip2, flip_pairs2, flip_first2, dst_size2,
                       interpolation=interpolation, fill=fill, order_mode='SRT', use_safe_padding=use_safe_padding)
    elif order_mode == 'SRTTRS':
        map2d = affine(map2d, translate1, rotate1, scale1, flip1, flip_pairs1, flip_first1, dst_size1,
                       interpolation=interpolation, fill=fill, order_mode='SRT', use_safe_padding=use_safe_padding)
        map2d = affine(map2d, translate2, rotate2, scale2, flip2, flip_pairs2, flip_first2, dst_size2,
                       interpolation=interpolation, fill=fill, order_mode='TRS', use_safe_padding=use_safe_padding)
    else:
        raise ValueError(f'Invalid order_mode: {order_mode}')
    return map2d


def get_dual_affine_mask(map2d, translate1=(0, 0), rotate1=0, scale1=1, flip1=False, flip_first1=True, dst_size1=None,
                         scale2=1, rotate2=0, translate2=(0, 0), flip2=False, flip_first2=True, dst_size2=None,
                         interpolation=TF.InterpolationMode.BILINEAR, order_mode='TRSSRT', use_safe_padding=True):
    mask = torch.ones(map2d.size()[-2:]).to(map2d)
    return dual_affine(mask[None], translate1, rotate1, scale1, flip1, None, flip_first1, dst_size1,
                       scale2, rotate2, translate2, flip2, None, flip_first2, dst_size2,
                       interpolation=interpolation, order_mode=order_mode, use_safe_padding=use_safe_padding)[0]


@WARP_LAYERS.register_class('fliplr')
class Fliplr(nn.Module):
    def __init__(self, flip_pairs=None, flip_dim=-1):
        super().__init__()
        self.flip_pairs = flip_pairs
        self.flip_dim = flip_dim

    def forward(self, x, flip_pairs=None, flip_dim=None):
        if flip_pairs is None:
            flip_pairs = self.flip_pairs
        if flip_dim is None:
            flip_dim = self.flip_dim

        return fliplr(x, flip_pairs, flip_dim)


@WARP_LAYERS.register_class('affine')
class Affine(nn.Module):
    def __init__(self, order_mode='TRS'):
        super().__init__()
        self.order_mode = order_mode

    def forward(self, x, translate=(0, 0), rotate=0, scale=1, interpolation=TF.InterpolationMode.BILINEAR, fill=None):
        return affine(x, translate, rotate, scale, interpolation, fill, self.order_mode)


@WARP_LAYERS.register_class('dual_affine')
class DualAffine(nn.Module):
    def __init__(self, order_mode='TRSSRT'):
        super().__init__()
        self.order_mode = order_mode

    def forward(self, x, translate1=(0, 0), rotate1=0, scale1=1, scale2=1, rotate2=0, translate2=(0, 0), interpolation=TF.InterpolationMode.BILINEAR, fill=None):
        return affine(x, translate1, rotate1, scale1, scale2, rotate2, translate2, interpolation, fill, self.order_mode)
