import numpy as np
import cv2

from ...registry import REID_TRANSFORMS
from ...utils import TruncNorm
from ..datautils import cs2bbox, npf, get_warp_matrix, get_affine_matrix, apply_warp_to_coord, apply_warp_to_map2d
from ..dataconsts import REID_BBOX_SCALE_UNIT, REID_BORDER_COLOR_VALUE


def warp(img, center, scale, rotate, dst_size, kpts=None, mask=None, use_unbiased_processing=False, img2=None):
    if use_unbiased_processing:
        mat = get_warp_matrix(center, scale, dst_size, rotate=rotate, scale_unit=REID_BBOX_SCALE_UNIT)
    else:
        mat = get_affine_matrix(center, scale, dst_size, rotate=rotate, scale_unit=REID_BBOX_SCALE_UNIT)

    img = apply_warp_to_map2d(img, mat, dst_size, flags=cv2.INTER_LINEAR, border_value=REID_BORDER_COLOR_VALUE)
    if img2 is not None:
        img2 = apply_warp_to_map2d(img2, mat, dst_size, flags=cv2.INTER_LINEAR, border_value=REID_BORDER_COLOR_VALUE)

    if kpts is not None:
        kpts[:, :2] = apply_warp_to_coord(kpts[:, :2], mat)
        kpts[kpts[:, 2] == 0] = 0

    if mask is not None:
        mask = apply_warp_to_map2d(mask.astype(float), mat, dst_size, flags=cv2.INTER_AREA, border_value=0) > 0.5

    return img, kpts, mask, img2


def flip(img, center, kpts=None, mask=None, flip_pairs=None, img2=None):
    # flip img
    img = np.fliplr(img)
    if img2 is not None:
        img2 = np.fliplr(img2)

    # flip center
    img_width = img.shape[1]
    center[0] = img_width - 1 - center[0]

    # flip kpts
    if kpts is not None:
        kpts[:, 0] = img_width - 1 - kpts[:, 0]
        for left, right in flip_pairs:
            temp = kpts[left].copy()
            kpts[left] = kpts[right]
            kpts[right] = temp

    # flip mask
    if mask is not None:
        mask = np.fliplr(mask)

    return img, center, kpts, mask, img2


@REID_TRANSFORMS.register_class('batch_warp')
class BatchWarp:
    def __init__(self, use_unbiased_processing=False):
        self.use_unbiased_processing = use_unbiased_processing

    def __call__(self, input_batch, dataset, step):
        input_size = dataset.input_size

        for input_dict in input_batch:
            img = input_dict['img']
            center = input_dict['center']
            scale = input_dict['scale']
            rotate = input_dict.get('rotate', 0)
            kpts = input_dict.get('kpts')
            mask = input_dict.get('mask')

            img, kpts, mask, _ = warp(img, center, scale, rotate, input_size, kpts, mask, self.use_unbiased_processing)

            input_dict['img'] = img
            input_dict['kpts'] = kpts
            input_dict['mask'] = mask

        return input_batch


@REID_TRANSFORMS.register_class('batch_random_warp')
class BatchRandomWarp:
    _MIN_SCALE_FACTOR = 0.1

    def __init__(self, flip_prob=0.5,
                 translate_prob=1.0, translate_factor=None, translate_sigma=1. / 3,
                 scale_prob=1.0, scale_factor=None, scale_sigma=1.,
                 rotate_prob=0.5, rotate_factor=30, rotate_sigma=1. / 2,
                 use_bbox_aspect_ratio=False, use_unbiased_processing=False):
        self.flip_prob = flip_prob
        self.translate_prob = translate_prob
        self.translate_factor = translate_factor
        self.translate_sigma = translate_sigma
        self.scale_prob = scale_prob
        self.scale_factor = scale_factor
        self.scale_sigma = scale_sigma
        self.rotate_prob = rotate_prob
        self.rotate_factor = rotate_factor
        self.rotate_sigma = rotate_sigma
        self.use_bbox_aspect_ratio = use_bbox_aspect_ratio
        self.use_unbiased_processing = use_unbiased_processing

        self.truncnorm_scale = TruncNorm(lower=-1, upper=1, mu=0, sigma=scale_sigma)
        self.truncnorm_translate = TruncNorm(lower=-1, upper=1, mu=0, sigma=translate_sigma)
        self.truncnorm_rotate = TruncNorm(lower=-1, upper=1, mu=0, sigma=rotate_sigma)

    def __call__(self, input_batch, dataset, step):
        batch_size = len(input_batch)
        input_size = dataset.input_size
        bbox_padding_ratio = dataset.bbox_padding_ratio
        flip_pairs = dataset.keypoint_info['flip_pairs_ids']
        flip_prob_rs = np.random.rand(batch_size)
        scale_prob_rs = np.random.rand(batch_size)
        translate_prob_rs = np.random.rand(batch_size)
        rotate_prob_rs = np.random.rand(batch_size)
        scale_rs = self.truncnorm_scale.rvs(size=batch_size)
        translate_rs = self.truncnorm_translate.rvs(size=(batch_size, 2))
        rotate_rs = self.truncnorm_rotate.rvs(size=batch_size)

        scale_factor_delta = 0
        scale_factor = self.scale_factor
        if scale_factor is None:
            scale_factor = 1 - 1 / bbox_padding_ratio  # e.g. 1 - 1 / 1.25 = 0.2
        if scale_factor < self._MIN_SCALE_FACTOR:
            scale_factor_delta = self._MIN_SCALE_FACTOR - scale_factor  # avoids content going out of box
            scale_factor = self._MIN_SCALE_FACTOR

        for i, input_dict in enumerate(input_batch):
            img = input_dict['img']
            orig_img = input_dict.get('orig_img')
            center = input_dict['center']
            scale = input_dict['scale']
            bbox = input_dict['bbox']
            kpts = input_dict.get('kpts')
            mask = input_dict.get('mask')

            bbox_w, bbox_h = bbox[2:4]
            content_w, content_h = cs2bbox(center, scale,
                                           padding_ratio=bbox_padding_ratio,
                                           aspect_ratio=(bbox_w / bbox_h) if self.use_bbox_aspect_ratio else None)[2:4]

            if flip_prob_rs[i] < self.flip_prob:
                img, center, kpts, mask, orig_img = flip(img, center, kpts, mask, flip_pairs, orig_img)
                input_dict['flipped'] = 1
            else:
                input_dict['flipped'] = 0

            if scale_prob_rs[i] < self.scale_prob:
                scale = scale * (1 + scale_rs[i] * scale_factor + scale_factor_delta)

            if translate_prob_rs[i] < self.translate_prob:
                if self.translate_factor is None:
                    padding = (scale * REID_BBOX_SCALE_UNIT - [content_w, content_h]) / 2
                    offset = padding * translate_rs[i]
                else:
                    offset = npf([bbox_w, bbox_h]) * self.translate_factor * translate_rs[i]
                center = center + offset

            if rotate_prob_rs[i] < self.rotate_prob:
                rotate = self.rotate_factor * rotate_rs[i]
            else:
                rotate = 0.

            img, kpts, mask, orig_img = warp(img, center, scale, rotate, input_size, kpts, mask, self.use_unbiased_processing, orig_img)

            input_dict['img'] = img
            input_dict['center'] = center
            input_dict['scale'] = scale
            input_dict['rotate'] = rotate
            input_dict['kpts'] = kpts
            input_dict['mask'] = mask
            if orig_img is not None:
                input_dict['orig_img'] = orig_img

        return input_batch
