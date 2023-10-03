import numpy as np
import cv2

from ...registry import POSE_TRANSFORMS
from ..datautils import bbox_scale, get_warp_matrix, get_affine_matrix, apply_warp_to_coord, apply_warp_to_map2d, npf
from ..dataconsts import BORDER_COLOR_VALUE, BBOX_SCALE_UNIT


@POSE_TRANSFORMS.register_class('flip')
class Flip:
    """ Left-right flip """

    def __init__(self, apply_prob=0.5):
        self.apply_prob = apply_prob

    def __call__(self, input_dict, dataset, step):
        if np.random.rand() < self.apply_prob:
            img = input_dict['img']
            kpts = input_dict['kpts']
            center = input_dict['center']
            mask = input_dict.get('mask')

            img_width = img.shape[1]
            img = np.fliplr(img)

            kpts[:, 0] = img_width - 1 - kpts[:, 0]
            flip_pairs = dataset.keypoint_info['flip_pairs_ids']
            for left, right in flip_pairs:
                temp = kpts[left].copy()
                kpts[left] = kpts[right]
                kpts[right] = temp

            center[0] = img_width - 1 - center[0]

            if mask is not None:
                mask = mask[:, ::-1]

            input_dict['img'] = img
            input_dict['kpts'] = kpts
            input_dict['center'] = center
            input_dict['mask'] = mask
            input_dict['flipped'] = 1
        else:
            input_dict['flipped'] = 0

        return input_dict


@POSE_TRANSFORMS.register_class('lazy_half_body')
class LazyHalfBody:
    def __init__(self, min_kpts=8, padding_ratio=1.25, apply_prob=0.5):
        self.min_kpts = min_kpts
        self.padding_ratio = padding_ratio
        self.apply_prob = apply_prob

    def __call__(self, input_dict, dataset, step):
        kpts = input_dict['kpts']

        if np.sum(kpts[:, 2] > 0) > self.min_kpts and np.random.rand() < self.apply_prob:
            upper_body_ids = dataset.keypoint_info['upper_body_ids']
            upper_kpts = [kpts[id] for id in upper_body_ids if kpts[id, 2] > 0]
            lower_body_ids = dataset.keypoint_info['lower_body_ids']
            lower_kpts = [kpts[id] for id in lower_body_ids if kpts[id, 2] > 0]

            selected_kpts = None
            if np.random.rand() < 0.5:
                if len(upper_kpts) > 2:
                    selected_kpts = upper_kpts
            else:
                if len(lower_kpts) > 2:
                    selected_kpts = upper_kpts

            if selected_kpts:
                selected_kpts = npf(selected_kpts)
                center = selected_kpts[:, :2].mean(axis=0)
                x0, y0 = np.amin(selected_kpts[:, :2], axis=0)
                x1, y1 = np.amax(selected_kpts[:, :2], axis=0)
                w, h = x1 - x0, y1 - y0
                scale = npf(bbox_scale([x0, y0, w, h], dataset.input_aspect_ratio, self.padding_ratio))

                input_dict['center'] = center
                input_dict['scale'] = scale

        return input_dict


@POSE_TRANSFORMS.register_class('lazy_translate')
class LazyTranslate:
    def __init__(self, translate_factor=0.2, apply_prob=0.5):
        self.translate_factor = translate_factor
        self.apply_prob = apply_prob

    def __call__(self, input_dict, dataset, step):
        if np.random.rand() < self.apply_prob:
            center = input_dict['center']
            w, h = input_dict['bbox'][2:4]
            center += self.translate_factor * (np.random.rand(2) - 0.5) * 2 * [w, h]
            input_dict['center'] = center

        return input_dict


@POSE_TRANSFORMS.register_class('lazy_scale')
class LazyScale:
    def __init__(self, scale_factor=0.25, apply_prob=0.5):
        self.scale_factor = scale_factor
        self.apply_prob = apply_prob

    def __call__(self, input_dict, dataset, step):
        if np.random.rand() < self.apply_prob:
            scale = input_dict['scale']
            sf = self.scale_factor
            scale = scale * ((np.random.rand() - 0.5) * 2 * sf + 1)
            input_dict['scale'] = scale

        return input_dict


@POSE_TRANSFORMS.register_class('lazy_rotate')
class LazyRotate:
    def __init__(self, rotate_factor=30, apply_prob=0.5):
        self.rotate_factor = rotate_factor
        self.apply_prob = apply_prob

    def __call__(self, input_dict, dataset, step):
        rotate = 0.
        if np.random.rand() < self.apply_prob:
            rf = self.rotate_factor
            rotate = (np.random.randn() - 0.5) * 2 * rf
        input_dict['rotate'] = rotate
        return input_dict


@POSE_TRANSFORMS.register_class('warp')
class Warp:
    def __init__(self, use_unbiased_processing=False, return_inv=True):
        self.use_unbiased_processing = use_unbiased_processing
        self.return_inv = return_inv

    def __call__(self, input_dict, dataset, step):
        img = input_dict['img']
        center = input_dict['center']
        scale = input_dict['scale']
        rotate = input_dict.get('rotate', 0)
        kpts = input_dict.get('kpts')
        mask = input_dict.get('mask')
        dst_size = dataset.input_size

        if self.use_unbiased_processing:  # revert preds by setting dark_udp = True
            ret = get_warp_matrix(center, scale, dst_size, rotate=rotate, scale_unit=BBOX_SCALE_UNIT, return_inv=self.return_inv)
            mat, mat_inv = ret if self.return_inv else (ret, None)
        else:
            ret = get_affine_matrix(center, scale, dst_size, rotate=rotate, scale_unit=BBOX_SCALE_UNIT, return_inv=self.return_inv)
            mat, mat_inv = ret if self.return_inv else (ret, None)

        img = apply_warp_to_map2d(img, mat, dst_size, flags=cv2.INTER_LINEAR, border_value=BORDER_COLOR_VALUE)

        if mask is not None:
            mask = apply_warp_to_map2d(mask, mat, dst_size, flags=cv2.INTER_AREA, border_value=0)

        if kpts is not None:
            kpts[:, :2] = apply_warp_to_coord(kpts[:, :2], mat)
            kpts[kpts[:, 2] == 0] = 0

        input_dict['img'] = img
        input_dict['mask'] = mask

        input_dict['kpts'] = kpts
        if mat_inv is not None:
            input_dict['mat_inv'] = mat_inv
            input_dict['mat'] = mat
        return input_dict


@POSE_TRANSFORMS.register_class('warp_many')
class WarpMany:
    def __init__(self, use_unbiased_processing=False, return_inv=True):
        self.use_unbiased_processing = use_unbiased_processing
        self.return_inv = return_inv

    def __call__(self, input_dict, dataset, step):
        img = input_dict['img']
        center = input_dict['center']  # N x 2
        scale = input_dict['scale']    # N x 2
        assert center.ndim == 2 and center.shape == scale.shape
        n_warp = center.shape[0]
        rotate = input_dict.get('rotate', npf([0] * n_warp))  # N
        assert len(rotate) == n_warp
        kpts = input_dict.get('kpts')
        mask = input_dict.get('mask')
        dst_size = dataset.input_size

        if self.use_unbiased_processing:
            rets = [get_warp_matrix(c, s, dst_size, rotate=r, scale_unit=BBOX_SCALE_UNIT, return_inv=self.return_inv)
                    for c, s, r in zip(center, scale, rotate)]
            mats, mat_invs = zip(*rets) if self.return_inv else (rets, None)
        else:
            rets = [get_affine_matrix(c, s, dst_size, rotate=r, scale_unit=BBOX_SCALE_UNIT, return_inv=self.return_inv)
                    for c, s, r in zip(center, scale, rotate)]
            mats, mat_invs = zip(*rets) if self.return_inv else (rets, None)

        if img.ndim == 4:
            assert len(img) == len(mats)
            img_list = [apply_warp_to_map2d(img[i], mat, dst_size, flags=cv2.INTER_LINEAR, border_value=BORDER_COLOR_VALUE)
                        for i, mat in enumerate(mats)]
        else:
            img_list = [apply_warp_to_map2d(img, mat, dst_size, flags=cv2.INTER_LINEAR, border_value=BORDER_COLOR_VALUE)
                        for mat in mats]
        img = np.stack(img_list, 0)  # N x dst_h x dst_w x 3

        if mask is not None:
            mask_list = []
            for i, mat in enumerate(mats):
                if mask.ndim == 3:
                    new_mask = apply_warp_to_map2d(mask[i], mat, dst_size, flags=cv2.INTER_AREA, border_value=0)
                else:
                    new_mask = apply_warp_to_map2d(mask, mat, dst_size, flags=cv2.INTER_AREA, border_value=0)
                mask_list.append(new_mask)
            mask = np.stack(mask_list, 0)  # N x dst_h x dst_w

        if kpts is not None:
            kpts_list = []
            for i, mat in enumerate(mats):
                if kpts.ndim == 3:
                    new_kpts = kpts[i].copy()
                else:
                    new_kpts = kpts.copy()
                new_kpts[:, :2] = apply_warp_to_coord(new_kpts[:, :2], mat)
                new_kpts[new_kpts[:, 2] == 0] = 0
                kpts_list.append(new_kpts)
            kpts = np.stack(kpts_list, 0)  # N x n_kpts x 3

        input_dict['img'] = img  # N x dst_h x dst_w x 3
        input_dict['mask'] = mask  # N x dst_h x dst_w
        input_dict['kpts'] = kpts  # N x n_kpts x 3
        input_dict['rotate'] = rotate
        if self.return_inv:
            input_dict['mat_inv'] = np.stack(mat_invs, 0)  # N x 2 x 3
            input_dict['mat'] = np.stack(mats, 0)  # N x 2 x 3
        return input_dict
