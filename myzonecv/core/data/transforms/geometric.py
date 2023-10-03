import numpy as np
import cv2

from ...registry import DATA_TRANSFORMS
from ...utils import tolist, npf
from ..datautils import cs2bbox, bbox_scale, get_warp_matrix, get_affine_matrix, apply_warp_to_coord, apply_warp_to_map2d
from ..dataconsts import BORDER_COLOR_VALUE, BBOX_SCALE_UNIT


@DATA_TRANSFORMS.register_class('flip')
class Flip:
    """ Left-right flip """

    def __init__(self, apply_prob=0.5, flip_map=None, flip_coord=None, width_dim=1, x_index=0):
        self.apply_prob = apply_prob
        self.flip_map = ('img', *tolist(flip_map, exclude='img'))
        self.flip_coord = ('center', *tolist(flip_coord, exclude='center'))
        self.width_dim = width_dim
        self.x_index = x_index

    def __call__(self, input_dict, dataset, step):
        if np.random.rand() < self.apply_prob:
            map_width = None
            for name in self.flip_map:
                map = input_dict.get(name)  # matrix: h x w x ...
                if map is None:
                    continue
                map = np.fliplr(map)
                input_dict[name] = map
                if map_width is None:
                    map_width = map.shape[self.width_dim]  # accoroding to image width

            for name in self.flip_coord:
                coord = input_dict.get(name)  # coord: ... x 2 (last dim: [x, y])
                if coord is None:
                    continue
                assert map_width is not None
                coord[..., self.x_index] = map_width - 1 - coord[..., self.x_index]
                input_dict[name] = coord

            input_dict['flipped'] = 1
        else:
            input_dict['flipped'] = 0

        return input_dict


@DATA_TRANSFORMS.register_class('lazy_vertical_half')
class LazyVerticalHalf:  # img unaffected here
    def __init__(self, apply_prob=0.3, std_factor=0.3, min_factor=0.3):
        self.apply_prob = apply_prob
        self.std_factor = std_factor
        self.min_factor = min_factor

    def __call__(self, input_dict, dataset, step):
        if np.random.rand() < self.apply_prob:
            center = input_dict['center']
            scale = input_dict['scale']
            padding_ratio = dataset.bbox_padding_ratio
            x0, y0, w, h = cs2bbox(center, scale, padding_ratio)

            cx = center[0]
            std = h * 0.5 * self.std_factor
            if np.random.rand() < 0.5:
                cy = y0 + h * 0.25
                cy = np.random.normal(cy, std)
                new_h = (cy - y0) * 2
            else:
                cy = y0 + h * 0.75
                cy = np.random.normal(cy, std)
                new_h = (y0 + h - cy) * 2

            if new_h > h * self.min_factor:
                y0 = cy - new_h * 0.5
                center = npf([cx, cy])
                scale = npf(bbox_scale([x0, y0, w, new_h], dataset.input_aspect_ratio, padding_ratio))
                input_dict['center'] = center
                input_dict['scale'] = scale

        return input_dict


@DATA_TRANSFORMS.register_class('lazy_horizontal_half')
class LazyHorizontalHalf:  # img unaffected here
    def __init__(self, apply_prob=0.3, std_factor=0.3, min_factor=0.3):
        self.apply_prob = apply_prob
        self.std_factor = std_factor
        self.min_factor = min_factor

    def __call__(self, input_dict, dataset, step):
        if np.random.rand() < self.apply_prob:
            center = input_dict['center']
            scale = input_dict['scale']
            padding_ratio = dataset.bbox_padding_ratio
            x0, y0, w, h = cs2bbox(center, scale, padding_ratio)

            cy = center[1]
            std = w * 0.5 * self.std_factor
            if np.random.rand() < 0.5:
                cx = x0 + w * 0.25
                cx = np.random.normal(cx, std)
                new_w = (cx - x0) * 2
            else:
                cx = x0 + w * 0.75
                cx = np.random.normal(cx, std)
                new_w = (x0 + w - cx) * 2

            if new_w > w * self.min_factor:
                x0 = cx - new_w * 0.5
                center = npf([cx, cy])
                scale = npf(bbox_scale([x0, y0, new_w, h], dataset.input_aspect_ratio, padding_ratio))
                input_dict['center'] = center
                input_dict['scale'] = scale

        return input_dict


@DATA_TRANSFORMS.register_class('lazy_translate')
class LazyTranslate:  # img unaffected here
    def __init__(self, apply_prob=0.5, translate_factor=0.2):
        self.apply_prob = apply_prob
        self.translate_factor = translate_factor

    def __call__(self, input_dict, dataset, step):
        if np.random.rand() < self.apply_prob:
            center = input_dict['center']
            scale = input_dict['scale']
            padding_ratio = dataset.bbox_padding_ratio
            x0, y0, w, h = cs2bbox(center, scale, padding_ratio)
            center += self.translate_factor * (np.random.rand(2) - 0.5) * 2 * [w, h]
            input_dict['center'] = center

        return input_dict


@DATA_TRANSFORMS.register_class('lazy_scale')
class LazyScale:  # img unaffected here
    def __init__(self, apply_prob=0.5, scale_factor=0.5):
        self.apply_prob = apply_prob
        self.scale_factor = scale_factor

    def __call__(self, input_dict, dataset, step):
        if np.random.rand() < self.apply_prob:
            scale = input_dict['scale']
            sf = self.scale_factor
            scale = scale * ((np.random.rand() - 0.5) * 2 * sf + 1)
            input_dict['scale'] = scale

        return input_dict


@DATA_TRANSFORMS.register_class('lazy_rotate')
class LazyRotate:  # img unaffected here
    def __init__(self, apply_prob=0.6, rotate_factor=40):
        self.apply_prob = apply_prob
        self.rotate_factor = rotate_factor

    def __call__(self, input_dict, dataset, step):
        rotate = 0.
        if np.random.rand() < self.apply_prob:
            rf = self.rotate_factor
            rotate = np.clip(rf * np.random.randn(), -rf * 2, rf * 2)  # degree
        input_dict['rotate'] = rotate
        return input_dict


@DATA_TRANSFORMS.register_class('warp')
class Warp:
    def __init__(self, use_unbiased_processing=False, warp_map2d=None, warp_coord=None):
        self.use_unbiased_processing = use_unbiased_processing
        self.warp_map2d = ('img', *tolist(warp_map2d, exclude='img'))
        self.warp_coord = tolist(warp_coord)

    def __call__(self, input_dict, dataset, step):
        center = input_dict['center']
        scale = input_dict['scale']
        rotate = input_dict.get('rotate', 0)
        dst_size = dataset.input_size

        if self.use_unbiased_processing:
            mat = get_warp_matrix(center, scale, dst_size, rotate=rotate, scale_unit=BBOX_SCALE_UNIT)
        else:
            mat = get_affine_matrix(center, scale, dst_size, rotate=rotate, scale_unit=BBOX_SCALE_UNIT)

        for name in self.warp_map2d:
            map2d = input_dict.get(name)
            if map2d is None:
                continue
            map2d = apply_warp_to_map2d(map2d, mat, dst_size,
                                        flags=cv2.INTER_LINEAR if name == 'img' else cv2.INTER_AREA,
                                        border_value=BORDER_COLOR_VALUE if name == 'img' else 0)
            input_dict[name] = map2d

        for name in self.warp_coord:
            coord = input_dict.get(name)
            if coord is None:
                continue
            coord[..., :2] = apply_warp_to_coord(coord[..., :2], mat)
            input_dict[name] = coord

        return input_dict
