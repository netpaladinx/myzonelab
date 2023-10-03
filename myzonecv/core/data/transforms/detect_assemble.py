import random

import numpy as np

from ...registry import DETECT_TRANSFORMS
from ..dataconsts import BORDER_COLOR_VALUE


@DETECT_TRANSFORMS.register_class('mosaic')
class Mosaic:
    def __init__(self, apply_prob=0.5, border_value=BORDER_COLOR_VALUE):
        self.apply_prob = apply_prob
        self.border_value = border_value

    def __call__(self, input_dict, dataset, step):
        if np.random.rand() < self.apply_prob:
            img = input_dict['img']
            img_h, img_w, img_c = img.shape
            border_top, border_bottom, border_left, border_right = input_dict['_revert_params'][1]
            inner_h = img_h - border_top - border_bottom
            inner_w = img_w - border_left - border_right

            mosaic_h, mosaic_w = img_h * 2, img_w * 2   # e.g. 640 x 640 => 1280 x 1280
            mosaic_border = [-img_h // 2, -img_h // 2, -img_w // 2, -img_w // 2]  # e.g. 1280 x 1280 => 640 x 640
            mosaic_cx = int(random.uniform(img_w - inner_w // 2, img_w + inner_w // 2))
            mosaic_cy = int(random.uniform(img_h - inner_h // 2, img_h + inner_h // 2))
            mosaic_img = np.full((mosaic_h, mosaic_w, img_c), self.border_value[0], dtype=np.uint8)

            input_dicts = self.get_random_inputs(dataset, n=3)
            input_dicts.append(input_dict)
            random.shuffle(input_dicts)
            has_segs = all('segs' in dct for dct in input_dicts)

            mosaic_xyxy = np.concatenate([input_dicts[i]['xyxy'] for i in range(len(input_dicts))], axis=0)
            mosaic_cls = np.concatenate([input_dicts[i]['cls'] for i in range(len(input_dicts))], axis=0)
            if has_segs:
                mosaic_segs = []
                for i in range(len(input_dicts)):
                    mosaic_segs += input_dicts[i]['segs']

            start, end = 0, 0
            for i in range(4):
                img = input_dicts[i]['img']
                border_top, border_bottom, border_left, border_right = input_dicts[i]['_revert_params'][1]
                img = img[border_top:img.shape[0] - border_bottom, border_left:img.shape[1] - border_right]
                img_h, img_w = img.shape[:2]
                n = len(input_dicts[i]['xyxy'])

                if i == 0:  # top left
                    dst_x0, dst_y0, dst_x1, dst_y1 = max(mosaic_cx - img_w, 0), max(mosaic_cy - img_h, 0), mosaic_cx, mosaic_cy
                    src_x0, src_y0, src_x1, src_y1 = img_w - (dst_x1 - dst_x0), img_h - (dst_y1 - dst_y0), img_w, img_h

                elif i == 1:  # top right
                    dst_x0, dst_y0, dst_x1, dst_y1 = mosaic_cx, max(mosaic_cy - img_h, 0), min(mosaic_cx + img_w, mosaic_w), mosaic_cy
                    src_x0, src_y0, src_x1, src_y1 = 0, img_h - (dst_y1 - dst_y0), dst_x1 - dst_x0, img_h

                elif i == 2:  # bottom left
                    dst_x0, dst_y0, dst_x1, dst_y1 = max(mosaic_cx - img_w, 0), mosaic_cy, mosaic_cx, min(mosaic_cy + img_h, mosaic_h)
                    src_x0, src_y0, src_x1, src_y1 = img_w - (dst_x1 - dst_x0), 0, img_w, dst_y1 - dst_y0

                elif i == 3:  # bottom right
                    dst_x0, dst_y0, dst_x1, dst_y1 = mosaic_cx, mosaic_cy, min(mosaic_cx + img_w, mosaic_w), min(mosaic_cy + img_h, mosaic_h)
                    src_x0, src_y0, src_x1, src_y1 = 0, 0, dst_x1 - dst_x0, dst_y1 - dst_y0

                mosaic_img[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
                start, end = end, end + n

                mosaic_xyxy[start:end, [0, 2]] += dst_x0 - src_x0 - border_left
                mosaic_xyxy[start:end, [1, 3]] += dst_y0 - src_y0 - border_top
                if has_segs:
                    for j in range(start, end):
                        mosaic_segs[j][:, 0] += dst_x0 - src_x0 - border_left
                        mosaic_segs[j][:, 1] += dst_y0 - src_y0 - border_top

            mosaic_xyxy[:, [0, 2]] = np.clip(mosaic_xyxy[:, [0, 2]], 0, mosaic_w)
            mosaic_xyxy[:, [1, 3]] = np.clip(mosaic_xyxy[:, [1, 3]], 0, mosaic_h)
            if has_segs:
                for i in range(len(mosaic_segs)):
                    mosaic_segs[i][:, 0] = np.clip(mosaic_segs[i][:, 0], 0, mosaic_w)
                    mosaic_segs[i][:, 1] = np.clip(mosaic_segs[i][:, 1], 0, mosaic_h)
                    mosaic_segs[i] = np.unique(mosaic_segs[i], axis=0)

            input_dict['img'] = mosaic_img
            input_dict['xyxy'] = mosaic_xyxy
            input_dict['cls'] = mosaic_cls
            input_dict['border'] = mosaic_border  # used in perspective
            if has_segs:
                input_dict['segs'] = mosaic_segs

        return input_dict

    @staticmethod
    def get_random_inputs(dataset, n=1):
        input_dicts = []
        indices = random.choices(range(len(dataset)), k=n)
        for idx in indices:
            input_dicts.append(dataset.get_unprocessed_item(idx))
        return input_dicts


@DETECT_TRANSFORMS.register_class('mixup')
class Mixup:
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    def __init__(self, alpha=32, beta=32):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, input_dict, dataset, step):
        if 'input_dict2' in input_dict:
            input_dicts = input_dict.pop('input_dict2')
            input_dicts.insert(0, input_dict)

            input_dict2 = input_dicts.pop()
            img2 = input_dict2['img']
            xyxy2 = input_dict2['xyxy']
            cls2 = input_dict2['cls']
            segs2 = input_dict2.get('segs')

            while len(input_dicts) > 0:
                input_dict1 = input_dicts.pop()

                img1 = input_dict1['img']
                xyxy1 = input_dict1['xyxy']
                cls1 = input_dict1['cls']
                segs1 = input_dict1.get('segs')

                mixup_ratio = np.random.beta(self.alpha, self.beta)
                img2 = (img1 * mixup_ratio + img2 * (1 - mixup_ratio)).astype(np.uint8)
                xyxy2 = np.concatenate((xyxy1, xyxy2), 0)
                cls2 = np.concatenate((cls1, cls2), 0)
                if segs2 and segs1:
                    segs2 = segs1 + segs2
                else:
                    segs2 = None

            input_dict['img'] = img2
            input_dict['xyxy'] = xyxy2
            input_dict['cls'] = cls2
            if segs2 is not None:
                input_dict['segs'] = segs2

        return input_dict
