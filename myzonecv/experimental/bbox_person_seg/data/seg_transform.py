import numpy as np

from myzonecv.core.data.datautils import mask2mask
from ..registry import SEG_TRANSFORMS


@SEG_TRANSFORMS.register_class('generate_target_mask')
class GenereateBBoxSegMask:
    def __init__(self, use_hard_mask=True, hard_thr=0.5):
        self.use_hard_mask = use_hard_mask
        self.hard_thr = hard_thr

    def __call__(self, input_dict, dataset, step):
        seg_mask = input_dict['seg_mask']
        output_size = dataset.output_size
        target_mask = mask2mask(seg_mask, output_size)

        if self.use_hard_mask:
            target_mask = (target_mask > self.hard_thr).astype(target_mask.dtype)
        else:
            target_mask = np.clip(target_mask, 0.0, 1.0)
        target_mask = target_mask[None, ...]  # 1 x h x w

        input_dict['target_mask'] = target_mask
        return input_dict


@SEG_TRANSFORMS.register_class('points2mask')
class Points2Mask:
    def __init__(self, sigma_ratio=None, sigma=2):
        self.sigma_ratio = sigma_ratio
        self.sigma = sigma

    def __call__(self, input_dict, dataset, step):
        points = input_dict['points']
        img = input_dict['img']
        h, w = img.shape[:2]

        if self.sigma_ratio is not None:
            sigma = self.sigma_ratio * min(h, w)
        else:
            sigma = self.sigma

        points_mask = []
        for pnt in points:
            mu_x, mu_y = pnt[0], pnt[1]
            x = np.arange(w)
            y = np.arange(h)[:, None]
            points_mask.append(np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2)))
        points_mask = sum(points_mask) / len(points)

        input_dict['points_mask'] = points_mask
        return input_dict
