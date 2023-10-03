import numpy as np

from ...registry import POSE_TRANSFORMS
from ...utils import TruncNorm
from ..datautils import cs2bbox, npf, npi
from ..dataconsts import BBOX_PADDING_RATIO, BBOX_SCALE_UNIT


@POSE_TRANSFORMS.register_class('eval_stable')
class EvalStable:
    # 1 + 9 = 10 varying input boxes
    def __init__(self, bbox_padding_ratio=BBOX_PADDING_RATIO, use_bbox_aspect_ratio=False, eval_index=None):
        self.bbox_padding_ratio = bbox_padding_ratio
        self.use_bbox_aspect_ratio = use_bbox_aspect_ratio
        self.direction = [[-1, -1], [0, -1], [1, -1],
                          [-1, 0], [0, 0], [1, 0],
                          [-1, 1], [0, 1], [1, 1]]
        self.eval_multiple = 1 + len(self.direction)
        self.eval_index = eval_index  # or 5 (points to normal input box)

    def __call__(self, input_dict, dataset, step):
        dataset.eval_multiple = self.eval_multiple
        dataset.eval_index = self.eval_index

        center = input_dict['center']
        scale = input_dict['scale']
        w, h = input_dict['bbox'][2:4]
        w, h = cs2bbox(center, scale,
                       padding_ratio=self.bbox_padding_ratio,
                       aspect_ratio=(w / h) if self.use_bbox_aspect_ratio else None)[2:4]
        padding = (scale * BBOX_SCALE_UNIT - [w, h]) / 2
        center_list = [center]
        scale_list = [scale / self.bbox_padding_ratio]
        for di in self.direction:
            center_list.append(center + padding * npf(di))
            scale_list.append(scale)
        input_dict['center'] = np.stack(center_list, 0)  # 10 x 2
        input_dict['scale'] = np.stack(scale_list, 0)  # 10 x 2
        input_dict['eval_index'] = self.eval_index
        return input_dict


@POSE_TRANSFORMS.register_class('train_stable')
class TrainStable:
    def __init__(self, multiple=4, flip_prob=0.5, rotate_prob=1., rotate_factor=30,
                 bbox_padding_ratio=BBOX_PADDING_RATIO, use_bbox_aspect_ratio=False):
        self.multipe = multiple
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotate_factor = rotate_factor
        self.bbox_padding_ratio = bbox_padding_ratio
        self.use_bbox_aspect_ratio = use_bbox_aspect_ratio

        self.truncnorm = TruncNorm(lower=-1, upper=1, mu=0, sigma=1)
        self.truncnorm2 = TruncNorm(lower=-1, upper=1, mu=0, sigma=1. / 2)
        self.truncnorm3 = TruncNorm(lower=-1, upper=1, mu=0, sigma=1. / 3)

    @staticmethod
    def _flip(img, mask, kpts, center, flip_pairs):
        img = np.fliplr(img)
        img_width = img.shape[1]
        center[0] = img_width - 1 - center[0]

        if kpts is not None:
            kpts[:, 0] = img_width - 1 - kpts[:, 0]
            for left, right in flip_pairs:
                temp = kpts[left].copy()
                kpts[left] = kpts[right]
                kpts[right] = temp

        if mask is not None:
            mask = mask[:, ::-1]

        return img, mask, kpts, center

    def __call__(self, input_dict, dataset, step):
        img = input_dict['img']
        center = input_dict['center']
        scale = input_dict['scale']
        mask = input_dict.get('mask')
        kpts = input_dict.get('kpts')

        img_h, img_w = img.shape[:2]  # (np.ndarray) h,w,c
        input_w, input_h = dataset.input_size
        bbox_w, bbox_h = input_dict['bbox'][2:4]
        content_w, content_h = cs2bbox(center, scale,
                                       padding_ratio=self.bbox_padding_ratio,
                                       aspect_ratio=(bbox_w / bbox_h) if self.use_bbox_aspect_ratio else None)[2:4]
        scale_factor = 1 - 1 / self.bbox_padding_ratio  # e.g. 1 - 1 / 1.25 = 0.2

        img_list, mask_list = [], []
        kpts_list = []
        center_list, scale_list = [], []
        flipped_list = []
        rotate_list = []
        for i in range(self.multipe):
            new_img = img.copy()
            new_mask = mask.copy() if mask is not None else None
            new_kpts = kpts.copy() if kpts is not None else None

            new_scale = scale * (1 + self.truncnorm.rvs() * scale_factor)
            padding = (new_scale * BBOX_SCALE_UNIT - [content_w, content_h]) / 2
            offset = self.truncnorm3.rvs(2) * padding
            new_center = center + offset

            if np.random.rand() < self.flip_prob:
                new_img, new_mask, new_kpts, new_center = self._flip(
                    new_img, new_mask, new_kpts, new_center, dataset.keypoint_info['flip_pairs_ids'])
                flipped = 1  # flip before warp
            else:
                flipped = 0

            if np.random.rand() < self.rotate_prob:
                rotate = self.truncnorm2.rvs() * self.rotate_factor
            else:
                rotate = 0.0

            img_list.append(new_img)
            mask_list.append(new_mask)
            kpts_list.append(new_kpts)
            center_list.append(new_center)
            scale_list.append(new_scale)
            flipped_list.append(flipped)
            rotate_list.append(rotate)

        input_dict['img'] = np.stack(img_list, 0)  # multiple x h x w x c
        input_dict['mask'] = np.stack(mask_list, 0) if mask_list[0] is not None else None  # multiple x h x w
        input_dict['kpts'] = np.stack(kpts_list, 0) if kpts_list[0] is not None else None  # multiple x n_kpts x 3
        input_dict['center'] = np.stack(center_list, 0)  # multiple x 2
        input_dict['scale'] = np.stack(scale_list, 0)  # multiple x 2
        input_dict['flipped'] = np.stack(flipped_list, 0)  # multiple
        input_dict['rotate'] = np.stack(rotate_list, 0)  # multiple
        input_dict['params'] = npi([[self.multipe, img_w, img_h, input_w, input_h, BBOX_SCALE_UNIT]])  # 1 x 6
        return input_dict
