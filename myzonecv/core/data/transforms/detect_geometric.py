import math
import random

import numpy as np
import cv2

from ...registry import DETECT_TRANSFORMS
from ..datautils import bbox_ioa, resample_segs, npf, npi
from ..dataconsts import BORDER_COLOR_VALUE
from ..datasets.myzoneufc import MyZoneUFCDetect


@DETECT_TRANSFORMS.register_class('flip')
class Flip:
    """ Left-right flip """

    def __init__(self, apply_prob=0.5):
        self.apply_prob = apply_prob

    def __call__(self, input_dict, dataset, step):
        if np.random.rand() < self.apply_prob:
            img = input_dict['img']
            xyxy = input_dict['xyxy']
            segs = input_dict.get('segs')

            img = np.fliplr(img)

            img_w = img.shape[1]
            x0, y0, x1, y1 = np.split(xyxy, xyxy.shape[1], axis=1)
            xyxy = np.concatenate((img_w - x1, y0, img_w - x0, y1), axis=1)  # x0, y0, x1, y1 => img_w-x1, y0, img_w-x0, y1

            if segs is not None:
                for seg in segs:
                    seg[:, 0] = img_w - seg[:, 0]

            input_dict['img'] = img
            input_dict['xyxy'] = xyxy
            if segs is not None:
                input_dict['segs'] = segs

            if isinstance(dataset, MyZoneUFCDetect):
                cls = input_dict['cls'].tolist()
                cls = npi([dataset.flip_pair_ids.get(c, c) for c in cls])
                input_dict['cls'] = cls

        return input_dict


@DETECT_TRANSFORMS.register_class('copy_paste')
class CopyPaste:
    # Implement Copy-Paste (by flipping) augmentation https://arxiv.org/abs/2012.07177
    def __init__(self, apply_prob=0.5, apply_pct=0.5):
        self.apply_prob = apply_prob
        self.apply_pct = apply_pct

    def __call__(self, input_dict, dataset, step):
        if np.random.rand() < self.apply_prob:
            img = input_dict['img']
            xyxy = input_dict['xyxy']
            cls = input_dict['cls']
            segs = input_dict.get('segs')

            if segs is not None:  # depends on segs
                n_segs = len(segs)
                img_w = img.shape[1]
                new_img = np.zeros(img.shape, np.uint8)
                for i in random.sample(range(n_segs), k=round(n_segs * self.apply_pct)):
                    x0, y0, x1, y1 = xyxy[i]
                    c = cls[i]
                    seg = segs[i]
                    new_xyxy = (img_w - x1, y0, img_w - x0, y1)
                    ioa = bbox_ioa(new_xyxy, xyxy)
                    if (ioa < 0.3).all():  # allow 30% obscuration of existing bboxes
                        xyxy = np.concatenate((xyxy, [new_xyxy]), 0)
                        cls = np.concatenate((cls, [c]), 0)
                        new_seg = np.concatenate((img_w - seg[:, 0:1], seg[:, 1:2]), 1)
                        segs.append(new_seg)
                        cv2.drawContours(new_img, [seg.astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

                crop_img = cv2.bitwise_and(src1=img, src2=new_img)
                crop_img = cv2.flip(crop_img, 1)  # left-right flip
                crop_idx = crop_img > 0
                img[crop_idx] = crop_img[crop_idx]

                input_dict['img'] = img
                input_dict['xyxy'] = xyxy
                input_dict['cls'] = cls
                input_dict['segs'] = segs

        return input_dict


@DETECT_TRANSFORMS.register_class('random_warp')
class RandomWarp:
    def __init__(self, perspective_factor=0.0, rotate_factor=0.0, scale_factor=(0.2, 1.8), shear_factor=0., translate_factor=0.2,
                 width_thr=2, height_thr=2, aspect_ratio_thr=20, area_ratio_thr=0.1, eps=1e-16, use_seg_as_bbox=False,
                 border_value=BORDER_COLOR_VALUE):
        assert isinstance(scale_factor, (list, tuple)) and len(scale_factor) == 2
        self.perspective_factor = perspective_factor
        self.rotate_factor = rotate_factor
        self.scale_factor = scale_factor
        self.shear_factor = shear_factor
        self.translate_factor = translate_factor
        self.width_thr = width_thr
        self.height_thr = height_thr
        self.aspect_ratio_thr = aspect_ratio_thr
        self.area_ratio_thr = area_ratio_thr
        self.eps = eps
        self.use_seg_as_bbox = use_seg_as_bbox  # accurate when applying rotation
        self.border_value = border_value

    def __call__(self, input_dict, dataset, step):
        img = input_dict['img']
        border = input_dict.pop('border', (0, 0, 0, 0))
        xyxy = input_dict['xyxy']
        cls = input_dict['cls']
        segs = input_dict.get('segs')

        src_h, src_w = img.shape[:2]
        dst_h, dst_w = src_h + border[0] + border[1], src_w + border[2] + border[3]

        # Center
        C = np.eye(3)
        C[0, 2] = -src_w / 2  # x translation (pixels)
        C[1, 2] = -src_h / 2  # y translation (pixels)

        # Perspective
        psp_x = random.uniform(-self.perspective_factor, self.perspective_factor)
        psp_y = random.uniform(-self.perspective_factor, self.perspective_factor)
        P = np.eye(3)
        P[2, 0] = psp_x  # x perspective (divided by psp_x * x)
        P[2, 1] = psp_y  # y perspective (divided by psp_y * y)

        # Rotation and Scale
        angle = random.uniform(-self.rotate_factor, self.rotate_factor)
        scale = random.uniform(self.scale_factor[0], self.scale_factor[1])
        R = np.eye(3)
        R[:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=angle, scale=scale)

        # Shear
        shear_x = random.uniform(-self.shear_factor, self.shear_factor)
        shear_y = random.uniform(-self.shear_factor, self.shear_factor)
        S = np.eye(3)
        S[0, 1] = math.tan(shear_x * math.pi / 180)  # x shear by y
        S[1, 0] = math.tan(shear_y * math.pi / 180)  # y shear by x

        # Translation
        trans_x = random.uniform(0.5 - self.translate_factor, 0.5 + self.translate_factor)
        trans_y = random.uniform(0.5 - self.translate_factor, 0.5 + self.translate_factor)
        T = np.eye(3)
        T[0, 2] = trans_x * dst_w
        T[1, 2] = trans_y * dst_h

        M = T @ S @ R @ P @ C
        if any([b != 0 for b in border]) or (M != np.eye(3)).any():
            if self.perspective_factor > 0:
                img = cv2.warpPerspective(img, M, dsize=(dst_w, dst_h), borderValue=self.border_value)
            else:
                img = cv2.warpAffine(img, M[:2], dsize=(dst_w, dst_h), borderValue=self.border_value)

        new_xyxy = xyxy.copy()
        new_segs = []
        if self.use_seg_as_bbox and segs is not None:
            segs = resample_segs(segs)
            self.area_ratio_thr *= 0.1
            for i, seg in enumerate(segs):
                vertices = np.ones((len(seg), 3))
                vertices[:, :2] = seg
                vertices = vertices @ M.T
                if self.perspective_factor > 0:
                    vertices /= vertices[:, 2:3]
                vertices = vertices[:, :2]
                new_segs.append(vertices)
                xs_obj, ys_obj = vertices[:, 0], vertices[:, 1]
                new_xyxy[i] = npf([xs_obj.min(), ys_obj.min(), xs_obj.max(), ys_obj.max()])
        else:
            vertices = np.ones((len(new_xyxy) * 4, 3))
            vertices[:, :2] = new_xyxy[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(-1, 2)  # x0y0, x1y1, x0y1, x1y0
            vertices = vertices @ M.T
            if self.perspective_factor > 0:
                vertices /= vertices[:, 2:3]
            vertices = vertices[:, :2].reshape(-1, 8)

            xs_all = vertices[:, [0, 2, 4, 6]]
            ys_all = vertices[:, [1, 3, 5, 7]]
            new_xyxy = np.stack((xs_all.min(1), ys_all.min(1), xs_all.max(1), ys_all.max(1)), axis=1)

        # new_cx_unclipped = (new_xyxy[:, 0] + new_xyxy[:, 2]) / 2.
        # new_cy_unclipped = (new_xyxy[:, 1] + new_xyxy[:, 3]) / 2.
        new_xyxy[:, [0, 2]] = new_xyxy[:, [0, 2]].clip(0, dst_w - 1)
        new_xyxy[:, [1, 3]] = new_xyxy[:, [1, 3]].clip(0, dst_h - 1)

        w, h = (xyxy[:, 2] - xyxy[:, 0]) * scale, (xyxy[:, 3] - xyxy[:, 1]) * scale,
        new_w, new_h = new_xyxy[:, 2] - new_xyxy[:, 0], new_xyxy[:, 3] - new_xyxy[:, 1]
        aspect_ratio = np.maximum(new_w / (new_h + self.eps), new_h / (new_w + self.eps))
        # filtered = ((new_w > self.width_thr) & (new_h > self.height_thr) &
        #             (new_w * new_h / (w * h + self.eps) > self.area_ratio_thr) & (aspect_ratio < self.aspect_ratio_thr) &
        #             (new_cx_unclipped >= 0) & (new_cx_unclipped < dst_w) & (new_cy_unclipped >= 0) & (new_cy_unclipped < dst_h))
        filtered = ((new_w > self.width_thr) & (new_h > self.height_thr) &
                    (new_w * new_h / (w * h + self.eps) > self.area_ratio_thr) & (aspect_ratio < self.aspect_ratio_thr))

        new_xyxy = new_xyxy[filtered]
        cls = cls[filtered]

        input_dict['img'] = img
        input_dict['xyxy'] = new_xyxy
        input_dict['cls'] = cls
        if new_segs:
            input_dict['segs'] = new_segs

        return input_dict


@DETECT_TRANSFORMS.register_class('safe_random_warp')
class SafeRandomWarp:
    def __init__(self, scale_factor=(2 / 3, 3 / 2), translate_factor=0.4, eps=1e-16, use_seg_as_bbox=False, border_value=BORDER_COLOR_VALUE):
        assert isinstance(scale_factor, (list, tuple)) and len(scale_factor) == 2
        self.scale_factor = scale_factor
        self.translate_factor = translate_factor
        self.eps = eps
        self.use_seg_as_bbox = use_seg_as_bbox  # accurate when applying rotation
        self.border_value = border_value

    def __call__(self, input_dict, dataset, step):
        img = input_dict['img']
        border = input_dict.pop('border', (0, 0, 0, 0))
        xyxy = input_dict['xyxy']
        segs = input_dict.get('segs')

        src_h, src_w = img.shape[:2]
        dst_h, dst_w = src_h + border[0] + border[1], src_w + border[2] + border[3]
        min_x, min_y = np.min(xyxy[:, 0]), np.min(xyxy[:, 1])
        max_x, max_y = np.max(xyxy[:, 2]), np.max(xyxy[:, 3])

        # Center
        offset_x = -src_w / 2
        offset_y = -src_h / 2
        C = np.eye(3)
        C[0, 2] = offset_x  # x translation (pixels)
        C[1, 2] = offset_y  # y translation (pixels)
        min_x += offset_x
        max_x += offset_x
        min_y += offset_y
        max_y += offset_y

        # Scale
        max_w = dst_w / 2
        max_h = dst_h / 2
        max_scale = np.min([max_w / (np.abs(min_x) + self.eps), max_w / (np.abs(max_x) + self.eps),
                            max_h / (np.abs(min_y) + self.eps), max_h / 2 / (np.abs(max_y) + self.eps)])
        scale = random.uniform(self.scale_factor[0], min(self.scale_factor[1], max_scale))
        S = np.eye(3)
        S[:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=0, scale=scale)
        min_x *= scale
        max_x *= scale
        min_y *= scale
        max_y *= scale

        # Translation
        max_trans_left = min_x + dst_w / 2
        max_trans_right = dst_w / 2 - max_x
        max_trans_up = min_y + dst_h / 2
        max_trans_down = dst_h / 2 - max_y
        trans_x = random.uniform(0.5 - min(max_trans_left / dst_w, self.translate_factor),
                                 0.5 + min(max_trans_right / dst_w, self.translate_factor))
        trans_y = random.uniform(0.5 - min(max_trans_up / dst_h, self.translate_factor),
                                 0.5 + min(max_trans_down / dst_h, self.translate_factor))
        T = np.eye(3)
        T[0, 2] = trans_x * dst_w
        T[1, 2] = trans_y * dst_h

        M = T @ S  @ C
        if any([b != 0 for b in border]) or (M != np.eye(3)).any():
            img = cv2.warpAffine(img, M[:2], dsize=(dst_w, dst_h), borderValue=self.border_value)

        new_xyxy = xyxy.copy()
        new_segs = []
        if self.use_seg_as_bbox and segs is not None:
            segs = resample_segs(segs)
            for i, seg in enumerate(segs):
                vertices = np.ones((len(seg), 3))
                vertices[:, :2] = seg
                vertices = vertices @ M.T
                vertices = vertices[:, :2]
                new_segs.append(vertices)
                xs_obj, ys_obj = vertices[:, 0], vertices[:, 1]
                new_xyxy[i] = npf([xs_obj.min(), ys_obj.min(), xs_obj.max(), ys_obj.max()])
        else:
            vertices = np.ones((len(new_xyxy) * 4, 3))
            vertices[:, :2] = new_xyxy[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(-1, 2)  # x0y0, x1y1, x0y1, x1y0
            vertices = vertices @ M.T
            vertices = vertices[:, :2].reshape(-1, 8)

            xs_all = vertices[:, [0, 2, 4, 6]]
            ys_all = vertices[:, [1, 3, 5, 7]]
            new_xyxy = np.stack((xs_all.min(1), ys_all.min(1), xs_all.max(1), ys_all.max(1)), axis=1)

        input_dict['img'] = img
        input_dict['xyxy'] = new_xyxy

        return input_dict
