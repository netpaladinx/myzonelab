import numpy as np

from ...registry import DETECT_TRANSFORMS
from ..datautils import xyxy2cxywh, npf
from ..dataconsts import CAT_ID_FIGHTER_LEFT, CAT_ID_FIGHTER_RIGHT, CAT_ID_FIGHTER_NEAR, CAT_ID_FIGHTER_FAR, CAT_ID_FIGHTER_UP, CAT_ID_FIGHTER_DOWN


@DETECT_TRANSFORMS.register_class('generate_sparse_anchored_boxcls')
class GenerateSparseAnchoredBoxCls:
    def __init__(self, anchor_thr=3):
        self.anchor_thr = anchor_thr

    def __call__(self, input_dict, dataset, step):
        input_xyxy = input_dict['xyxy']
        input_cls = input_dict['cls']

        input_size = dataset.input_size
        strides = dataset.strides
        n_layers = dataset.num_anchor_layers
        n_anchors = dataset.num_anchors

        # one center anchored to 4 grid cells
        bias = 0.5
        offsets = npf([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]]) * bias  # center, up, down, left, right

        n_xyxy = len(input_xyxy)
        target_cxywh, target_cij, target_cls, target_anc_idx, target_anc, target_cnt = [], [], [], [], [], []

        # img_size: 640x640, strides: [8, 16, 32] => layer outs: [80x80, 40x40, 20x20]
        for i in range(n_layers):
            anchors = npf(dataset.anchors[i]).reshape(n_anchors, 2)  # n_anchors x 2
            anchors /= strides[i]
            out_w, out_h = int(input_size[0] / strides[i]), int(input_size[1] / strides[i])

            # rescale x0,y0,x1,y1 and convert to cx,cy,w,h
            cxywh = xyxy2cxywh(input_xyxy / strides[i])  # n_xyxy x 4
            cxywh = np.repeat(cxywh[None], n_anchors, axis=0)  # n_anchors x n_xyxy x 4
            cls = np.repeat(input_cls[None], n_anchors, axis=0)  # n_anchors x n_xyxy
            anc_i = np.repeat([[ai] for ai in range(n_anchors)], n_xyxy, axis=1)  # n_anchors x n_xyxy

            if n_xyxy > 0:
                # filter by wh ratio
                wh_ratio = cxywh[:, :, 2:4] / anchors[:, None]  # n_anchors x n_xyxy x 2, n_anchors x 1 x 2 => n_anchors x n_xyxy x 2
                filtered = np.maximum(wh_ratio, 1 / wh_ratio).max(axis=2) < self.anchor_thr  # n_anchors x n_xyxy
                cxywh = cxywh[filtered]  # n_anchors x n_xyxy x 4 => n_filter x 4
                cls = cls[filtered]  # n_filter
                anc_i = anc_i[filtered]  # n_filter

                # filter by up/down/left/right
                cxy = cxywh[:, :2]  # n_filter x 2
                left, up = ((cxy % 1. < bias) & (cxy > 1.)).T
                right, down = ((cxy % 1. > 1 - bias) & (cxy < npf([out_w, out_h]) - 1)).T
                filtered = np.stack((np.ones_like(up), up, down, left, right), axis=0)  # 5 x n_filter
                cxy_offsets = np.repeat(offsets[:, None], filtered.shape[1], axis=1)[filtered]  # 5 x 2 => 5 x n_filter x 2 => n_filter2 x 2
                cxywh = np.repeat(cxywh[None], 5, axis=0)[filtered]  # 5 x n_filter x 4 => n_filter2 x 4
                cls = np.repeat(cls[None], 5, axis=0)[filtered]      # 5 x n_filter     => n_filter2
                anc_i = np.repeat(anc_i[None], 5, axis=0)[filtered]  # 5 x n_filter     => n_filter2

                cij = (cxywh[:, :2] + cxy_offsets).astype(int)  # n_filter2 x 2
                cij[:, 0] = cij[:, 0].clip(0, out_w - 1)
                cij[:, 1] = cij[:, 1].clip(0, out_h - 1)
                cxywh[:, :2] = cxywh[:, :2] - cij

                target_cxywh.append(cxywh)  # list(n_filter2 x 4), cx,cy in (-0.5, 1.5), w,h in (1/thr, thr)*anchor
                target_cij.append(cij)  # list(n_filter2 x 2)
                target_cls.append(cls)  # list(n_filter2)
                target_anc_idx.append(anc_i)  # list(n_filter2)
                target_anc.append(anchors[anc_i])  # list(n_filter2 x 2)
                target_cnt.append(len(cxywh))  # list(int)
            else:
                target_cxywh.append(np.empty((0, 4)))
                target_cij.append(np.empty((0, 2)).astype(int))
                target_cls.append(np.empty((0,)).astype(int))
                target_anc_idx.append(np.empty((0,)).astype(int))
                target_anc.append(np.empty((0, 2)))
                target_cnt.append(0)

        input_dict['target_cxywh'] = target_cxywh
        input_dict['target_cij'] = target_cij
        input_dict['target_cls'] = target_cls
        input_dict['target_anc_idx'] = target_anc_idx
        input_dict['target_anc'] = target_anc
        input_dict['target_cnt'] = target_cnt
        return input_dict


@DETECT_TRANSFORMS.register_class('generate_myzone_sparse_anchored_boxcls')
class GenerateMyZoneSparseAnchoredBoxCls:
    duplicates = 2
    low_range_cat_ids_if_conflict = (CAT_ID_FIGHTER_LEFT, CAT_ID_FIGHTER_NEAR, CAT_ID_FIGHTER_UP)
    high_range_cat_ids_if_conflict = (CAT_ID_FIGHTER_RIGHT, CAT_ID_FIGHTER_FAR, CAT_ID_FIGHTER_DOWN)

    def __init__(self, anchor_thr=2.9):
        self.anchor_thr = anchor_thr

    def __call__(self, input_dict, dataset, step):
        input_xyxy = input_dict['xyxy']  # numpy: n_objs x 4
        input_cls = input_dict['cls']    # numpy: n_objs

        input_size = dataset.input_size                    # 640 x 640
        strides = dataset.strides                          # [8, 16, 32]
        n_layers = dataset.num_anchor_layers               # 3
        n_anchors = dataset.num_anchors * self.duplicates  # 6

        # one center anchored to 4 grid cells
        bias = 0.5
        offsets = npf([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]]) * bias  # 5 x 2, center, up, down, left, right

        n_xyxy = len(input_xyxy)  # n_objs
        target_cxywh, target_cij, target_cls, target_anc_idx, target_anc, target_cnt = [], [], [], [], [], []

        # img_size: 640x640, strides: [8, 16, 32] => layer outs: [80x80, 40x40, 20x20]
        for i in range(n_layers):
            # prepare and rescale anchors
            anchors = np.tile(npf(dataset.anchors[i]), self.duplicates).reshape(n_anchors, 2)  # n_anchors x 2
            anchors /= strides[i]
            out_w, out_h = int(input_size[0] / strides[i]), int(input_size[1] / strides[i])

            # rescale x0,y0,x1,y1 and convert to cx,cy,w,h
            cxywh = xyxy2cxywh(input_xyxy / strides[i])                           # n_xyxy x 4
            cxywh = np.repeat(cxywh[None], n_anchors, axis=0)                     # n_anchors x n_xyxy x 4
            cls = np.repeat(input_cls[None], n_anchors, axis=0)                   # n_anchors x n_xyxy
            anc_i = np.repeat([[ai] for ai in range(n_anchors)], n_xyxy, axis=1)  # n_anchors x n_xyxy

            if n_xyxy > 0:
                # filter by wh ratio
                wh_ratio = cxywh[:, :, 2:4] / anchors[:, None]  # n_anchors x n_xyxy x 2, n_anchors x 1 x 2 => n_anchors x n_xyxy x 2
                filtered = np.maximum(wh_ratio, 1 / wh_ratio).max(axis=2) < self.anchor_thr  # n_anchors x n_xyxy
                cxywh = cxywh[filtered]  # n_anchors x n_xyxy x 4 => n_filter x 4
                cls = cls[filtered]      # n_anchors x n_xyxy     => n_filter
                anc_i = anc_i[filtered]  # n_anchors x n_xyxy     => n_filter

                # filter by up/down/left/right
                cxy = cxywh[:, :2]  # n_filter x 2
                left, up = ((cxy % 1. < bias) & (cxy > 1.)).T
                right, down = ((cxy % 1. > 1 - bias) & (cxy < npf([out_w, out_h]) - 1)).T
                filtered = np.stack((np.ones_like(up), up, down, left, right), axis=0)  # 5 x n_filter
                cxy_offsets = np.repeat(offsets[:, None], filtered.shape[1], axis=1)[filtered]  # 5 x 2 => 5 x n_filter x 2 => n_filter2 x 2
                cxywh = np.repeat(cxywh[None], 5, axis=0)[filtered]  # 5 x n_filter x 4 => n_filter2 x 4
                cls = np.repeat(cls[None], 5, axis=0)[filtered]      # 5 x n_filter     => n_filter2
                anc_i = np.repeat(anc_i[None], 5, axis=0)[filtered]  # 5 x n_filter     => n_filter2

                cij = (cxywh[:, :2] + cxy_offsets).astype(int)  # n_filter2 x 2
                cij[:, 0] = cij[:, 0].clip(0, out_w - 1)
                cij[:, 1] = cij[:, 1].clip(0, out_h - 1)
                cxywh[:, :2] = cxywh[:, :2] - cij

                filtered_idx = self._deduplicate(anc_i, cij, cls, n_anchors)  # n_filter3
                target_cxywh.append(cxywh[filtered_idx])  # list(n_filter3 x 4), cx,cy in [-0.5, 1.5], w,h in (1/thr, thr)*anchor
                target_cij.append(cij[filtered_idx])  # list(n_filter3 x 2)
                target_cls.append(cls[filtered_idx])  # list(n_filter3)
                target_anc_idx.append(anc_i[filtered_idx])  # list(n_filter3)
                target_anc.append(anchors[anc_i[filtered_idx]])  # list(n_filter3 x 2)
                target_cnt.append(len(cxywh[filtered_idx]))  # list(int)
            else:
                target_cxywh.append(np.empty((0, 4)))
                target_cij.append(np.empty((0, 2)).astype(int))
                target_cls.append(np.empty((0,)).astype(int))
                target_anc_idx.append(np.empty((0,)).astype(int))
                target_anc.append(np.empty((0, 2)))
                target_cnt.append(0)

        input_dict['target_cxywh'] = target_cxywh
        input_dict['target_cij'] = target_cij
        input_dict['target_cls'] = target_cls
        input_dict['target_anc_idx'] = target_anc_idx
        input_dict['target_anc'] = target_anc
        input_dict['target_cnt'] = target_cnt
        return input_dict

    def _deduplicate(self, anc_i, cij, cls, n_anchors):
        """ anc_i: n
            cij: n x 2
            cls: n
            n_anchors: 6
        """
        anc_cij_cls = np.concatenate((anc_i[:, None], cij, cls[:, None]), axis=1)  # n x 4, (anc_i, ci, cj, cls)
        anc_cij_cls, filtered_idx = np.unique(anc_cij_cls, axis=0, return_index=True)
        _, filtered_idx2 = np.unique(anc_cij_cls[:, :3], axis=0, return_index=True)
        mask = np.ones_like(filtered_idx, dtype=bool)
        dup_range = np.stack((filtered_idx2[:-1], filtered_idx2[1:]), axis=1)
        dup_range = dup_range[dup_range[:, 1] - dup_range[:, 0] > 1]
        for beg, end in dup_range:
            assert end - beg == 2
            for i in range(beg, end):
                if ((anc_cij_cls[i, 0] < n_anchors / 2 and anc_cij_cls[i, 3] in self.high_range_cat_ids_if_conflict)
                        or (anc_cij_cls[i, 0] >= n_anchors / 2 and anc_cij_cls[i, 3] in self.low_range_cat_ids_if_conflict)):
                    mask[i] = False
        filtered_idx = filtered_idx[mask]
        return filtered_idx
