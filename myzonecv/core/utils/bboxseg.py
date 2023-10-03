import math

import numpy as np
import torch


def xywh2xyxy(b):
    b2 = b.clone() if isinstance(b, torch.Tensor) else np.copy(b)
    b2[:, 2] = b[:, 0] + b[:, 2]
    b2[:, 3] = b[:, 1] + b[:, 3]
    return b2


def cxywh2xyxy(b):
    b2 = b.clone() if isinstance(b, torch.Tensor) else np.copy(b)
    b2[:, 0] = b[:, 0] - b[:, 2] / 2
    b2[:, 1] = b[:, 1] - b[:, 3] / 2
    b2[:, 2] = b[:, 0] + b[:, 2] / 2
    b2[:, 3] = b[:, 1] + b[:, 3] / 2
    return b2


def xyxy2xywh(b):
    b2 = b.clone() if isinstance(b, torch.Tensor) else np.copy(b)
    b2[:, 2] = b[:, 2] - b[:, 0]
    b2[:, 3] = b[:, 3] - b[:, 1]
    return b2


def xyxy2cxywh(b):
    b2 = b.clone() if isinstance(b, torch.Tensor) else np.copy(b)
    b2[:, 0] = (b[:, 0] + b[:, 2]) / 2
    b2[:, 1] = (b[:, 1] + b[:, 3]) / 2
    b2[:, 2] = b[:, 2] - b[:, 0]
    b2[:, 3] = b[:, 3] - b[:, 1]
    return b2


def bbox_ioa(b1, b2, format='xyxy', eps=1e-7):
    """ b1: (list or np.array, 4 or m x 4)
        b2: (list or np.array, 4 or n x 4)
    """
    if isinstance(b1, (list, tuple)):
        b1 = np.array(b1, dtype=float)
        b1_ndim = b1.ndim
        if b1.ndim == 1:
            b1 = b1[None, :]
    else:
        b1_ndim = b1.ndim

    if isinstance(b2, (list, tuple)):
        b2 = np.array(b2, dtype=float)
        b2_ndim = b2.ndim
        if b2.ndim == 1:
            b2 = b2[None, :]
    else:
        b2_ndim = b2.ndim

    if format == 'xywh':
        b1 = xywh2xyxy(b1)
        b2 = xywh2xyxy(b2)
    elif format == 'cxywh':
        b1 = cxywh2xyxy(b1)
        b2 = cxywh2xyxy(b2)

    b1_x0, b1_y0, b1_x1, b1_y1 = b1.T
    b2_x0, b2_y0, b2_x1, b2_y1 = b2.T

    inter_w = (np.minimum(b1_x1[:, None], b2_x1) - np.maximum(b1_x0[:, None], b2_x0)).clip(0)
    inter_h = (np.minimum(b1_y1[:, None], b2_y1) - np.maximum(b1_y0[:, None], b2_y0)).clip(0)
    inter_area = inter_w * inter_h
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0) + eps
    ioa = inter_area / b2_area

    if b1_ndim == 1 and b2_ndim == 1:
        ioa = ioa[0, 0]
    elif b1_ndim == 1:
        ioa = ioa[0]
    elif b2_ndim == 1:
        ioa = ioa[:, 0]
    return ioa


def bbox_iou(b1, b2, format='xyxy', eps=1e-7, GIoU=False, DIoU=False, CIoU=False):
    """ b1: (list or np.array, 4 or m x 4)
        b2: (list or np.array, 4 or n x 4)
    """
    if isinstance(b1, (list, tuple)):
        b1 = np.array(b1, dtype=float)
        b1_ndim = b1.ndim
        if b1.ndim == 1:
            b1 = b1[None, :]
    else:
        b1_ndim = b1.ndim

    if isinstance(b2, (list, tuple)):
        b2 = np.array(b2, dtype=float)
        b2_ndim = b2.ndim
        if b2.ndim == 1:
            b2 = b2[None, :]
    else:
        b2_ndim = b2.ndim

    if format == 'xywh':
        b1 = xywh2xyxy(b1)
        b2 = xywh2xyxy(b2)
    elif format == 'cxywh':
        b1 = cxywh2xyxy(b1)
        b2 = cxywh2xyxy(b2)

    b1_x0, b1_y0, b1_x1, b1_y1 = b1.T
    b2_x0, b2_y0, b2_x1, b2_y1 = b2.T

    inter_w = (np.minimum(b1_x1[:, None], b2_x1) - np.maximum(b1_x0[:, None], b2_x0)).clip(0)
    inter_h = (np.minimum(b1_y1[:, None], b2_y1) - np.maximum(b1_y0[:, None], b2_y0)).clip(0)
    inter_area = inter_w * inter_h
    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)
    union_area = b1_area[:, None] + b2_area
    iou = inter_area / union_area

    if GIoU or DIoU or CIoU:
        convex_w = (np.maximum(b1_x1[:, None], b2_x1) - np.minimum(b1_x0[:, None], b2_x0))
        convex_h = (np.maximum(b1_y1[:, None], b2_y1) - np.minimum(b1_y0[:, None], b2_y0))
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            convex_diag_sq = convex_w**2 + convex_h**2 + eps
            center_dist_sq = ((b1_x0[:, None] + b1_x1[:, None] - b2_x0 - b2_x1) ** 2 +
                              (b1_y0[:, None] + b1_y1[:, None] - b2_y0 - b2_y1) ** 2) / 4
            if DIoU:
                iou = iou - center_dist_sq / convex_diag_sq
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                b1_w, b1_h = b1_x1 - b1_x0, b1_y1 - b1_y0
                b2_w, b2_h = b2_x1 - b2_x0, b2_y1 - b2_y0
                v = (np.arctan(b2_w / (b2_h + eps)) - np.arctan(b1_w / (b1_h + eps))[:, None]) ** 2 / (math.pi / 2)**2
                alpha = v / (v - iou + (1 + eps))
                iou = iou - (center_dist_sq / convex_diag_sq + v * alpha)
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            convex_area = convex_w * convex_h + eps
            iou = iou - (convex_area - union_area) / convex_area

    if b1_ndim == 1 and b2_ndim == 1:
        iou = iou[0, 0]
    elif b1_ndim == 1:
        iou = iou[0]
    elif b2_ndim == 1:
        iou = iou[:, 0]

    return iou


def resample_segs(segs, n=1000):
    # Up-sample an (n,2) segment
    for i, seg in enumerate(segs):
        x = np.linspace(0, len(seg) - 1, n)
        xp = np.arange(len(seg))
        seg = np.unique(seg, axis=0)
        seg = np.stack([np.interp(x, xp, seg[:, 0]), np.interp(x, xp, seg[:, 1])], axis=1)
        seg = np.unique(seg, axis=0)
        segs[i] = seg
    return segs
