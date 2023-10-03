import torch
import torch.nn.functional as F

from ...registry import DETECT_POSTPROCESSOR
from ...consts import IMAGENET_MEAN
from .base_process import BaseProcess


@DETECT_POSTPROCESSOR.register_class('process')
class DetectProcess(BaseProcess):
    @staticmethod
    def flip_scale_img(img, flip=False, scale=1.):
        """ img: bs x c x img_h x img_w
        """
        if flip:
            img = img.flip(3)  # left-right
        if scale != 1.:
            img_h, img_w = img.shape[2:]
            new_h, new_w = (int(img_h * scale), int(img_w * scale))
            img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
            img = F.pad(img, (0, img_w - new_w, 0, img_h - new_h), value=IMAGENET_MEAN)
        return img

    @staticmethod
    def scale_fip_preds_back(preds, img_w, flip=False, scale=1.):
        """ preds: list(bs x (n_anchors*out_h*out_w) x out_dims) 
        """
        rev_preds = []
        for pred in preds:
            cx, cy, wh = pred[..., 0:1], pred[..., 1:2], pred[..., 2:4]
            cx, cy, wh = cx / scale, cy / scale, wh / scale
            if flip:
                cx = img_w - cx
            rev_pred = torch.cat((cx, cy, wh, pred[..., 4:]), -1)
            rev_preds.append(rev_pred)
        return rev_preds

    @staticmethod
    def parse_cxywh(bbox):
        cx, cy, w, h = bbox.T
        x0 = cx - w / 2
        y0 = cy - h / 2
        x1 = cx + w / 2
        y1 = cy + h / 2
        return cx, cy, w, h, x0, y0, x1, y1

    @staticmethod
    def parse_xyxy(bbox):
        x0, y0, x1, y1 = bbox.T
        w = x1 - x0
        h = y1 - y0
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        return cx, cy, w, h, x0, y0, x1, y1

    @staticmethod
    def parse_xywh(bbox):
        x0, y0, w, h = bbox.T
        x1 = x0 + w
        y1 = y0 + h
        cx = x0 + w / 2
        cy = y0 + h / 2
        return cx, cy, w, h, x0, y0, x1, y1

    def iou_matrix(self, bbox1, bbox2, format='xyxy', eps=1e-20):
        """ bbox1: n x 4
            bbox2: m x 4
        """
        if format == 'xyxy':
            b1_x0, b1_y0, b1_x1, b1_y1 = bbox1.T
            b2_x0, b2_y0, b2_x1, b2_y1 = bbox2.T
            b1_w, b1_h = b1_x1 - b1_x0, b1_y1 - b1_y0
            b2_w, b2_h = b2_x1 - b2_x0, b2_y1 - b2_y0
        elif format == 'cxywh':
            b1_w, b1_h, b1_x0, b1_y0, b1_x1, b1_y1 = self.parse_cxywh(bbox1)[2:]
            b2_w, b2_h, b2_x0, b2_y0, b2_x1, b2_y1 = self.parse_cxywh(bbox2)[2:]
        elif format == 'xywh':
            b1_w, b1_h, b1_x0, b1_y0, b1_x1, b1_y1 = self.parse_xywh(bbox1)[2:]
            b2_w, b2_h, b2_x0, b2_y0, b2_x1, b2_y1 = self.parse_xywh(bbox2)[2:]

        inter_w = (torch.min(b1_x1[:, None], b2_x1) - torch.max(b1_x0[:, None], b2_x0)).clamp(0)  # n x m
        inter_h = (torch.min(b1_y1[:, None], b2_y1) - torch.max(b1_y0[:, None], b2_y0)).clamp(0)  # n x m
        inter_area = inter_w * inter_h

        b1_area = b1_w * b1_h
        b2_area = b2_w * b2_h
        union_area = b1_area[:, None] + b2_area - inter_area
        iou_matrix = inter_area / (union_area + eps)
        return iou_matrix

    @staticmethod
    def revert_preds(pred_xyxy, ratio, border_left, border_top, img_w, img_h):
        pred_xyxy[:, [0, 2]] -= border_left
        pred_xyxy[:, [1, 3]] -= border_top
        pred_xyxy /= ratio
        pred_xyxy[:, [0, 2]] = pred_xyxy[:, [0, 2]].clip(0, img_w)
        pred_xyxy[:, [1, 3]] = pred_xyxy[:, [1, 3]].clip(0, img_h)
        return pred_xyxy
