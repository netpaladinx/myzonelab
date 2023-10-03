import math

import torch
import torch.nn as nn

from ...registry import DETECT_LOSSES, POSTPROCESSORS
from ..base_module import BaseModule
from .focal_loss import FocalLoss


@DETECT_LOSSES.register_class('cls_bce')
class DetectClsBCE(BaseModule):
    """ Binary Cross Entropy on classes """

    def __init__(self, cls_pos_weight=1., focal_loss_gamma=0., loss_weight=1.):
        super().__init__()
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_pos_weight]))  # default reduction: mean
        if focal_loss_gamma > 0:
            bce = FocalLoss(bce, focal_loss_gamma)
        self.criterion = bce
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        return loss * self.loss_weight


@DETECT_LOSSES.register_class('obj_bce')
class DetectObjBCE(BaseModule):
    """ Binary Cross Entropy on objectness """

    def __init__(self, obj_pos_weight=1., focal_loss_gamma=0., loss_weight=1.):
        super().__init__()
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pos_weight]))  # default reduction: mean
        if focal_loss_gamma > 0:
            bce = FocalLoss(bce, focal_loss_gamma)
        self.criterion = bce
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        return loss * self.loss_weight


@DETECT_LOSSES.register_class('bbox_iou')
class DetectBBoxIoU(BaseModule):
    def __init__(self, bbox_format='cxywh', GIoU=False, DIoU=False, CIoU=False, loss_weight=1., eps=1e-7):
        super().__init__()
        self.bbox_format = bbox_format
        self.GIoU = GIoU
        self.DIoU = DIoU
        self.CIoU = CIoU
        self.loss_weight = loss_weight
        self.eps = eps
        self.bbox_process = POSTPROCESSORS.create({'type': 'detect_postprocessor.process'})

    def forward(self, pred_bbox, target_bbox):
        """ pred_bbox: n x 4
            target_bbox: n x 4
        """
        if self.bbox_format == 'cxywh':
            pred_cx, pred_cy, pred_w, pred_h, pred_x0, pred_y0, pred_x1, pred_y1 = self.bbox_process.parse_cxywh(pred_bbox)
            tar_cx, tar_cy, tar_w, tar_h, tar_x0, tar_y0, tar_x1, tar_y1 = self.bbox_process.parse_cxywh(target_bbox)
        elif self.bbox_format == 'xyxy':
            pred_cx, pred_cy, pred_w, pred_h, pred_x0, pred_y0, pred_x1, pred_y1 = self.bbox_process.parse_xyxy(pred_bbox)
            tar_cx, tar_cy, tar_w, tar_h, tar_x0, tar_y0, tar_x1, tar_y1 = self.bbox_process.parse_xyxy(target_bbox)
        elif self.bbox_format == 'xywh':
            pred_cx, pred_cy, pred_w, pred_h, pred_x0, pred_y0, pred_x1, pred_y1 = self.bbox_process.parse_xywh(pred_bbox)
            tar_cx, tar_cy, tar_w, tar_h, tar_x0, tar_y0, tar_x1, tar_y1 = self.bbox_process.parse_xywh(target_bbox)
        else:
            raise ValueError(f"Invalid bbox_format {self.bbox_format}")

        # intersaction area
        inter_w = (torch.min(pred_x1, tar_x1) - torch.max(pred_x0, tar_x0)).clamp(0)
        inter_h = (torch.min(pred_y1, tar_y1) - torch.max(pred_y0, tar_y0)).clamp(0)
        inter_area = inter_w * inter_h

        # union area
        pred_h, tar_h = pred_h + self.eps, tar_h + self.eps
        pred_area = pred_w * pred_h
        tar_area = tar_w * tar_h
        union_area = pred_area + tar_area - inter_area + self.eps

        iou = inter_area / union_area

        if self.GIoU or self.DIoU or self.CIoU:
            convex_w = torch.max(pred_x1, tar_x1) - torch.min(pred_x0, tar_x0)  # convex (smallest enclosing box) width
            convex_h = torch.max(pred_y1, tar_y1) - torch.min(pred_y0, tar_y0)  # convex height

            if self.CIoU or self.DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                convex_diag = convex_w ** 2 + convex_h ** 2 + self.eps
                center_dist = (pred_cx - tar_cx) ** 2 + (pred_cy - tar_cy) ** 2

                if self.DIoU:
                    score = iou - center_dist / convex_diag

                elif self.CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    pred_angle = torch.atan(pred_w / pred_h)
                    tar_angle = torch.atan(tar_w / tar_h)
                    diff_angle = 4 / math.pi ** 2 * torch.pow(tar_angle - pred_angle, 2)
                    with torch.no_grad():
                        alpha = diff_angle / (diff_angle + (1 - iou) + self.eps)
                    score = iou - center_dist / convex_diag - diff_angle * alpha

            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                convex_area = convex_w * convex_h + self.eps
                score = iou - (convex_area - union_area) / convex_area

        else:
            score = iou  # n
        loss = (1 - score).mean()
        return loss * self.loss_weight, score


@DETECT_LOSSES.register_class('bbox_obj_cls')
class DetectBBoxObjCls(BaseModule):
    def __init__(self, cls_pos_weight=1., obj_pos_weight=1., focal_loss_gamma=0., GIoU=False, DIoU=False, CIoU=False,
                 cls_loss_weight=1., obj_loss_weight=1., bbox_loss_weight=1., layer_obj_weights=(1., 1., 1.),
                 label_smoothing=0., eps=1e-10):
        super().__init__()
        self.cls_bce = DetectClsBCE(cls_pos_weight, focal_loss_gamma, cls_loss_weight)
        self.obj_bce = DetectObjBCE(obj_pos_weight, focal_loss_gamma, obj_loss_weight)
        self.bbox_iou = DetectBBoxIoU('cxywh', GIoU, DIoU, CIoU, bbox_loss_weight, eps)

        self.layer_obj_weights = layer_obj_weights
        self.cls_pos_smoothing = 1 - label_smoothing * 0.5
        self.cls_neg_smoothing = label_smoothing * 0.5
        self.eps = eps

    def forward(self, output, target_cxywh, target_cij, target_cls, target_anc_idx, target_anc, target_cnt):
        """ output: list(bs x n_anchors x h x w x out_dims) (out_dims: cx,cy,w,h,obj,n_cls)
            target_cxywh: list(n_tar x 4)
            target_cij: list(n_tar x 2)
            target_cls: list(n_tar)
            target_anc: list(n_tar x 2)
            target_anc_idx: list(n_tar)
            target_cnt: list(bs)
        """
        device = output[0].device
        loss_bbox = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        loss_cls = torch.zeros(1, device=device)

        for i, layer_out in enumerate(output):  # anchor layer i
            tar_cxywh = target_cxywh[i]
            tar_cij = target_cij[i]
            tar_cls = target_cls[i]
            tar_anc_idx = target_anc_idx[i]
            tar_anc = target_anc[i]
            tar_cnt = target_cnt[i]

            device = tar_cnt.device
            max_cnt = tar_cnt.max()
            n_tar = tar_cnt.sum()
            bs = len(tar_cnt)
            bs_, n_anc, out_h, out_w, _ = layer_out.shape
            assert bs == bs_
            tar_obj_prob = torch.zeros((bs, n_anc, out_h, out_w)).to(device)  # bs x n_anchors x h x w

            if n_tar > 0:
                bi_grid, cnt_grid = torch.meshgrid(torch.arange(bs).to(device),
                                                   torch.arange(max_cnt).to(device))
                batch_idx = bi_grid[cnt_grid < tar_cnt[:, None]]  # n_tar

                # layer_out: bs x n_anchors x h x w x out_dims (!!!notice: h x w not w x h)
                pred_out = layer_out[batch_idx, tar_anc_idx, tar_cij[:, 1], tar_cij[:, 0]]  # n_tar x out_dims

                # regression loss
                cxy = pred_out[:, :2].sigmoid() * 2. - 0.5  # -0.5 ~ 1.5
                wh = (pred_out[:, 2:4].sigmoid() * 2.) ** 2 * tar_anc  # 0 ~ 4 * anchor
                pred_bbox = torch.cat((cxy, wh), 1)  # n_tar x 4
                bbox_loss, iou_score = self.bbox_iou(pred_bbox, tar_cxywh)  # iou_score: n_tar, tar_cxywh: n_tar x 4
                loss_bbox += bbox_loss

                # set objectness by iou
                score = iou_score.detach().clamp(0)
                # !!!notice: h x w not w x h
                tar_obj_prob[batch_idx, tar_anc_idx, tar_cij[:, 1], tar_cij[:, 0]] = score

                # classification loss
                pred_cls_logit = pred_out[:, 5:]
                tar_cls_prob = torch.full_like(pred_cls_logit, self.cls_neg_smoothing).to(device)  # n_tar x n_cls
                tar_cls_prob[range(n_tar), tar_cls] = self.cls_pos_smoothing
                cls_loss = self.cls_bce(pred_cls_logit, tar_cls_prob)  # tar_cls_prob: n_tar x n_cls
                loss_cls += cls_loss

            pred_obj_logit = layer_out[..., 4]
            obj_loss = self.obj_bce(pred_obj_logit, tar_obj_prob)  # tar_obj_prob: bs x n_anchors x h x w
            loss_obj += obj_loss * self.layer_obj_weights[i]

        loss = (loss_bbox + loss_obj + loss_cls) * bs
        return {'loss': loss, 'lbbox': loss_bbox.detach(), 'lobj': loss_obj.detach(), 'lcls': loss_cls.detach()}
