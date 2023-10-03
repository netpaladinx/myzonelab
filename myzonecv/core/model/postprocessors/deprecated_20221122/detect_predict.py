import numpy as np
import torch
import torchvision

from ...registry import DETECT_POSTPROCESSOR
from ...utils import npf
from ...data.dataconsts import (CAT_ID_FIGHTER_LEFT, CAT_ID_FIGHTER_RIGHT,
                                CAT_ID_FIGHTER_NEAR, CAT_ID_FIGHTER_FAR,
                                CAT_ID_FIGHTER_UP, CAT_ID_FIGHTER_DOWN,
                                CAT_ID_FIGHTER_CROWD, CAT_ID_FIGHTER_SINGLE, CAT_ID_FIGHTER_NONE)
from .base_process import BaseProcess
from .detect_process import DetectProcess


@DETECT_POSTPROCESSOR.register_class('predict')
class DetectPredict(BaseProcess):
    def __init__(self,
                 anchors,
                 default='predict_box_obj_cls',
                 conf_thr=0.25,
                 iou_thr=0.45,
                 min_wh=2,
                 max_wh=4096,
                 max_det=300,    # max bboxes to return
                 max_nms=30000,  # max bboxes to call nms
                 require_redundant=True,  # used when merge_bboxes is True
                 merge_bboxes=False):
        super().__init__(default)
        self.n_layers = len(anchors)
        self.n_anchors = len(anchors[0]) // 2
        anchors = torch.tensor(anchors).float().view(self.n_layers, -1, 2)
        self.register_buffer('anchors', anchors)
        self.anchor_rescaled = False
        self.strides = None

        self.ij_grids = [None] * self.n_layers  # elem's shape: 1,n_anchors,h,w,2
        self.wh_scale_grids = [None] * self.n_layers  # elem's shape: 1,n_anchors,h,w,2

        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.max_det = max_det
        self.min_wh = min_wh
        self.max_wh = max_wh
        self.max_nms = max_nms
        self.require_redundant = require_redundant
        self.merge_bboxes = merge_bboxes

        self.process = DetectProcess()

    def predict_box_obj_cls(self, output):
        """ output: list(bs x n_anchors x out_h x out_w x out_dims)
        """
        assert self.anchor_rescaled is True

        preds = []
        for i, out in enumerate(output):
            bs, _, h, w, d = out.shape
            if self.ij_grids[i] is None or tuple(self.ij_grids[i].shape[2:4]) != (h, w):
                self.ij_grids[i], self.wh_scale_grids[i] = self.make_grid(w, h, self.anchors[i], self.strides[i])

            pred = out.sigmoid()
            cxy, wh = pred[..., 0:2], pred[..., 2:4]
            cxy = (cxy * 2. - 0.5 + self.ij_grids[i]) * self.strides[i]  # (0,1) => ((-0.5,1.5) + offset) * stride
            wh = (wh * 2)**2 * self.wh_scale_grids[i]  # (0,1) => (0,2**2) * (anchors * stride)
            pred = torch.cat((cxy, wh, pred[..., 4:]), -1).view(bs, -1, d)
            preds.append(pred)
        return preds  # list(bs x (n_anchors*out_h*out_w) x out_dims)

    def make_grid(self, w, h, anchors, stride):
        device = anchors.device
        j_grid, i_grid = torch.meshgrid(torch.arange(h).to(device), torch.arange(w).to(device))
        ij_grid = torch.stack((i_grid, j_grid), dim=2).expand((1, self.n_anchors, h, w, 2)).float()
        wh_scale_grid = (anchors.clone() * stride).view((1, self.n_anchors, 1, 1, 2)).expand((1, self.n_anchors, h, w, 2)).float()
        return ij_grid, wh_scale_grid

    def init_strides(self, strides):
        # rescale anchors
        assert len(self.anchors) == len(strides)
        for i, stride in enumerate(strides):
            self.anchors[i] /= stride
        self.anchor_rescaled = True
        self.strides = strides

    def non_max_suppression(self, preds, conf_thr, iou_thr, topk=None):
        """ preds: bs x n_all_preds x out_dims
        """
        nms_preds = [torch.zeros((0, 7)).to(preds)] * preds.shape[0]
        filtered = ((preds[..., 2:4] > self.min_wh).all(-1) &  # not too small
                    (preds[..., 2:4] < self.max_wh).all(-1) &  # not too big
                    (preds[..., 4] > conf_thr))           # objectness
        max_det = self.max_det if topk is None else topk

        for i, pred in enumerate(preds):  # pred: n_pred x out_dims
            pred = pred[filtered[i]]  # n_pred2 x out_dims
            if pred.shape[0] == 0:
                continue

            pred_x0, pred_y0, pred_x1, pred_y1 = self.process.parse_cxywh(pred[:, :4])[4:]
            pred[:, 5:] *= pred[:, 4:5]  # conf = obj_conf * cls_conf
            r, c = (pred[:, 5:] > conf_thr).nonzero(as_tuple=False).T

            # row: x0, y0, x1, y1, conf, cls, obj_conf
            res = torch.stack((pred_x0[r], pred_y0[r], pred_x1[r], pred_y1[r], pred[r, c + 5], c.float(), pred[r, 4]), 1)  # n_pred3 x 7

            if res.shape[0] == 0:
                continue
            elif res.shape[0] > self.max_nms:
                res = res[res[:, 4].argsort(descending=True)[:self.max_nms]]  # n_pred4 x 7

            cls_offset = res[:, 5:6] * self.max_wh
            xyxy = res[:, :4] + cls_offset  # n_pred4 x 4
            conf = res[:, 4]  # n_pred4
            nms_indices = torchvision.ops.nms(xyxy, conf, iou_thr)  # n_pred5
            if nms_indices.shape[0] > max_det:
                nms_indices = nms_indices[:max_det]

            if self.merge_bboxes:
                iou_hit = self.process.iou_matrix(xyxy[nms_indices], xyxy) > iou_thr  # n_pred4 x n_pred5
                iou_wei = iou_hit * conf[None]
                res[nms_indices, :4] = torch.mm(iou_wei, res[:, :4]).float() / iou_wei.sum(1, keepdim=True)
                if self.require_redundant:
                    nms_indices = nms_indices[iou_hit.sum(1) > 1]

            nms_preds[i] = res[nms_indices]  # n_pred6 x 7
        return nms_preds

    def get_direct_results(self, preds, conf_thr=None, iou_thr=None, topk=None):
        """ preds (torch.Tensor): bs x n_all_preds x out_dims

            Returned results correspond to input images
        """
        conf_thr = conf_thr or self.conf_thr
        iou_thr = iou_thr or self.iou_thr
        preds = self.non_max_suppression(preds, conf_thr, iou_thr, topk)
        img_idx, xyxy_results, cls_results, conf_results, obj_conf_results, cls_conf_results = [], [], [], [], [], []
        for i, pred in enumerate(preds):
            if len(pred) > 0:
                pred = pred.detach().cpu().numpy()
                img_idx.append(i)
                xyxy_results.append(pred[:, :4])
                cls_results.append(pred[:, 5])
                conf_results.append(pred[:, 4])
                obj_conf_results.append(pred[:, 6])
                cls_conf_results.append(conf_results[-1] / obj_conf_results[-1].clip(1e-10))
        return {
            'img_idx': img_idx,
            'xyxy_results': xyxy_results,
            'cls_results': cls_results,
            'conf_results': conf_results,
            'obj_conf_results': obj_conf_results,
            'cls_conf_results': cls_conf_results
        }

    def get_final_results(self, preds, batch_dict, conf_thr=None, iou_thr=None, topk=None):
        """ preds (torch.Tensor): bs x n_all_preds x out_dims

            Returned results correspond to original images
        """
        conf_thr = conf_thr or self.conf_thr
        iou_thr = iou_thr or self.iou_thr
        preds = self.non_max_suppression(preds, conf_thr, iou_thr, topk)
        img_ids, xyxy_results, cls_results, conf_results, obj_conf_results, cls_conf_results = [], [], [], [], [], []
        for i, pred in enumerate(preds):
            if len(pred) > 0:
                pred = pred.detach().cpu().numpy()
                ratio, (border_top, _, border_left, _), (img_w, img_h) = batch_dict['_revert_params'][i]
                img_ids.append(batch_dict['img_id'][i])
                xyxy_results.append(self.process.revert_preds(pred[:, :4], ratio, border_left, border_top, img_w, img_h))
                cls_results.append(pred[:, 5])
                conf_results.append(pred[:, 4])
                obj_conf_results.append(pred[:, 6])
                cls_conf_results.append(conf_results[-1] / obj_conf_results[-1].clip(1e-10))
        return {
            'img_ids': img_ids,
            'xyxy_results': xyxy_results,
            'cls_results': cls_results,
            'conf_results': conf_results,
            'obj_conf_results': obj_conf_results,
            'cls_conf_results': cls_conf_results
        }


@DETECT_POSTPROCESSOR.register_class('predict_myzone')
class DetectPredictMyZone(DetectPredict):
    duplicates = 2

    def __init__(self,
                 anchors,
                 default='predict_box_obj_cls',
                 cxy_margin=0.0,
                 anchor_thr=2.9,
                 wh_margin=0.05,
                 wh_gamma=2,
                 conf_thr=0.01,
                 iou_thr=0.6,
                 min_wh=8,
                 max_wh=640,
                 max_nms=30000,  # max bboxes to call nms
                 ):
        BaseProcess.__init__(self, default)
        self.n_layers = len(anchors)
        self.n_anchors = len(anchors[0]) // 2 * self.duplicates  # 6
        anchors = torch.tensor(np.tile(npf(anchors), self.duplicates)).view(self.n_layers, -1, 2)  # 3 x 6 x 2
        self.register_buffer('anchors', anchors)
        self.anchor_rescaled = False
        self.strides = None

        self.ij_grids = [None] * self.n_layers  # elem's shape: 1,n_anchors,h,w,2
        self.wh_scale_grids = [None] * self.n_layers  # elem's shape: 1,n_anchors,h,w,2

        self.cxy_margin = cxy_margin
        self.anchor_thr = anchor_thr
        self.wh_margin = wh_margin
        self.wh_gamma = wh_gamma
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.min_wh = min_wh
        self.max_wh = max_wh
        self.max_nms = max_nms

        self.process = DetectProcess()

    def predict_box_obj_cls(self, output):
        """ output: list(bs x n_anchors(6) x out_h x out_w x out_dims)
        """
        assert self.anchor_rescaled is True

        preds = []
        for i, out in enumerate(output):
            bs, _, h, w, d = out.shape
            if self.ij_grids[i] is None or tuple(self.ij_grids[i].shape[2:4]) != (h, w):
                self.ij_grids[i], self.wh_scale_grids[i] = self.make_grid(w, h, self.anchors[i], self.strides[i])

            pred = out.sigmoid()
            cxy, wh = pred[..., 0:2], pred[..., 2:4]
            cxy = (cxy * (2 + self.cxy_margin * 2) - (0.5 + self.cxy_margin) + self.ij_grids[i]) * self.strides[i]  # (0,1) => ((-0.5,1.5) + offset) * stride
            # wh = (wh * 2)**2 * self.wh_scale_grids[i]  # (0,1) => (0,2**2) * (anchors * stride)
            wh = (wh**self.wh_gamma) * (self.anchor_thr + self.wh_margin * 2) * self.wh_scale_grids[i]  # (0,1) => (0,3) * (anchors * stride)
            pred = torch.cat((cxy, wh, pred[..., 4:]), -1).view(bs, -1, d)
            preds.append(pred)
        return preds  # list(bs x (n_anchors*out_h*out_w) x out_dims)

    def filter_predictions(self, preds, conf_thr, iou_thr):
        """ preds: bs x n_all_preds x out_dims, (out_dims: 5 + 9)
        """
        filtered = ((preds[..., 2:4] > self.min_wh).all(-1) &  # not too small
                    (preds[..., 2:4] < self.max_wh).all(-1) &  # not too big
                    (preds[..., 4] > conf_thr))                # objectness

        # row: x0, y0, x1, y1, conf, cls, obj_conf
        filter_preds = [torch.zeros((0, 7)).to(preds)] * preds.shape[0]

        for i, pred in enumerate(preds):  # pred: n_all_preds x out_dims
            pred = pred[filtered[i]]  # n_pred2 x out_dims
            if pred.shape[0] == 0:
                continue

            pred_x0, pred_y0, pred_x1, pred_y1 = self.process.parse_cxywh(pred[:, :4])[4:]
            pred[:, 5:] *= pred[:, 4:5]  # conf = obj_conf * cls_conf

            # left-right
            pred_left_best, left_row_idx = pred[:, 5 + CAT_ID_FIGHTER_LEFT].max(dim=0)
            pred2 = pred.clone()
            pred2[left_row_idx, 5 + CAT_ID_FIGHTER_RIGHT] = 0
            pred_right_best, right_row_idx = pred2[:, 5 + CAT_ID_FIGHTER_RIGHT].max(dim=0)
            cls_pair = 'left-right'
            cls_best = torch.min(pred_left_best, pred_right_best)

            # near-far
            pred_near_best, near_row_idx = pred[:, 5 + CAT_ID_FIGHTER_NEAR].max(dim=0)
            pred2 = pred.clone()
            pred2[near_row_idx, 5 + CAT_ID_FIGHTER_FAR] = 0
            pred_far_best, far_row_idx = pred2[:, 5 + CAT_ID_FIGHTER_FAR].max(dim=0)
            cls_best_2 = torch.min(pred_near_best, pred_far_best)
            if cls_best_2 > cls_best:
                cls_pair = 'near-far'
                cls_best = cls_best_2

            # up-down
            pred_up_best, up_row_idx = pred[:, 5 + CAT_ID_FIGHTER_UP].max(dim=0)
            pred2 = pred.clone()
            pred2[up_row_idx, 5 + CAT_ID_FIGHTER_DOWN] = 0
            pred_down_best, down_row_idx = pred2[:, 5 + CAT_ID_FIGHTER_DOWN].max(dim=0)
            cls_best_2 = torch.min(pred_up_best, pred_down_best)
            if cls_best_2 > cls_best:
                cls_pair = 'up-down'
                cls_best = cls_best_2

            # crowd-crowd
            pred_crowd = pred[:, 5 + CAT_ID_FIGHTER_CROWD]
            r = (pred_crowd > conf_thr).nonzero(as_tuple=True)[0]
            # row: x0, y0, x1, y1, conf, obj_conf, row_idx
            res = torch.stack((pred_x0[r], pred_y0[r], pred_x1[r], pred_y1[r], pred_crowd[r], pred[r, 4], r.float()), 1)  # n_pred3 x 7
            if res.shape[0] > 0:
                if res.shape[0] > self.max_nms:
                    res = res[res[:, 4].argsort(descending=True)[:self.max_nms]]  # n_pred4 x 6
                xyxy = res[:, :4]  # n_pred4 x 4
                conf = res[:, 4]  # n_pred4
                nms_indices = torchvision.ops.nms(xyxy, conf, iou_thr)[:2]  # n_pred5
                if len(nms_indices) > 0:
                    pred_crowd1_best, crowd1_row_idx = res[nms_indices[0]][[4, 6]]
                    crowd1_row_idx = crowd1_row_idx.long()
                    if len(nms_indices) > 1:
                        pred_crowd2_best, crowd2_row_idx = res[nms_indices[1]][[4, 6]]
                        crowd2_row_idx = crowd2_row_idx.long()
                        cls_best_2 = torch.min(pred_crowd1_best, pred_crowd2_best)
                    else:
                        pred_crowd2_best, crowd2_row_idx = None, None
                        cls_best_2 = pred_crowd1_best
                    if cls_best_2 > cls_best:
                        cls_pair = 'crowd-crowd'
                        cls_best = cls_best_2

            # single
            pred_single_best, single_row_idx = pred[:, 5 + CAT_ID_FIGHTER_SINGLE].max(dim=0)
            if pred_single_best > cls_best:
                cls_pair = 'single'
                cls_best = pred_single_best

            # none
            pred_none_best, none_row_idx = pred[:, 5 + CAT_ID_FIGHTER_NONE].max(dim=0)
            if pred_none_best > cls_best:
                cls_pair = 'none'
                cls_best = pred_none_best

            if cls_best <= conf_thr or cls_pair == 'none':
                continue

            if cls_pair == 'left-right':
                row_idx = [left_row_idx.item(), right_row_idx.item()]
                col_idx = [5 + CAT_ID_FIGHTER_LEFT, 5 + CAT_ID_FIGHTER_RIGHT]
                cls_idx = torch.tensor([CAT_ID_FIGHTER_LEFT, CAT_ID_FIGHTER_RIGHT]).to(pred)
            elif cls_pair == 'near-far':
                row_idx = [near_row_idx.item(), far_row_idx.item()]
                col_idx = [5 + CAT_ID_FIGHTER_NEAR, 5 + CAT_ID_FIGHTER_FAR]
                cls_idx = torch.tensor([CAT_ID_FIGHTER_NEAR, CAT_ID_FIGHTER_FAR]).to(pred)
            elif cls_pair == 'up-down':
                row_idx = [up_row_idx.item(), down_row_idx.item()]
                col_idx = [5 + CAT_ID_FIGHTER_UP, 5 + CAT_ID_FIGHTER_DOWN]
                cls_idx = torch.tensor([CAT_ID_FIGHTER_UP, CAT_ID_FIGHTER_DOWN]).to(pred)
            elif cls_pair == 'crowd-crowd':
                if crowd2_row_idx is not None:
                    row_idx = [crowd1_row_idx.item(), crowd2_row_idx.item()]
                    col_idx = [5 + CAT_ID_FIGHTER_CROWD, 5 + CAT_ID_FIGHTER_CROWD]
                    cls_idx = torch.tensor([CAT_ID_FIGHTER_CROWD, CAT_ID_FIGHTER_CROWD]).to(pred)
                else:
                    row_idx = [crowd1_row_idx.item()]
                    col_idx = [5 + CAT_ID_FIGHTER_CROWD]
                    cls_idx = torch.tensor([CAT_ID_FIGHTER_CROWD]).to(pred)
            elif cls_pair == 'single':
                row_idx = [single_row_idx.item()]
                col_idx = [5 + CAT_ID_FIGHTER_SINGLE]
                cls_idx = torch.tensor([CAT_ID_FIGHTER_SINGLE]).to(pred)
            else:
                raise ValueError(f'cls_pair = {cls_pair} should not happen here')

            filter_preds[i] = torch.stack((pred_x0[row_idx], pred_y0[row_idx], pred_x1[row_idx], pred_y1[row_idx],
                                           pred[row_idx, col_idx], cls_idx, pred[row_idx, 4]), 1)
        return filter_preds

    def get_direct_results(self, preds, conf_thr=None, iou_thr=None):
        """ preds (torch.Tensor): bs x n_all_preds x out_dims

            Returned results correspond to input images
        """
        conf_thr = conf_thr or self.conf_thr
        iou_thr = iou_thr or self.iou_thr
        preds = self.filter_predictions(preds, conf_thr, iou_thr)
        img_idx, xyxy_results, cls_results, conf_results, obj_conf_results, cls_conf_results = [], [], [], [], [], []
        for i, pred in enumerate(preds):
            if len(pred) > 0:
                pred = pred.detach().cpu().numpy()
                img_idx.append(i)
                xyxy_results.append(pred[:, :4])
                cls_results.append(pred[:, 5].astype(int))
                conf_results.append(pred[:, 4])
                obj_conf_results.append(pred[:, 6])
                cls_conf_results.append(conf_results[-1] / obj_conf_results[-1].clip(1e-10))
            else:
                img_idx.append(i)
                xyxy_results.append(np.empty((0, 4)))
                cls_results.append(np.empty((0,)).astype(int))
                conf_results.append(np.empty((0,)))
                obj_conf_results.append(np.empty((0,)))
                cls_conf_results.append(np.empty((0,)))
        return {
            'img_idx': img_idx,
            'xyxy_results': xyxy_results,
            'cls_results': cls_results,
            'conf_results': conf_results,
            'obj_conf_results': obj_conf_results,
            'cls_conf_results': cls_conf_results
        }

    def get_final_results(self, preds, batch_dict, conf_thr=None, iou_thr=None):
        """ preds (torch.Tensor): bs x n_all_preds x out_dims

            Returned results correspond to original images
        """
        conf_thr = conf_thr or self.conf_thr
        iou_thr = iou_thr or self.iou_thr
        preds = self.filter_predictions(preds, conf_thr, iou_thr)
        img_ids, xyxy_results, cls_results, conf_results, obj_conf_results, cls_conf_results = [], [], [], [], [], []
        for i, pred in enumerate(preds):
            if len(pred) > 0:
                pred = pred.detach().cpu().numpy()
                ratio, (border_top, _, border_left, _), (img_w, img_h) = batch_dict['_revert_params'][i]
                img_ids.append(batch_dict['img_id'][i])
                xyxy_results.append(self.process.revert_preds(pred[:, :4], ratio, border_left, border_top, img_w, img_h))
                cls_results.append(pred[:, 5])
                conf_results.append(pred[:, 4])
                obj_conf_results.append(pred[:, 6])
                cls_conf_results.append(conf_results[-1] / obj_conf_results[-1].clip(1e-10))
            else:
                img_ids.append(batch_dict['img_id'][i])
                xyxy_results.append(np.empty((0, 4)))
                cls_results.append(np.empty((0,)).astype(int))
                conf_results.append(np.empty((0,)))
                obj_conf_results.append(np.empty((0,)))
                cls_conf_results.append(np.empty((0,)))
        return {
            'img_ids': img_ids,
            'xyxy_results': xyxy_results,
            'cls_results': cls_results,
            'conf_results': conf_results,
            'obj_conf_results': obj_conf_results,
            'cls_conf_results': cls_conf_results
        }


@DETECT_POSTPROCESSOR.register_class('predict_myzone_v2')
class DetectPredictMyZoneV2(DetectPredictMyZone):
    duplicates = None

    def __init__(self,
                 anchors,
                 default='predict_box_obj_cls',
                 cxy_margin=0.0,
                 anchor_thr=2.9,
                 wh_margin=0.05,
                 wh_gamma=2,
                 conf_thr=0.01,
                 iou_thr=0.6,
                 min_wh=8,
                 max_wh=640,
                 max_nms=30000,  # max bboxes to call nms
                 ):
        BaseProcess.__init__(self, default)
        self.n_layers = len(anchors)
        self.n_anchors = len(anchors[0]) // 2
        anchors = torch.tensor(npf(anchors)).view(self.n_layers, -1, 2)  # 3 x n_anchors x 2
        self.register_buffer('anchors', anchors)
        self.anchor_rescaled = False
        self.strides = None

        self.ij_grids = [None] * self.n_layers  # elem's shape: 1,n_anchors,h,w,2
        self.wh_scale_grids = [None] * self.n_layers  # elem's shape: 1,n_anchors,h,w,2

        self.cxy_margin = cxy_margin
        self.anchor_thr = anchor_thr
        self.wh_margin = wh_margin
        self.wh_gamma = wh_gamma
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.min_wh = min_wh
        self.max_wh = max_wh
        self.max_nms = max_nms

        self.process = DetectProcess()

    @staticmethod
    def get_top_pair(pred):
        """ pred: n x 2
        """
        values, indices = torch.topk(pred, 2, dim=0)
        val1, val2 = values.T
        idx1, idx2 = indices.T
        val_pair = [val1[0], val2[0]]
        idx_pair = [idx1[0], idx2[0]]
        if idx1[0] == idx2[0]:
            if val1[0] >= val2[0]:
                idx_pair[1] = idx2[1]
                val_pair[1] = val2[1]
            else:
                idx_pair[0] = idx1[1]
                val_pair[0] = val1[1]
        return val_pair, idx_pair

    def filter_predictions(self, preds, conf_thr, iou_thr):
        """ preds: bs x n_all_preds x out_dims, (out_dims: 5 + 9)
        """
        filtered = ((preds[..., 2:4] > self.min_wh).all(-1) &  # not too small
                    (preds[..., 2:4] < self.max_wh).all(-1) &  # not too big
                    (preds[..., 4] > conf_thr))                # objectness

        # row: x0, y0, x1, y1, conf, cls, obj_conf
        filter_preds = [torch.zeros((0, 7)).to(preds)] * preds.shape[0]

        for i, pred in enumerate(preds):  # pred: n_all_preds x out_dims
            pred = pred[filtered[i]]  # n_pred2 x out_dims
            if pred.shape[0] == 0:
                continue

            pred_x0, pred_y0, pred_x1, pred_y1 = self.process.parse_cxywh(pred[:, :4])[4:]
            pred[:, 5:] *= pred[:, 4:5]  # conf = obj_conf * cls_conf

            # none
            pred_none_best, _ = pred[:, 5 + CAT_ID_FIGHTER_NONE].max(dim=0)
            cls_pair = 'none'
            cls_best = pred_none_best

            # left-right
            if pred.shape[0] >= 2:
                (pred_left_best, pred_right_best), (left_row_idx, right_row_idx) = \
                    self.get_top_pair(pred[:, [5 + CAT_ID_FIGHTER_LEFT, 5 + CAT_ID_FIGHTER_RIGHT]])
                cls_best_2 = torch.min(pred_left_best, pred_right_best)
                if cls_best_2 > cls_best:
                    cls_pair = 'left-right'
                    cls_best = cls_best_2

            # near-far
            if pred.shape[0] >= 2:
                (pred_near_best, pred_far_best), (near_row_idx, far_row_idx) = \
                    self.get_top_pair(pred[:, [5 + CAT_ID_FIGHTER_NEAR, 5 + CAT_ID_FIGHTER_FAR]])
                cls_best_2 = torch.min(pred_near_best, pred_far_best)
                if cls_best_2 > cls_best:
                    cls_pair = 'near-far'
                    cls_best = cls_best_2

            # up-down
            if pred.shape[0] >= 2:
                (pred_up_best, pred_down_best), (up_row_idx, down_row_idx) = \
                    self.get_top_pair(pred[:, [5 + CAT_ID_FIGHTER_UP, 5 + CAT_ID_FIGHTER_DOWN]])
                cls_best_2 = torch.min(pred_up_best, pred_down_best)
                if cls_best_2 > cls_best:
                    cls_pair = 'up-down'
                    cls_best = cls_best_2

            # crowd-crowd
            if pred.shape[0] >= 2:
                pred_crowd = pred[:, 5 + CAT_ID_FIGHTER_CROWD]
                r = (pred_crowd > conf_thr).nonzero(as_tuple=True)[0]
                # row: x0, y0, x1, y1, conf, obj_conf, row_idx
                res = torch.stack((pred_x0[r], pred_y0[r], pred_x1[r], pred_y1[r], pred_crowd[r], pred[r, 4], r.float()), 1)  # n_pred3 x 7
                if res.shape[0] > 0:
                    if res.shape[0] > self.max_nms:
                        res = res[res[:, 4].argsort(descending=True)[:self.max_nms]]  # n_pred4 x 6
                    xyxy = res[:, :4]  # n_pred4 x 4
                    conf = res[:, 4]  # n_pred4
                    nms_indices = torchvision.ops.nms(xyxy, conf, iou_thr)[:2]  # n_pred5
                    if len(nms_indices) > 0:
                        pred_crowd1_best, crowd1_row_idx = res[nms_indices[0]][[4, 6]]
                        crowd1_row_idx = crowd1_row_idx.long()
                        if len(nms_indices) > 1:
                            pred_crowd2_best, crowd2_row_idx = res[nms_indices[1]][[4, 6]]
                            crowd2_row_idx = crowd2_row_idx.long()
                            cls_best_2 = torch.min(pred_crowd1_best, pred_crowd2_best)
                        else:
                            pred_crowd2_best, crowd2_row_idx = None, None
                            cls_best_2 = pred_crowd1_best
                        if cls_best_2 > cls_best:
                            cls_pair = 'crowd-crowd'
                            cls_best = cls_best_2

            # single
            pred_single_best, single_row_idx = pred[:, 5 + CAT_ID_FIGHTER_SINGLE].max(dim=0)
            if pred_single_best > cls_best:
                cls_pair = 'single'
                cls_best = pred_single_best

            if cls_best <= conf_thr or cls_pair == 'none':
                continue

            if cls_pair == 'left-right':
                row_idx = [left_row_idx.item(), right_row_idx.item()]
                col_idx = [5 + CAT_ID_FIGHTER_LEFT, 5 + CAT_ID_FIGHTER_RIGHT]
                cls_idx = torch.tensor([CAT_ID_FIGHTER_LEFT, CAT_ID_FIGHTER_RIGHT]).to(pred)
            elif cls_pair == 'near-far':
                row_idx = [near_row_idx.item(), far_row_idx.item()]
                col_idx = [5 + CAT_ID_FIGHTER_NEAR, 5 + CAT_ID_FIGHTER_FAR]
                cls_idx = torch.tensor([CAT_ID_FIGHTER_NEAR, CAT_ID_FIGHTER_FAR]).to(pred)
            elif cls_pair == 'up-down':
                row_idx = [up_row_idx.item(), down_row_idx.item()]
                col_idx = [5 + CAT_ID_FIGHTER_UP, 5 + CAT_ID_FIGHTER_DOWN]
                cls_idx = torch.tensor([CAT_ID_FIGHTER_UP, CAT_ID_FIGHTER_DOWN]).to(pred)
            elif cls_pair == 'crowd-crowd':
                if crowd2_row_idx is not None:
                    row_idx = [crowd1_row_idx.item(), crowd2_row_idx.item()]
                    col_idx = [5 + CAT_ID_FIGHTER_CROWD, 5 + CAT_ID_FIGHTER_CROWD]
                    cls_idx = torch.tensor([CAT_ID_FIGHTER_CROWD, CAT_ID_FIGHTER_CROWD]).to(pred)
                else:
                    row_idx = [crowd1_row_idx.item()]
                    col_idx = [5 + CAT_ID_FIGHTER_CROWD]
                    cls_idx = torch.tensor([CAT_ID_FIGHTER_CROWD]).to(pred)
            elif cls_pair == 'single':
                row_idx = [single_row_idx.item()]
                col_idx = [5 + CAT_ID_FIGHTER_SINGLE]
                cls_idx = torch.tensor([CAT_ID_FIGHTER_SINGLE]).to(pred)
            else:
                raise ValueError(f'cls_pair = {cls_pair} should not happen here')

            filter_preds[i] = torch.stack((pred_x0[row_idx], pred_y0[row_idx], pred_x1[row_idx], pred_y1[row_idx],
                                           pred[row_idx, col_idx], cls_idx, pred[row_idx, 4]), 1)
        return filter_preds


@DETECT_POSTPROCESSOR.register_class('predict_myzone_v3')
class DetectPredictMyZoneV3(DetectPredictMyZoneV2):
    bbox_dims = 5

    def predict_box_obj_cls(self, output):
        """ output: list(bs x n_anchors x out_h x out_w x (n_classes*bbox_dims) (bbox_dims: cx,cy,w,h,obj)
        """
        assert self.anchor_rescaled is True

        preds = []

        for i, out in enumerate(output):
            bs, n_anc, h, w, d = out.shape
            n_cls = int(d / self.bbox_dims)
            if self.ij_grids[i] is None or tuple(self.ij_grids[i].shape[2:4]) != (h, w):
                self.ij_grids[i], self.wh_scale_grids[i] = self.make_grid(w, h, n_cls, self.anchors[i], self.strides[i])

            pred = out.sigmoid()
            pred = pred.view(bs, n_anc, h, w, n_cls, self.bbox_dims)  # bs x n_anc x h x w x n_cls x 5
            cxy, wh = pred[..., 0:2], pred[..., 2:4]  # bs x n_anc x h x w x n_cls x 2
            cxy = (cxy * (2 + self.cxy_margin * 2) - (0.5 + self.cxy_margin) + self.ij_grids[i]) * self.strides[i]  # (0,1) => ((-0.5,1.5) + offset) * stride
            # wh = (wh * 2)**2 * self.wh_scale_grids[i]  # (0,1) => (0,2**2) * (anchors * stride)
            wh = (wh**self.wh_gamma) * (self.anchor_thr + self.wh_margin) * self.wh_scale_grids[i]  # (0,1) => (0,3) * (anchors * stride)
            pred = torch.cat((cxy, wh, pred[..., 4:]), -1).view(bs, -1, d)
            preds.append(pred)
        return preds  # list(bs x (n_anchors*out_h*out_w) x (n_classes*bbox_dims))

    def make_grid(self, w, h, n_cls, anchors, stride):
        """ anchors: n_anchors x 2
            w, h, n_cls, stride: scalar
        """
        device = anchors.device
        j_grid, i_grid = torch.meshgrid(torch.arange(h).to(device), torch.arange(w).to(device))
        ij_grid = torch.stack((i_grid, j_grid), dim=2).view(1, 1, h, w, 1, 2).expand((1, self.n_anchors, h, w, n_cls, 2)).float()
        wh_scale_grid = (anchors.clone() * stride).view(1, self.n_anchors, 1, 1, 1, 2).expand((1, self.n_anchors, h, w, n_cls, 2)).float()
        return ij_grid, wh_scale_grid  # 1 x n_anc x h x w x n_cls x 2, last_dim: (i, j) and (w, h)

    def filter_predictions(self, preds, conf_thr, iou_thr):
        """ preds: bs x n_all_preds x out_dims, (out_dims: n_classes*bbox_dims)
        """
        device = preds.device
        dtype = preds.dtype
        bs, n, d = preds.shape
        n_cls = int(d / self.bbox_dims)
        preds = list(preds.view(bs, n, n_cls, self.bbox_dims).split(1, dim=2))  # [bs x n_all_preds x 1 x bbox_dims, ...]

        filtered = []
        for c in range(n_cls):
            preds[c] = preds[c].squeeze(2)  # bs x n_all_preds x bbox_dims
            if c != CAT_ID_FIGHTER_NONE:
                filtered.append((preds[c][..., 2:4] > self.min_wh).all(-1) &  # not too small
                                (preds[c][..., 2:4] < self.max_wh).all(-1) &  # not too big
                                (preds[c][..., 4] > conf_thr))                # objectness
            else:
                filtered.append(preds[c][..., 4] > conf_thr)

        # row: x0, y0, x1, y1, conf, cls
        filter_preds = [torch.zeros((0, 6), device=device, dtype=dtype)] * bs

        for i in range(bs):
            pred, pred_x0, pred_y0, pred_x1, pred_y1 = [], [], [], [], []
            for c in range(n_cls):
                pr = preds[c][i]     # n_all_preds x bbox_dims
                fi = filtered[c][i]  # n_all_preds
                pr = pr[fi]          # n_pred2 x bbox_dims
                pred.append(pr)

                if pr.shape[0] > 0:
                    pr_x0, pr_y0, pr_x1, pr_y1 = self.process.parse_cxywh(pr[:, :4])[4:]
                else:
                    pr_x0, pr_y0, pr_x1, pr_y1 = None, None, None, None

                pred_x0.append(pr_x0)  # n_pred2
                pred_y0.append(pr_y0)  # n_pred2
                pred_x1.append(pr_x1)  # n_pred2
                pred_y1.append(pr_y1)  # n_pred2

            cls_pair = 'none'
            cls_best = conf_thr

            # none
            pr_none = pred[CAT_ID_FIGHTER_NONE]
            if pr_none.shape[0] > 0:
                pred_none_best, _ = pr_none[:, 4].max(dim=0)
                if pred_none_best > cls_best:
                    cls_pair = 'none'
                    cls_best = pred_none_best

            # left-right
            pr_left, pr_right = pred[CAT_ID_FIGHTER_LEFT], pred[CAT_ID_FIGHTER_RIGHT]
            if pr_left.shape[0] > 0 and pr_right.shape[0] > 0:
                pred_left_best, left_row_idx = pr_left[:, 4].max(dim=0)
                pred_right_best, right_row_idx = pr_right[:, 4].max(dim=0)
                cls_best_2 = torch.min(pred_left_best, pred_right_best)
                if cls_best_2 > cls_best:
                    cls_pair = 'left-right'
                    cls_best = cls_best_2

            # near-far
            pr_near, pr_far = pred[CAT_ID_FIGHTER_NEAR], pred[CAT_ID_FIGHTER_FAR]
            if pr_near.shape[0] > 0 and pr_far.shape[0] > 0:
                pred_near_best, near_row_idx = pr_near[:, 4].max(dim=0)
                pred_far_best, far_row_idx = pr_far[:, 4].max(dim=0)
                cls_best_2 = torch.min(pred_near_best, pred_far_best)
                if cls_best_2 > cls_best:
                    cls_pair = 'near-far'
                    cls_best = cls_best_2

            # up-down
            pr_up, pr_down = pred[CAT_ID_FIGHTER_UP], pred[CAT_ID_FIGHTER_DOWN]
            if pr_up.shape[0] > 0 and pr_down.shape[0] > 0:
                pred_up_best, up_row_idx = pr_up[:, 4].max(dim=0)
                pred_down_best, down_row_idx = pr_down[:, 4].max(dim=0)
                cls_best_2 = torch.min(pred_up_best, pred_down_best)
                if cls_best_2 > cls_best:
                    cls_pair = 'up-down'
                    cls_best = cls_best_2

            # crowd-crowd
            pr_crowd = pred[CAT_ID_FIGHTER_CROWD]
            pr_x0 = pred_x0[CAT_ID_FIGHTER_CROWD]
            pr_y0 = pred_y0[CAT_ID_FIGHTER_CROWD]
            pr_x1 = pred_x1[CAT_ID_FIGHTER_CROWD]
            pr_y1 = pred_y1[CAT_ID_FIGHTER_CROWD]
            if pr_crowd.shape[0] > 1:
                r = (pr_crowd[:, 4] > conf_thr).nonzero(as_tuple=True)[0]
                # row: x0, y0, x1, y1, conf, row_idx
                res = torch.stack((pr_x0[r], pr_y0[r], pr_x1[r], pr_y1[r], pr_crowd[r, 4], r.float()), 1)  # n_pred3 x 7
                if res.shape[0] > 0:
                    if res.shape[0] > self.max_nms:
                        res = res[res[:, 4].argsort(descending=True)[:self.max_nms]]  # n_pred4 x 6
                    xyxy = res[:, :4]  # n_pred4 x 4
                    conf = res[:, 4]  # n_pred4
                    nms_indices = torchvision.ops.nms(xyxy, conf, iou_thr)[:2]  # n_pred5
                    if len(nms_indices) > 0:
                        pred_crowd1_best, crowd1_row_idx = res[nms_indices[0]][[4, 5]]
                        crowd1_row_idx = crowd1_row_idx.long()
                        if len(nms_indices) > 1:
                            pred_crowd2_best, crowd2_row_idx = res[nms_indices[1]][[4, 5]]
                            crowd2_row_idx = crowd2_row_idx.long()
                            cls_best_2 = torch.min(pred_crowd1_best, pred_crowd2_best)
                        else:
                            pred_crowd2_best, crowd2_row_idx = None, None
                            cls_best_2 = pred_crowd1_best
                        if cls_best_2 > cls_best:
                            cls_pair = 'crowd-crowd'
                            cls_best = cls_best_2

            # single
            pr_single = pred[CAT_ID_FIGHTER_SINGLE]
            if pr_single.shape[0] > 0:
                pred_single_best, single_row_idx = pr_single[:, 4].max(dim=0)
                if pred_single_best > cls_best:
                    cls_pair = 'single'
                    cls_best = pred_single_best

            if cls_best <= conf_thr or cls_pair == 'none':
                continue

            cls_ids = []
            row_indices = []
            if cls_pair == 'left-right':
                cls_ids = [CAT_ID_FIGHTER_LEFT, CAT_ID_FIGHTER_RIGHT]
                row_indices = [left_row_idx, right_row_idx]
            elif cls_pair == 'near-far':
                cls_ids = [CAT_ID_FIGHTER_NEAR, CAT_ID_FIGHTER_FAR]
                row_indices = [near_row_idx, far_row_idx]
            elif cls_pair == 'up-down':
                cls_ids = [CAT_ID_FIGHTER_UP, CAT_ID_FIGHTER_DOWN]
                row_indices = [up_row_idx, down_row_idx]
            elif cls_pair == 'crowd-crowd':
                if crowd2_row_idx is not None:
                    cls_ids = [CAT_ID_FIGHTER_CROWD, CAT_ID_FIGHTER_CROWD]
                    row_indices = [crowd1_row_idx, crowd2_row_idx]
                else:
                    cls_ids = [CAT_ID_FIGHTER_CROWD]
                    row_indices = [crowd1_row_idx]
            elif cls_pair == 'single':
                cls_ids = [CAT_ID_FIGHTER_SINGLE]
                row_indices = [single_row_idx]
            else:
                raise ValueError(f'cls_pair = {cls_pair} should not happen here')

            filter_preds[i] = torch.zeros((len(cls_ids), 6), device=device, dtype=dtype)
            for j, c in enumerate(cls_ids):
                row_idx = row_indices[j]
                filter_preds[i][j, 0] = pred_x0[c][row_idx]
                filter_preds[i][j, 1] = pred_y0[c][row_idx]
                filter_preds[i][j, 2] = pred_x1[c][row_idx]
                filter_preds[i][j, 3] = pred_y1[c][row_idx]
                filter_preds[i][j, 4] = pred[c][row_idx, 4]
                filter_preds[i][j, 5] = c

        return filter_preds

    def get_direct_results(self, preds, conf_thr=None, iou_thr=None):
        """ preds (torch.Tensor): bs x n_all_preds x out_dims, (out_dims: n_classes*bbox_dims)

            Returned results correspond to input images
        """
        conf_thr = conf_thr or self.conf_thr
        iou_thr = iou_thr or self.iou_thr
        preds = self.filter_predictions(preds, conf_thr, iou_thr)  # preds[i]: n x 6 (row: x0, y0, x1, y1, conf, cls)
        img_idx, xyxy_results, cls_results, conf_results, obj_conf_results, cls_conf_results = [], [], [], [], [], []
        for i, pred in enumerate(preds):
            if len(pred) > 0:
                pred = pred.detach().cpu().numpy()
                img_idx.append(i)
                xyxy_results.append(pred[:, :4])
                cls_results.append(pred[:, 5].astype(int))
                conf_results.append(pred[:, 4])
                obj_conf_results.append(pred[:, 4])
                cls_conf_results.append(1)
            else:
                img_idx.append(i)
                xyxy_results.append(np.empty((0, 4)))
                cls_results.append(np.empty((0,)).astype(int))
                conf_results.append(np.empty((0,)))
                obj_conf_results.append(np.empty((0,)))
                cls_conf_results.append(np.empty((0,)))
        return {
            'img_idx': img_idx,
            'xyxy_results': xyxy_results,
            'cls_results': cls_results,
            'conf_results': conf_results,
            'obj_conf_results': obj_conf_results,
            'cls_conf_results': cls_conf_results
        }

    def get_final_results(self, preds, batch_dict, conf_thr=None, iou_thr=None):
        """ preds (torch.Tensor): bs x n_all_preds x out_dims

            Returned results correspond to original images
        """
        conf_thr = conf_thr or self.conf_thr
        iou_thr = iou_thr or self.iou_thr
        preds = self.filter_predictions(preds, conf_thr, iou_thr)  # preds[i]: n x 6 (row: x0, y0, x1, y1, conf, cls)
        img_ids, xyxy_results, cls_results, conf_results, obj_conf_results, cls_conf_results = [], [], [], [], [], []
        for i, pred in enumerate(preds):
            if len(pred) > 0:
                pred = pred.detach().cpu().numpy()
                ratio, (border_top, _, border_left, _), (img_w, img_h) = batch_dict['_revert_params'][i]
                img_ids.append(batch_dict['img_id'][i])
                xyxy_results.append(self.process.revert_preds(pred[:, :4], ratio, border_left, border_top, img_w, img_h))
                cls_results.append(pred[:, 5])
                conf_results.append(pred[:, 4])
                obj_conf_results.append(pred[:, 4])
                cls_conf_results.append(1)
            else:
                img_ids.append(batch_dict['img_id'][i])
                xyxy_results.append(np.empty((0, 4)))
                cls_results.append(np.empty((0,)).astype(int))
                conf_results.append(np.empty((0,)))
                obj_conf_results.append(np.empty((0,)))
                cls_conf_results.append(np.empty((0,)))
        return {
            'img_ids': img_ids,
            'xyxy_results': xyxy_results,
            'cls_results': cls_results,
            'conf_results': conf_results,
            'obj_conf_results': obj_conf_results,
            'cls_conf_results': cls_conf_results
        }


@DETECT_POSTPROCESSOR.register_class('predict_myzone_v4')
class DetectPredictMyZoneV4(DetectPredictMyZoneV2):
    bbox_dims = 5

    def predict_box_obj_cls(self, output):
        """ output: list(bs x n_anchors x out_h x out_w x (n_classes*(bbox_dims+1)) (bbox_dims: cx,cy,w,h,obj)
        """
        assert self.anchor_rescaled is True

        preds = []

        for i, out in enumerate(output):
            bs, n_anc, h, w, d = out.shape
            n_cls = int(d / (self.bbox_dims + 1))
            if self.ij_grids[i] is None or tuple(self.ij_grids[i].shape[2:4]) != (h, w):
                self.ij_grids[i], self.wh_scale_grids[i] = self.make_grid(w, h, n_cls, self.anchors[i], self.strides[i])

            pred = out.sigmoid()
            pred = pred.view(bs, n_anc, h, w, n_cls, self.bbox_dims + 1)  # bs x n_anc x h x w x n_cls x 6
            cxy, wh = pred[..., 0:2], pred[..., 2:4]  # bs x n_anc x h x w x n_cls x 2
            cxy = (cxy * (2 + self.cxy_margin * 2) - (0.5 + self.cxy_margin) + self.ij_grids[i]) * self.strides[i]  # (0,1) => ((-0.5,1.5) + offset) * stride
            # wh = (wh * 2)**2 * self.wh_scale_grids[i]  # (0,1) => (0,2**2) * (anchors * stride)
            wh = (wh**self.wh_gamma) * (self.anchor_thr + self.wh_margin) * self.wh_scale_grids[i]  # (0,1) => (0,3) * (anchors * stride)
            pred = torch.cat((cxy, wh, pred[..., 4:]), -1).view(bs, -1, d)
            preds.append(pred)
        return preds  # list(bs x (n_anchors*out_h*out_w) x (n_classes*(bbox_dims+1)))

    def make_grid(self, w, h, n_cls, anchors, stride):
        """ anchors: n_anchors x 2
            w, h, n_cls, stride: scalar
        """
        device = anchors.device
        j_grid, i_grid = torch.meshgrid(torch.arange(h).to(device), torch.arange(w).to(device))
        ij_grid = torch.stack((i_grid, j_grid), dim=2).view(1, 1, h, w, 1, 2).expand((1, self.n_anchors, h, w, n_cls, 2)).float()
        wh_scale_grid = (anchors.clone() * stride).view(1, self.n_anchors, 1, 1, 1, 2).expand((1, self.n_anchors, h, w, n_cls, 2)).float()
        return ij_grid, wh_scale_grid  # 1 x n_anc x h x w x n_cls x 2, last_dim: (i, j) and (w, h)

    def filter_predictions(self, preds, conf_thr, iou_thr):
        """ preds: bs x n_all_preds x out_dims, (out_dims: n_classes*(bbox_dims+1))
        """
        device = preds.device
        dtype = preds.dtype
        bs, n, d = preds.shape
        n_cls = int(d / (self.bbox_dims + 1))
        preds = list(preds.view(bs, n, n_cls, self.bbox_dims + 1).split(1, dim=2))  # [bs x n_all_preds x 1 x (bbox_dims+1), ...]

        filtered = []
        for c in range(n_cls):
            preds[c] = preds[c].squeeze(2)  # bs x n_all_preds x (bbox_dims+1)
            if c != CAT_ID_FIGHTER_NONE:
                filtered.append((preds[c][..., 2:4] > self.min_wh).all(-1) &  # not too small
                                (preds[c][..., 2:4] < self.max_wh).all(-1) &  # not too big
                                (preds[c][..., 4] * preds[c][..., 5] > conf_thr))                # objectness * cls_conf
            else:
                filtered.append(preds[c][..., 5] > conf_thr)

        # row: x0, y0, x1, y1, conf, cls, obj_conf, cls_conf
        filter_preds = [torch.zeros((0, 8), device=device, dtype=dtype)] * bs

        for i in range(bs):
            pred, pred_x0, pred_y0, pred_x1, pred_y1 = [], [], [], [], []
            for c in range(n_cls):
                pr = preds[c][i]     # n_all_preds x (bbox_dims+1)
                fi = filtered[c][i]  # n_all_preds
                pr = pr[fi]          # n_pred2 x (bbox_dims+1)
                pred.append(pr)

                if pr.shape[0] > 0:
                    pr_x0, pr_y0, pr_x1, pr_y1 = self.process.parse_cxywh(pr[:, :4])[4:]
                else:
                    pr_x0, pr_y0, pr_x1, pr_y1 = None, None, None, None

                pred_x0.append(pr_x0)  # n_pred2
                pred_y0.append(pr_y0)  # n_pred2
                pred_x1.append(pr_x1)  # n_pred2
                pred_y1.append(pr_y1)  # n_pred2

            cls_pair = 'none'
            cls_best = conf_thr

            # none
            pr_none = pred[CAT_ID_FIGHTER_NONE]
            if pr_none.shape[0] > 0:
                pred_none_best, _ = pr_none[:, 5].max(dim=0)
                if pred_none_best > cls_best:
                    cls_pair = 'none'
                    cls_best = pred_none_best

            # left-right
            pr_left, pr_right = pred[CAT_ID_FIGHTER_LEFT], pred[CAT_ID_FIGHTER_RIGHT]
            if pr_left.shape[0] > 0 and pr_right.shape[0] > 0:
                pred_left_best, left_row_idx = (pr_left[:, 4] * pr_left[:, 5]).max(dim=0)
                pred_right_best, right_row_idx = (pr_right[:, 4] * pr_right[:, 5]).max(dim=0)
                cls_best_2 = torch.min(pred_left_best, pred_right_best)
                if cls_best_2 > cls_best:
                    cls_pair = 'left-right'
                    cls_best = cls_best_2

            # near-far
            pr_near, pr_far = pred[CAT_ID_FIGHTER_NEAR], pred[CAT_ID_FIGHTER_FAR]
            if pr_near.shape[0] > 0 and pr_far.shape[0] > 0:
                pred_near_best, near_row_idx = (pr_near[:, 4] * pr_near[:, 5]).max(dim=0)
                pred_far_best, far_row_idx = (pr_far[:, 4] * pr_far[:, 5]).max(dim=0)
                cls_best_2 = torch.min(pred_near_best, pred_far_best)
                if cls_best_2 > cls_best:
                    cls_pair = 'near-far'
                    cls_best = cls_best_2

            # up-down
            pr_up, pr_down = pred[CAT_ID_FIGHTER_UP], pred[CAT_ID_FIGHTER_DOWN]
            if pr_up.shape[0] > 0 and pr_down.shape[0] > 0:
                pred_up_best, up_row_idx = (pr_up[:, 4] * pr_up[:, 5]).max(dim=0)
                pred_down_best, down_row_idx = (pr_down[:, 4] * pr_down[:, 5]).max(dim=0)
                cls_best_2 = torch.min(pred_up_best, pred_down_best)
                if cls_best_2 > cls_best:
                    cls_pair = 'up-down'
                    cls_best = cls_best_2

            # crowd-crowd
            pr_crowd = pred[CAT_ID_FIGHTER_CROWD]
            pr_x0 = pred_x0[CAT_ID_FIGHTER_CROWD]
            pr_y0 = pred_y0[CAT_ID_FIGHTER_CROWD]
            pr_x1 = pred_x1[CAT_ID_FIGHTER_CROWD]
            pr_y1 = pred_y1[CAT_ID_FIGHTER_CROWD]
            if pr_crowd.shape[0] > 1:
                r = (pr_crowd[:, 4] > conf_thr).nonzero(as_tuple=True)[0]
                # row: x0, y0, x1, y1, conf, row_idx
                res = torch.stack((pr_x0[r], pr_y0[r], pr_x1[r], pr_y1[r], pr_crowd[r, 4] * pr_crowd[r, 5], r.float()), 1)  # n_pred3 x 7
                if res.shape[0] > 0:
                    if res.shape[0] > self.max_nms:
                        res = res[res[:, 4].argsort(descending=True)[:self.max_nms]]  # n_pred4 x 6
                    xyxy = res[:, :4]  # n_pred4 x 4
                    conf = res[:, 4]  # n_pred4
                    nms_indices = torchvision.ops.nms(xyxy, conf, iou_thr)[:2]  # n_pred5
                    if len(nms_indices) > 0:
                        pred_crowd1_best, crowd1_row_idx = res[nms_indices[0]][[4, 5]]
                        crowd1_row_idx = crowd1_row_idx.long()
                        if len(nms_indices) > 1:
                            pred_crowd2_best, crowd2_row_idx = res[nms_indices[1]][[4, 5]]
                            crowd2_row_idx = crowd2_row_idx.long()
                            cls_best_2 = torch.min(pred_crowd1_best, pred_crowd2_best)
                        else:
                            pred_crowd2_best, crowd2_row_idx = None, None
                            cls_best_2 = pred_crowd1_best
                        if cls_best_2 > cls_best:
                            cls_pair = 'crowd-crowd'
                            cls_best = cls_best_2

            # single
            pr_single = pred[CAT_ID_FIGHTER_SINGLE]
            if pr_single.shape[0] > 0:
                pred_single_best, single_row_idx = (pr_single[:, 4] * pr_single[:, 5]).max(dim=0)
                if pred_single_best > cls_best:
                    cls_pair = 'single'
                    cls_best = pred_single_best

            if cls_best <= conf_thr or cls_pair == 'none':
                continue

            cls_ids = []
            row_indices = []
            if cls_pair == 'left-right':
                cls_ids = [CAT_ID_FIGHTER_LEFT, CAT_ID_FIGHTER_RIGHT]
                row_indices = [left_row_idx, right_row_idx]
            elif cls_pair == 'near-far':
                cls_ids = [CAT_ID_FIGHTER_NEAR, CAT_ID_FIGHTER_FAR]
                row_indices = [near_row_idx, far_row_idx]
            elif cls_pair == 'up-down':
                cls_ids = [CAT_ID_FIGHTER_UP, CAT_ID_FIGHTER_DOWN]
                row_indices = [up_row_idx, down_row_idx]
            elif cls_pair == 'crowd-crowd':
                if crowd2_row_idx is not None:
                    cls_ids = [CAT_ID_FIGHTER_CROWD, CAT_ID_FIGHTER_CROWD]
                    row_indices = [crowd1_row_idx, crowd2_row_idx]
                else:
                    cls_ids = [CAT_ID_FIGHTER_CROWD]
                    row_indices = [crowd1_row_idx]
            elif cls_pair == 'single':
                cls_ids = [CAT_ID_FIGHTER_SINGLE]
                row_indices = [single_row_idx]
            else:
                raise ValueError(f'cls_pair = {cls_pair} should not happen here')

            filter_preds[i] = torch.zeros((len(cls_ids), 8), device=device, dtype=dtype)
            for j, c in enumerate(cls_ids):
                row_idx = row_indices[j]
                filter_preds[i][j, 0] = pred_x0[c][row_idx]
                filter_preds[i][j, 1] = pred_y0[c][row_idx]
                filter_preds[i][j, 2] = pred_x1[c][row_idx]
                filter_preds[i][j, 3] = pred_y1[c][row_idx]
                filter_preds[i][j, 4] = pred[c][row_idx, 4] * pred[c][row_idx, 5] if c != CAT_ID_FIGHTER_NONE else pred[c][row_idx, 5]
                filter_preds[i][j, 5] = c
                filter_preds[i][j, 6] = pred[c][row_idx, 4] if c != CAT_ID_FIGHTER_NONE else 0.
                filter_preds[i][j, 7] = pred[c][row_idx, 5]

        return filter_preds

    def get_direct_results(self, preds, conf_thr=None, iou_thr=None):
        """ preds (torch.Tensor): bs x n_all_preds x out_dims, (out_dims: n_classes*bbox_dims)

            Returned results correspond to input images
        """
        conf_thr = conf_thr or self.conf_thr
        iou_thr = iou_thr or self.iou_thr
        preds = self.filter_predictions(preds, conf_thr, iou_thr)  # preds[i]: n x 8 (row: x0, y0, x1, y1, conf, cls, obj_conf, cls_conf)
        img_idx, xyxy_results, cls_results, conf_results, obj_conf_results, cls_conf_results = [], [], [], [], [], []
        for i, pred in enumerate(preds):
            if len(pred) > 0:
                pred = pred.detach().cpu().numpy()
                img_idx.append(i)
                xyxy_results.append(pred[:, :4])
                cls_results.append(pred[:, 5].astype(int))
                conf_results.append(pred[:, 4])
                obj_conf_results.append(pred[:, 6])
                cls_conf_results.append(pred[:, 7])
            else:
                img_idx.append(i)
                xyxy_results.append(np.empty((0, 4)))
                cls_results.append(np.empty((0,)).astype(int))
                conf_results.append(np.empty((0,)))
                obj_conf_results.append(np.empty((0,)))
                cls_conf_results.append(np.empty((0,)))
        return {
            'img_idx': img_idx,
            'xyxy_results': xyxy_results,
            'cls_results': cls_results,
            'conf_results': conf_results,
            'obj_conf_results': obj_conf_results,
            'cls_conf_results': cls_conf_results
        }

    def get_final_results(self, preds, batch_dict, conf_thr=None, iou_thr=None):
        """ preds (torch.Tensor): bs x n_all_preds x out_dims

            Returned results correspond to original images
        """
        conf_thr = conf_thr or self.conf_thr
        iou_thr = iou_thr or self.iou_thr
        preds = self.filter_predictions(preds, conf_thr, iou_thr)  # preds[i]: n x 6 (row: x0, y0, x1, y1, conf, cls)
        img_ids, xyxy_results, cls_results, conf_results, obj_conf_results, cls_conf_results = [], [], [], [], [], []
        for i, pred in enumerate(preds):
            if len(pred) > 0:
                pred = pred.detach().cpu().numpy()
                ratio, (border_top, _, border_left, _), (img_w, img_h) = batch_dict['_revert_params'][i]
                img_ids.append(batch_dict['img_id'][i])
                xyxy_results.append(self.process.revert_preds(pred[:, :4], ratio, border_left, border_top, img_w, img_h))
                cls_results.append(pred[:, 5].astype(int))
                conf_results.append(pred[:, 4])
                obj_conf_results.append(pred[:, 6])
                cls_conf_results.append(pred[:, 7])
            else:
                img_ids.append(batch_dict['img_id'][i])
                xyxy_results.append(np.empty((0, 4)))
                cls_results.append(np.empty((0,)).astype(int))
                conf_results.append(np.empty((0,)))
                obj_conf_results.append(np.empty((0,)))
                cls_conf_results.append(np.empty((0,)))
        return {
            'img_ids': img_ids,
            'xyxy_results': xyxy_results,
            'cls_results': cls_results,
            'conf_results': conf_results,
            'obj_conf_results': obj_conf_results,
            'cls_conf_results': cls_conf_results
        }
