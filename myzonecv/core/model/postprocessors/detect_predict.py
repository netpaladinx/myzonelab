import torch
import torchvision

from ...registry import DETECT_POSTPROCESSOR
from .base_process import BaseProcess
from .detect_process import DetectProcess


@DETECT_POSTPROCESSOR.register_class('predict')
class DetectPredict(BaseProcess):
    def __init__(self,
                 anchors,
                 default='predict_box_obj_cls',
                 conf_thr=0.3,
                 iou_thr=0.7,
                 min_wh=2,
                 max_wh=4096,
                 max_det=2,    # max bboxes to return
                 max_nms=30000,  # max bboxes to call nms
                 require_redundant=True,  # used when merge_bboxes is True
                 merge_bboxes=False):
        """ Note: lower conf_thr (e.g. 0.01), higher iou_thr (e.g. 0.9), larger max_det (e.g. 300) can give higher AP 
        """
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
