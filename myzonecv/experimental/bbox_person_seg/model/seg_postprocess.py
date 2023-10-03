import os.path as osp

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from myzonecv.core.model import BaseProcess
from myzonecv.core.utils import to_numpy, get_affine_matrix, apply_warp_to_map2d, dump_numpy, list2str
from myzonecv.core.data.datautils import int_bbox
from myzonecv.core.data.dataconsts import BBOX_SCALE_UNIT
from ..registry import SEG_POSTPROCESSORS


@SEG_POSTPROCESSORS.register_class('predict')
class SegPredict(BaseProcess):
    def __init__(self, default='predict_with_batch_dict', align_corners=None):
        super().__init__(default)
        self.align_corners = align_corners

    def predict_with_batch_dict(self, output, batch_dict, pred_cache_dir=None):
        """ output (logits, torch.Tensor): N x H x W or N x 1 x H x W
            batch_dict (dict):
                'orig_size': list([orig_w, orig_h])
                'input_size': list([input_w, input_h])
                'output_size': list([output_w, output_h])
                'orig_bbox': list([x0, y0, w, h])
                'center': list([cx, cy])
                'scale': list([s_x, s_y])
                'flipped' (optional): list(bool)
        """
        assert osp.isdir(pred_cache_dir)

        if output.ndim == 4:
            assert output.shape[1] == 1
            output = output.squeeze(1)

        ann_ids = batch_dict['ann_id']
        orig_size = batch_dict['orig_size']
        input_size = batch_dict['input_size']
        output_size = batch_dict['output_size']
        orig_bbox = batch_dict['orig_bbox']
        center = batch_dict['center']
        scale = batch_dict['scale']
        flipped = batch_dict.get('flipped')

        with torch.no_grad():
            output = output.sigmoid()
            conf_scores = []
            seg_masks = []
            bbox_seg_masks = []

            for i, out in enumerate(output):  # out: H x W
                pos_out = out[out > 0.5]
                conf_scores.append(to_numpy(pos_out.mean()) if len(pos_out) > 0 else 0.)

                # output size => input size
                in_size, out_size = input_size[i], output_size[i]
                stride = (in_size[0] / out_size[0], in_size[1] / out_size[1])
                out = F.interpolate(out[None, None], scale_factor=(stride[1], stride[0]), mode='bilinear', align_corners=self.align_corners)[0, 0]
                out = to_numpy(out)

                # input size => original image size
                _, revert_mat = get_affine_matrix(center[i], scale[i], in_size, scale_unit=BBOX_SCALE_UNIT, return_inv=True)
                out = apply_warp_to_map2d(out, revert_mat, orig_size[i], flags=cv2.INTER_AREA, border_value=0)

                if flipped is not None and flipped[i]:
                    out = np.fliplr(out)  # left-right flip

                mask = (out > 0.5).astype(np.float32)
                x0, y0, w, h = int_bbox(orig_bbox[i])
                bbox_mask = mask[y0:y0 + h, x0:x0 + w].copy()

                ann_id_str = list2str(ann_ids[i])
                mask_path = osp.join(pred_cache_dir, f'seg_mask_{ann_id_str}.npy')
                dump_numpy(mask, mask_path)
                bbox_mask_path = osp.join(pred_cache_dir, f'bbox_seg_mask_{ann_id_str}.npy')
                dump_numpy(bbox_mask, bbox_mask_path)

                seg_masks.append(mask_path)
                bbox_seg_masks.append(bbox_mask_path)

            return {'seg_mask_results': seg_masks,
                    'bbox_seg_mask_results': bbox_seg_masks,
                    'bbox_results': orig_bbox,
                    'conf_scores': conf_scores,
                    'ann_ids': ann_ids}
