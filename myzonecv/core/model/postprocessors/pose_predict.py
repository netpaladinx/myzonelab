import numpy as np
import cv2

from ...registry import POSE_POSTPROCESSOR
from ...consts import BBOX_SCALE_UNIT
from ...utils import npf
from .base_process import BaseProcess
from .pose_process import PoseProcess


@POSE_POSTPROCESSOR.register_class('simple_predict')
class PoseSimplePredict(BaseProcess):
    def __init__(self, default='predict_by_max'):
        super().__init__(default)

    def predict_by_max(self, heatmaps):
        assert isinstance(heatmaps, np.ndarray) and heatmaps.ndim == 4
        batch_size, n_kpts, _, width = heatmaps.shape
        heatmaps_reshaped = heatmaps.reshape((batch_size, n_kpts, -1))
        indices = np.argmax(heatmaps_reshaped, 2).reshape((batch_size, n_kpts, 1))
        maxvals = np.amax(heatmaps_reshaped, 2).reshape((batch_size, n_kpts, 1))

        preds = np.tile(indices, (1, 1, 2)).astype(np.float32)
        preds[:, :, 0] = preds[:, :, 0] % width  # x
        preds[:, :, 1] = preds[:, :, 1] // width  # y
        preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)  # filter negative

        return preds, maxvals  # preds: bs x kpts x 2, maxvals: bs x kpts x 1


@POSE_POSTPROCESSOR.register_class('predict')
class PosePredict(PoseSimplePredict):
    def __init__(self, decode=None, default='predict_with_batch_dict'):
        super().__init__(default)
        self.decode = decode
        self.pred_process = PoseProcess(default='revert_preds')

    def predict_with_warp_inverse_matrix(self, heatmaps, input_size, warp_mat_inv, decode='default', kernel=11):
        """ heatmaps: bs x kpts x heatmap_h x heatmap_w
            input_size: (input_w, input_h)
            warp_mat_inv: bs x 2 x 3
        """
        decode = self.decode or decode
        heatmap_h, heatmap_w = heatmaps.shape[2:]
        input_w, input_h = input_size
        stride = (input_w / heatmap_w, input_h / heatmap_h)
        preds, maxvals = self.predict_via_decode(heatmaps, decode, kernel)  # preds: bs x kpts x 2, maxvals: bs x kpts x 1
        preds = preds * npf(stride)

        preds_ones = np.concatenate((preds, np.ones(preds.shape[:2] + (1,), dtype=preds.dtype)), 2)  # bs x kpts x 3
        preds = preds_ones @ warp_mat_inv.transpose((0, 2, 1))  # bs x kpts x 2

        return preds, maxvals

    def predict_with_batch_dict(self, heatmaps, batch_dict, decode='default', kernel=11):
        decode = self.decode or decode
        preds, maxvals = self.predict_via_decode(heatmaps, decode, kernel)

        height, width = heatmaps.shape[2:]
        batch_size = len(batch_dict['center'])
        n_anns = len(batch_dict['ann_id'])
        assert batch_size >= n_anns and batch_size % n_anns == 0
        multiple = batch_size // n_anns
        centers = np.zeros((batch_size, 2))
        scales = np.zeros((batch_size, 2))
        bbox_scores = np.zeros(batch_size)
        ann_ids = []
        for i in range(batch_size):
            centers[i, :] = batch_dict['center'][i]
            scales[i, :] = batch_dict['scale'][i]

            preds[i] = self.pred_process(preds[i].copy(), centers[i], scales[i], (width, height), dark_udp=(decode == 'dark_udp'))

            j = i // multiple
            bbox_scores[i] = batch_dict['bbox_score'][j] if 'bbox_score' in batch_dict else 1
            ann_ids.append(batch_dict['ann_id'][j] if 'ann_id' in batch_dict else -1)

        n_kpts = preds.shape[1]
        kpts_results = np.zeros((batch_size, n_kpts, 3))
        bbox_results = np.zeros((batch_size, 5))
        kpts_results[:, :, 0:2] = preds
        kpts_results[:, :, 2:3] = maxvals
        bbox_results[:, 0:2] = centers - scales * BBOX_SCALE_UNIT / 2.
        bbox_results[:, 2:4] = scales * BBOX_SCALE_UNIT
        bbox_results[:, 4] = bbox_scores

        return {'kpts_results': kpts_results, 'bbox_results': bbox_results, 'ann_ids': ann_ids, 'multiple': multiple}

    def predict_via_decode(self, heatmaps, decode='default', kernel=11):
        """ heatmaps: bs x kpts x heatmap_h x heatmap_w
            decode: 'default', 'unbiased', 'megvii', or 'dark_udp'
            kernel: Gaussian kernel size for modulation (kernel=17 for sigma=3 and kernel=11 for sigma=2)
        """
        assert decode in ('default', 'unbiased', 'megvii', 'dark_udp')
        heatmaps = heatmaps.copy()

        if decode == 'default':
            preds, maxvals = self.default_decode(heatmaps)
        elif decode == 'unbiased':
            preds, maxvals = self.unbiased_decode(heatmaps, kernel)
        elif decode == 'megvii':
            preds, maxvals = self.megvii_decode(heatmaps, kernel)
        elif decode == 'dark_udp':
            preds, maxvals = self.dark_udp_decode(heatmaps, kernel)

        return preds, maxvals  # preds: bs x kpts x 2, maxvals: bs x kpts x 1

    def default_decode(self, heatmaps):
        """ adding +/-0.25 shift """
        preds, maxvals = self.predict_by_max(heatmaps)  # preds: bs x kpts x 2
        batch_size, n_kpts, height, width = heatmaps.shape
        for i in range(batch_size):
            for j in range(n_kpts):
                heatmap = heatmaps[i][j]
                x, y = preds[i][j]
                x, y = int(x), int(y)
                if 0 < x < width - 1 and 0 < y < height - 1:
                    diff = npf([heatmap[y][x + 1] - heatmap[y][x - 1], heatmap[y + 1][x] - heatmap[y - 1][x]])
                    preds[i][j] += np.sign(diff) * .25
        return preds, maxvals

    def unbiased_decode(self, heatmaps, kernel):
        """ applying Gaussian distribute modulation """
        preds, maxvals = self.predict_by_max(heatmaps)  # preds: bs x kpts x 2
        batch_size, n_kpts, height, width = heatmaps.shape
        heatmaps = np.log(np.maximum(self.padded_gaussian_blur(heatmaps, kernel), 1e-10))
        for i in range(batch_size):
            for j in range(n_kpts):
                preds[i][j] = self.taylor(heatmaps[i][j], preds[i][j])
        return preds, maxvals

    def megvii_decode(self, heatmaps, kernel):
        heatmaps = self.padded_gaussian_blur(heatmaps, kernel=kernel)
        preds, maxvals = self.default_decode(heatmaps)
        batch_size, n_kpts, height, width = heatmaps.shape
        for i in range(batch_size):
            for j in range(n_kpts):
                x, y = preds[i][j]
                if 0 < x < width - 1 and 0 < y < height - 1:
                    preds[i][j] += 0.5
        maxvals = maxvals / 255.0 + 0.5
        return preds, maxvals

    def dark_udp_decode(self, heatmaps, kernel=3):
        """ using Newton's method """
        preds, maxvals = self.predict_by_max(heatmaps)  # preds: bs x kpts x 2
        batch_size, n_kpts, height, width = heatmaps.shape
        heatmaps = self.simple_gaussian_blur(heatmaps, kernel)
        heatmaps = np.transpose(np.log(np.clip(heatmaps, 0.001, 50)), (2, 3, 0, 1)).reshape(height, width, -1)  # height x width x (bs*kpts)
        heatmaps_pad = cv2.copyMakeBorder(heatmaps, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)
        heatmaps_pad = np.transpose(heatmaps_pad.reshape(height + 2, width + 2, batch_size, n_kpts), (2, 3, 0, 1)).flatten()  # bs*kpts*(h+2)*(w+2)

        indices = preds[..., 0] + 1 + (preds[..., 1] + 1) * (width + 2)  # x + 1 + (y + 1) * (width + 2), size: bs x kpts
        indices += (width + 2) * (height + 2) * np.arange(batch_size * n_kpts).reshape(-1, n_kpts)  # bs x kpts
        indices = indices.astype(int).reshape(-1, 1)  # (bs*kpts) x 1
        v_tl = heatmaps_pad[indices - width - 3]
        v_tc = heatmaps_pad[indices - width - 2]
        v_tr = heatmaps_pad[indices - width - 1]
        v_cl = heatmaps_pad[indices - 1]
        v_cc = heatmaps_pad[indices]
        v_cr = heatmaps_pad[indices + 1]
        v_bl = heatmaps_pad[indices + width + 1]
        v_bc = heatmaps_pad[indices + width + 2]
        v_br = heatmaps_pad[indices + width + 3]
        dx = 0.5 * (v_cr - v_cl)
        dy = 0.5 * (v_bc - v_tc)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(batch_size, n_kpts, 2, 1)  # bs x kpts x 2 x 1
        dxx = (v_cr - v_cc) - (v_cc - v_cl)
        dyy = (v_bc - v_cc) - (v_cc - v_tc)
        dxy = ((v_cc - v_cl) - (v_tc - v_tl) + (v_br - v_bc) - (v_cr - v_cc)) * 0.5
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(batch_size, n_kpts, 2, 2)  # bs x kpts x 2 x 2
        hessian_inv = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        preds -= np.einsum('ijmn,ijnk->ijmk', hessian_inv, derivative).squeeze()  # bs x kpts x 2
        return preds, maxvals

    @staticmethod
    def simple_gaussian_blur(self, heatmaps, kernel):
        batch_size, n_kpts, height, width = heatmaps.shape
        for i in range(batch_size):
            for j in range(n_kpts):
                heatmaps[i, j] = cv2.GaussianBlur(heatmaps[i, j], (kernel, kernel), 0)
        return heatmaps

    @staticmethod
    def padded_gaussian_blur(self, heatmaps, kernel=11):
        """ 
        Relation between Gaussian kernel size and heatmap guassian sigma when training:
            sigma = 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8

            - if kernel = 17, sigma ~= 3
            - if kernel = 11, sigma = 2
            - if kernel = 7, sigma ~= 1.5
            - if kernel = 3, sigma ~= 1
        """
        assert kernel % 2 == 1
        border = (kernel - 1) // 2
        batch_size, n_kpts, height, width = heatmaps.shape
        for i in range(batch_size):
            for j in range(n_kpts):
                maxval = np.max(heatmaps[i, j])
                hm = np.zeros((height + 2 * border, width + 2 * border))
                hm[border:-border, border:-border] = heatmaps[i, j].copy()
                hm = cv2.GaussianBlur(hm, (kernel, kernel), 0)
                heatmaps[i, j] = hm[border:-border, border:-border].copy()
                heatmaps[i, j] *= maxval / np.max(heatmaps[i, j])
        return heatmaps

    @staticmethod
    def taylor(heatmap, pred):
        """ Distribution aware coordinate decoding method """
        height, width = heatmap.shape
        x, y = int(pred[0]), int(pred[1])
        if 1 < x < width - 2 and 1 < y < height - 2:
            dx = 0.5 * (heatmap[y][x + 1] - heatmap[y][x - 1])
            dy = 0.5 * (heatmap[y + 1][x] - heatmap[y - 1][x])
            dxx = 0.25 * (heatmap[y][x + 2] - 2 * heatmap[y][x] + heatmap[y][x - 2])
            dxy = 0.25 * (heatmap[y + 1][x + 1] - heatmap[y - 1][x + 1] - heatmap[y + 1][x - 1] + heatmap[y - 1][x - 1])
            dyy = 0.25 * (heatmap[y + 2][x] - 2 * heatmap[y][x] + heatmap[y - 2][x])
            derivative = npf([[dx], [dy]])
            hessian = npf([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy**2 != 0:
                hessian_inv = np.linalg.inv(hessian)
                offset = -hessian_inv @ derivative
                offset = np.squeeze(npf(offset.T), axis=0)
                pred += offset
        return pred
