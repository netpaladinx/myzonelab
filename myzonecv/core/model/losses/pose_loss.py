import numpy as np
import torch
import torch.nn as nn

from ...registry import POSE_LOSSES
from ..bricks import dual_affine, get_dual_affine_mask
from ..base_module import BaseModule
from ..postprocessors import PosePredict
from ...utils import fliplr_coord, apply_warp_to_coord, npf
from ...data.transforms.pose_target import GenerateHeatmaps


def weighted_heatmap_mse(criterion, output, target, weight=None):
    """ output: n x m x h x w
        target: n x m x h x w
        weight: n x m x 1 x 1
    """
    if weight is not None:
        return criterion(output * weight, target * weight)
    else:
        return criterion(output, target)


@POSE_LOSSES.register_class('mse')
class PoseMSE(BaseModule):
    """ Mean Squared Error (squared L2 norm) on heatmaps """

    def __init__(self, use_target_weights=False, loss_weight=1., loss_name='mse_loss', dist_aware=False, dist_gamma=0.5):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weights = use_target_weights
        self.loss_weight = loss_weight
        assert loss_name.endswith('_loss')
        self.loss_name = loss_name
        self.dist_aware = dist_aware
        self.dist_gamma = dist_gamma

    def forward(self, output, target, target_weights=None, dist_weights=None):
        """ output (torch.Tensor): bs x kpts x heatmap_h x heatmap_w
            target (torch.Tensor): bs x kpts x heatmap_h x heatmap_w
            target_weights: bs x kpts x 1 (or bs x kpts or bs x kpts x 1 x 1)
            dist_weights: bs x kpts
        """
        target_weights = target_weights if self.use_target_weights and target_weights is not None else None
        if target_weights.ndim == 2:
            target_weights = target_weights[..., None, None]
        elif target_weights.ndim == 3:
            target_weights = target_weights[..., None]

        if self.dist_aware and dist_weights is not None:
            dist_weights = torch.pow(dist_weights, self.dist_gamma)[..., None, None]
            target_weights *= dist_weights

        loss = weighted_heatmap_mse(self.criterion, output, target, target_weights)
        loss = loss * self.loss_weight
        return {self.loss_name: loss}


@POSE_LOSSES.register_class('stable_mse')
class PoseStableMSE(BaseModule):
    def __init__(self, loss_weight=1., loss_name='stable_loss', stop_dst_gradient=False, reg_weights=(1., 1.), warmup_steps=1000):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.loss_weight = loss_weight
        assert loss_name.endswith('_loss')
        self.loss_name = loss_name
        self.stop_dst_gradient = stop_dst_gradient
        self.reg_weights = reg_weights
        self.warmup_steps = warmup_steps
        self.cur_step = 0

    def forward(self, output, center, scale, flipped, rotate, params, flip_pairs=None, heatmap_reg=None):
        """ output (torch.Tensor): bs x kpts x heatmap_h x heatmap_w
            center (np.ndarray): bs x 2
            scale (np.ndarray): bs x 2
            flipped (np.ndarray): bs
            rotate (np.ndarray): bs
            params (np.ndarray): n_anns x 6
            flip_pairs (list): n_pairs x 2
        """
        bs, n_kpts, out_h, out_w = output.size()
        n_anns = len(params)
        dst_list, src_list, mask_list = [], [], []
        base = 0
        box_size = npf([out_w, out_h])
        for i in range(n_anns):
            m, img_w, img_h, in_w, in_h, scale_unit = params[i]
            assert in_w / out_w == in_h / out_h
            stride = in_w / out_w
            img_size = npf([img_w, img_h]) / stride

            for src_j in range(m):
                src_out = output[base + src_j]  # n_kpts x h x w
                for dst_j in range(m):
                    if src_j == dst_j:
                        continue
                    dst_out = output[base + dst_j]  # n_kpts x h x w

                    src_c = center[base + src_j] / stride
                    src_s = scale[base + src_j] / stride
                    src_f = flipped[base + src_j]
                    src_r = rotate[base + src_j]
                    dst_c = center[base + dst_j] / stride
                    dst_s = scale[base + dst_j] / stride
                    dst_f = flipped[base + dst_j]
                    dst_r = rotate[base + dst_j]

                    dst_from_src = dual_affine(src_out,
                                               translate1=src_c - img_size / 2,
                                               rotate1=-src_r,
                                               scale1=(src_s * scale_unit) / box_size,
                                               flip1=bool(src_f),
                                               flip_pairs1=flip_pairs,
                                               flip_first1=False,
                                               dst_size1=img_size,
                                               scale2=box_size / (dst_s * scale_unit),
                                               rotate2=dst_r,
                                               translate2=img_size / 2 - dst_c,
                                               flip2=bool(dst_f),
                                               flip_pairs2=flip_pairs,
                                               flip_first2=True,
                                               dst_size2=box_size,
                                               order_mode='SRTTRS')  # n_kpts x h x w
                    mask_from_src = get_dual_affine_mask(src_out,
                                                         translate1=src_c - img_size / 2,
                                                         rotate1=-src_r,
                                                         scale1=(src_s * scale_unit) / box_size,
                                                         flip1=bool(src_f),
                                                         flip_first1=False,
                                                         dst_size1=img_size,
                                                         scale2=box_size / (dst_s * scale_unit),
                                                         rotate2=dst_r,
                                                         translate2=img_size / 2 - dst_c,
                                                         flip2=bool(dst_f),
                                                         flip_first2=True,
                                                         dst_size2=box_size,
                                                         order_mode='SRTTRS')  # h x w

                    dst_list.append(dst_out)
                    src_list.append(dst_from_src)
                    mask_list.append(mask_from_src)

            base += m

        bs_eff = len(dst_list) * 2
        dst = torch.stack(dst_list)  # N x n_kpts x h x w, N = bs * (multiple-1)
        src = torch.stack(src_list)  # N x n_kpts x h x w
        mask = torch.stack(mask_list)[:, None]  # N x 1 x h x w

        if self.stop_dst_gradient:
            dst = dst.detach()
            bs_eff /= 2

        weight = self.loss_weight * min(self.cur_step / self.warmup_steps, 1.)
        mse_loss = weighted_heatmap_mse(self.criterion, src, dst, mask) * weight * bs / bs_eff
        res = {self.loss_name: mse_loss}

        if heatmap_reg is not None:
            out = output.view(bs, n_kpts, -1)
            out1 = out.mean(-1)               # bs x n_kpts
            out2 = (out**2).mean(-1).sqrt()  # bs x n_kpts
            reg1_loss = self.criterion(out1, torch.full_like(out1, heatmap_reg[0])) * weight * self.reg_weights[0]
            reg2_loss = self.criterion(out2, torch.full_like(out2, heatmap_reg[1])) * weight * self.reg_weights[1]
            res['reg1_loss'] = reg1_loss
            res['reg2_loss'] = reg2_loss

        self.cur_step += 1
        return res


@POSE_LOSSES.register_class('stable_v2_mse')
class PoseStableV2MSE(BaseModule):
    def __init__(self, loss_weight=1., loss_name='stable_v2_loss', warmup_steps=1000, predict_kwargs=None, heatmap_gen_kwargs=None):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.loss_weight = loss_weight
        assert loss_name.endswith('_loss')
        self.loss_name = loss_name
        self.warmup_steps = warmup_steps
        self.cur_step = 0

        self.predict_kwargs = predict_kwargs or {'decode': 'default', 'kernel': 11}
        self.predict = PosePredict(default='predict_with_warp_inverse_matrix')

        self.heatmap_gen_kwargs = heatmap_gen_kwargs or {}
        self.heatmap_gen = GenerateHeatmaps(**self.heatmap_gen_kwargs)

    def forward(self, output, flipped, warp_mat_inv, warp_mat, params, flip_pairs=None):
        """ output (torch.Tensor): bs x kpts x heatmap_h x heatmap_w
            flipped (np.ndarray): bs
            warp_mat_inv (np.ndarray): bs x 2 x 3
            warp_mat (np.ndarray): bs x 2 x 3
            params (np.ndarray): n_anns x 6, (multiple, img_w, img_h, input_w, input_h, scale_unit)
            flip_pairs (list): n_pairs x 2
        """
        # decode heatmap and warp coords back
        output_np = output.detach().cpu().numpy()
        bs, _, out_h, out_w = output_np.shape
        out_size = (out_w, out_h)
        in_size = params[0][3:5]  # (in_w, in_h)
        preds, maxvals = self.predict(output_np, in_size, warp_mat_inv, **self.predict_kwargs)  # preds: bs x kpts x 2, maxvals: bs x kpts x 1

        n_anns = len(params)
        base = 0
        for i in range(n_anns):
            m, img_w = params[i][:2]
            for j in range(m):
                index = base + j
                if flipped[index]:
                    preds[index] = fliplr_coord(preds[index].copy(), img_w, flip_pairs)  # flip backward

            base += m

        # re-warp coords to target heatmaps
        base = 0
        out_list, tar_list = [], []
        for i in range(n_anns):
            m, img_w = params[i][:2]

            for src_j in range(m):
                src_out = output[base + src_j]  # (torch.Tensor) n_kpts x h x w
                for dst_j in range(m):
                    if src_j == dst_j:
                        continue
                    dst_tar = preds[base + dst_j].copy()       # kpts x 2
                    dst_maxval = maxvals[base + dst_j].copy()  # kpts x 1

                    if flipped[base + src_j]:
                        dst_tar = fliplr_coord(dst_tar, img_w, flip_pairs)

                    dst_tar = apply_warp_to_coord(dst_tar, warp_mat[base + src_j])

                    out_list.append(src_out)  # list(n_kpts x h x w)
                    tar_list.append(np.concatenate((dst_tar, dst_maxval), 1))  # list(kpts x 3)

            base += m

        dst_target = np.stack(tar_list, 0)  # N x kpts x 3, N = bs * (multiple-1)
        target, target_weights = self.heatmap_gen.encode_kpts(dst_target, in_size, out_size, None)  # (np.ndarray) N x n_kpts x h x w, N x n_kpts x 1

        src_output = torch.stack(out_list)  # N x n_kpts x h x w
        target = torch.from_numpy(target).to(src_output)
        target_weights = torch.from_numpy(target_weights[..., None]).to(src_output)

        bs_eff = len(out_list)
        weight = self.loss_weight * min(self.cur_step / self.warmup_steps, 1.)
        mse_loss = weighted_heatmap_mse(self.criterion, src_output, target, target_weights) * weight * bs / bs_eff
        res = {self.loss_name: mse_loss}

        self.cur_step += 1
        return res
