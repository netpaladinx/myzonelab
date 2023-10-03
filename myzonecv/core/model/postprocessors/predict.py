import torch
import torch.nn.functional as F

from ...registry import POSTPROCESSORS
from .base_process import BaseProcess


@POSTPROCESSORS.register_class('map_predict')
class MapPredict(BaseProcess):
    def __init__(self, default='predict_with_batch_dict', align_corners=None):
        super().__init__(default)
        self.align_corners = align_corners

    def predict_with_batch_dict(self, output, batch_dict):
        """ output (torch.Tensor): N x ... x H x W
            batch_dict (dict):
                'ori_shape': [ori_w, ori_h] or list([ori_w, ori_h])
                'scale': [s_w, s_h] or list([s_w, s_h])
                'border': [top, bottom, left, right] or list([top, bottom, left, right])
                'flipped': bool or list(bool)
                'flip_direction': str or list(str)
        """
        assert output.ndim >= 3
        ori_shape = batch_dict.get('ori_shape')
        scale = batch_dict.get('scale')
        border = batch_dict.get('border')
        flipped = batch_dict.get('flipped')
        flip_direction = batch_dict.get('flip_direction')

        with torch.no_grad():
            outs = output.split(1)

            if ori_shape is not None:
                if isinstance(ori_shape[0], (list, tuple)):
                    assert len(ori_shape) == len(outs)
                else:
                    ori_shape = [ori_shape] * len(outs)

                outs = [F.interpolate(out[None, None], size=(ori_h, ori_w), mode='bilinear', align_cornesr=self.align_corners)[0, 0]
                        for out, (ori_w, ori_h) in zip(outs, ori_shape)]

            elif scale is not None:
                if isinstance(scale[0], (list, tuple)):
                    assert len(scale) == len(outs)
                else:
                    scale = [scale] * len(outs)

                outs = [F.interpolate(out[None, None], scale_factor=(1. / s_h, 1. / s_w), mode='bilinear', align_cornesr=self.align_corners)[0, 0]
                        for out, (s_w, s_h) in zip(outs, scale)]

            if border is not None:
                if isinstance(border[0], (list, tuple)):
                    assert len(border) == len(outs)
                else:
                    border = [border] * len(outs)

                outs = [out[..., top:(out.shape[-2] - bottom), left:(out.shape[-1] - right)]
                        for out, (top, bottom, left, right) in zip(outs, border)]

            if flipped is not None:
                if isinstance(flipped, (list, tuple)):
                    assert len(flipped) == len(outs)
                else:
                    flipped = [flipped] * len(outs)

                if isinstance(flip_direction, (list, tuple)):
                    assert len(flip_direction) == len(outs)
                else:
                    flip_direction = [flip_direction] * len(outs)

                outs = [(out.flip(dims=(-1,)) if fd == 'horizontal' else out.flip(dims=(-2,))) if is_f else out
                        for out, is_f, fd in zip(outs, flipped, flip_direction)]

            if output.ndim == 3 or (output.n_dim > 3 and output.shape[1] == 1):
                outs = [out.sigmoid() for out in outs]
                preds = [out > 0.5 for out in outs]
            else:
                outs = [F.softmax(out, dim=0) for out in outs]
                preds = [out.argmax(dim=0) for out in outs]

            outs = [out.detach().cpu().numpy() for out in outs]
            preds = [pred.detach().cpu().numpy() for pred in preds]
            return {'pred_results': preds, 'pred_outputs': outs}
