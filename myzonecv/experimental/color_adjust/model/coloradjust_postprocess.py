import numpy as np

from myzonecv.core.model import BaseProcess
from myzonecv.core.utils import img_as_float, get_color_stats, apply_sigmoidal, apply_gamma, apply_saturation, img_as_ubyte
from ..registry import COLORADJUST_POSTPROCESSORS
from ..data.coloradjust_consts import SIDE_TRUNCATION, IMAGE_STATS, CONTRAST_RANGE, BIAS_RANGE, GAMMA_RANGE, SATURATION_RANGE


@COLORADJUST_POSTPROCESSORS.register_class('predict')
class ColorAdjustPredict(BaseProcess):
    def __init__(self, default='predict_from_scratch'):
        super().__init__(default)
        self.contrast_range = CONTRAST_RANGE
        self.bias_range = BIAS_RANGE
        self.gamma_range = GAMMA_RANGE
        self.saturation_range = SATURATION_RANGE
        self.side_truncation = SIDE_TRUNCATION

    def predict_from_scratch(self, batch_dict, model):
        in_imgs = batch_dict['img_np']
        in_stats = self.compute_stats(in_imgs)
        op_params = self.predict_op_params(in_stats, model)
        out_imgs = [self.adjust_image(img,
                                      op_params['contrast'][i], op_params['bias'][i],
                                      op_params['red'][i], op_params['green'][i], op_params['blue'][i],
                                      op_params['saturation'][i])
                    for i, img in enumerate(in_imgs)]
        out_stats = self.compute_stats(out_imgs)
        return in_imgs, out_imgs, in_stats, out_stats, op_params

    def compute_stats(self, imgs):
        in_stats = []
        for img in imgs:
            assert img.dtype == np.uint8
            img = img_as_float(img, min=1)
            stats = get_color_stats(img, side_truncation=self.side_truncation)
            stats = np.array([stats[name] for name in IMAGE_STATS])
            in_stats.append(stats)
        in_stats = np.stack(in_stats, axis=0)
        return in_stats

    def predict_op_params(self, x, model):
        """ x: bs x n_stats
        """
        out = x
        layers = model.backbone.export_to_numpy()
        for layer in layers:
            if layer[0] == 'linear':
                weight, bias = layer[1:]
                out = out @ weight.T + bias
            elif layer[0] == 'relu':
                out = np.clip(out, 0, None)

        out = 1 / (1 + np.exp(-out))
        contrast = out[:, 0] * (self.contrast_range[1] - self.contrast_range[0]) + self.contrast_range[0]  # bs
        bias = out[:, 1] * (self.bias_range[1] - self.bias_range[0]) + self.bias_range[0]  # bs
        red = out[:, 2] * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]  # bs
        green = out[:, 3] * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]  # bs
        blue = out[:, 4] * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]  # bs
        saturation = out[:, 5] * (self.saturation_range[1] - self.saturation_range[0]) + self.saturation_range[0]  # bs
        out_params = {
            'contrast': contrast, 'bias': bias,
            'red': red, 'green': green, 'blue': blue,
            'saturation': saturation
        }
        return out_params

    @staticmethod
    def adjust_image(img, contrast, bias, red, green, blue, saturation):
        assert img.dtype == np.uint8
        img = img_as_float(img, min=1)
        img = apply_sigmoidal(img, contrast, bias)
        img = apply_gamma(img, (red, green, blue))
        img = apply_saturation(img, saturation)
        img = img_as_ubyte(img)
        return img
