import numpy as np

from myzonecv.core.utils import get_color_stats, img_as_float
from ..registry import COLORADJUST_TRANSFORMS
from .coloradjust_consts import SIDE_TRUNCATION


@COLORADJUST_TRANSFORMS.register_class('generate_input_stats')
class GenerateInputStats:
    def __init__(self, side_truncation=SIDE_TRUNCATION):
        self.side_truncation = side_truncation

    def __call__(self, input_dict, dataset, step):
        img = input_dict['img']

        img = np.clip(img, 1, None)
        stats = get_color_stats(img_as_float(img), side_truncation=self.side_truncation)
        assert len(set(dataset.input_stats) & stats.keys()) == len(dataset.input_stats)
        input_stats = np.array([stats[name] for name in dataset.input_stats])
        input_dict.update(stats)

        input_dict['img'] = img
        input_dict['input_stats'] = input_stats
        return input_dict
