import numpy as np
import torch

from ...registry import DATA_TRANSFORMS
from ...errors import DataTransformCheckError
from ...utils import make_divisible


@DATA_TRANSFORMS.register_class('check_img_size')
class CheckImgSize:
    def __init__(self, divisable_by=32, floor=32):
        self.divisable_by = divisable_by
        self.floor = floor

    def __call__(self, input_dict, dataset, step):
        img = input_dict['img']
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
        elif isinstance(img, torch.Tensor):
            h, w = img.shape[1:]
        else:
            raise DataTransformCheckError(f"Invliad img: {img}")

        new_h = max(make_divisible(h, self.divisable_by), self.floor)
        new_w = max(make_divisible(w, self.divisable_by), self.floor)
        if h != new_h or w != new_w:
            raise DataTransformCheckError(f"Image size {h}x{w} must be multiple of {self.divisable_by}")

        return input_dict
