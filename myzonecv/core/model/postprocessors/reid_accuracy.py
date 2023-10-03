from collections import OrderedDict

import numpy as np

from ...registry import REID_POSTPROCESSOR
from ...utils import compute_map
from .base_process import BaseProcess


@REID_POSTPROCESSOR.register_class('accuracy')
class ReIDAccuracy(BaseProcess):
    def __init__(self, reid_head, default='compute_batch_map', distance='l2'):
        super().__init__(default)
        if distance == 'l1':
            self.distance = self.l1_distance
        elif distance == 'l2':
            self.distance = self.l2_distance
        elif distance == 'angular':
            self.distance = self.angular_distance
        else:
            raise ValueError(f"Invalid distance: {distance}")

    def compute_batch_map(self, x_dict, groups, members):
        N, nvec = members * groups, [members] * groups
        results = OrderedDict()
        dists = self.compute_distances(x_dict)
        for k, dist in dists.items():
            assert dist.shape == (N, N)
            results[f'{k}_mAP'] = compute_map(dist, nvec, nvec, query_is_gallery=True)
        return results

    def compute_distances(self, x_dict):
        dists = OrderedDict()
        for k, v in x_dict.items():
            dists[k] = self.distance(v)
        return dists

    @staticmethod
    def l1_distance(x):
        return np.linalg.norm(x[:, None] - x, ord=1, axis=-1)

    @staticmethod
    def l2_distance(x):
        return np.linalg.norm(x[:, None] - x, ord=2, axis=-1)

    @staticmethod
    def angular_distance(x):
        x_n = x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
        return np.arccos(np.clip(np.sum(x_n[:, None] * x_n, axis=-1), -1, 1))
