import numpy as np

from ...registry import POSE_POSTPROCESSOR
from ...utils import npf
from .base_process import BaseProcess
from .pose_predict import PoseSimplePredict


@POSE_POSTPROCESSOR.register_class('accuracy')
class PoseAccuracy(BaseProcess):
    def __init__(self, thr=0.05, default='accuracy_by_heatmap', accuracy_name='acc'):
        super().__init__(default)
        self.thr = thr
        self.accuracy_name = accuracy_name
        self.predict = PoseSimplePredict()

    def accuracy_by_heatmap(self, output, target, thr=None, mask=None, units=None):
        """ output, target: bs x kpts x height x width
            mask: bs x kpts
            units: bs x 2
        """
        n_kpts = output.shape[1]
        if n_kpts == 0:
            return dict(accs=None, avg_acc=0, valid_kpts=0)

        if thr is None:
            thr = self.thr
        if mask is None:
            mask = self._get_mask(output)
        if units is None:
            units = self._get_units(output)

        preds, _ = self.predict(output)
        gts, _ = self.predict(target)
        return self.accuracy_by_location(preds, gts, thr, mask, units)

    def accuracy_by_location(self, preds, gts, thr, mask, units):
        """ preds, gts: bs x kpts x 2
            mask: bs x kpts
            units: bs x 2

            Accuracy is computed based on PCK.
        """
        dist_px, dist_nm = self._calc_distances(preds, gts, mask, units)  # ktps x bs, kpts x bs

        avg_dist_per_obj, max_dist_per_obj = zip(*[self._distance_stats_in_pixels(d) for d in dist_px.T])  # bs, bs

        avg_dist_per_obj = npf(avg_dist_per_obj)
        max_dist_per_obj = npf(max_dist_per_obj)
        avg_dist = self._valid_mean(avg_dist_per_obj, default=0)
        avgmax_dist = self._valid_mean(max_dist_per_obj, default=0)
        maxmax_dist = self._valid_max(max_dist_per_obj, default=0)

        acc_per_kpt = npf([self._accuracy_by_distance(d, thr) for d in dist_nm])  # kpts
        avg_acc = self._valid_mean(acc_per_kpt, default=0)

        return {f'{self.accuracy_name}_per_kpt': acc_per_kpt,
                f'avg_{self.accuracy_name}': avg_acc,
                'avg_dist': avg_dist,
                'amax_dist': avgmax_dist,
                'mmax_dist': maxmax_dist,
                'pixel_dist': dist_px,
                'valid_kpts': (acc_per_kpt >= 0).sum(),
                'summary_keys': (f'avg_{self.accuracy_name}', 'avg_dist', 'amax_dist', 'mmax_dist')}

    @staticmethod
    def _get_mask(heatmaps):
        batch_size, n_kpts = heatmaps.shape[:2]
        return np.ones((batch_size, n_kpts), dtype=np.bool_)  # bs x kpts

    @staticmethod
    def _get_units(heatmaps):
        batch_size, _, height, width = heatmaps.shape
        return np.tile(npf([[height, width]]), (batch_size, 1))  # bs x 2

    @staticmethod
    def _valid_mean(arr, default):
        arr_valid = arr[arr >= 0]
        n_valid = len(arr_valid)
        return arr_valid.mean() if n_valid > 0 else default

    @staticmethod
    def _valid_max(arr, default):
        arr_valid = arr[arr >= 0]
        n_valid = len(arr_valid)
        return arr_valid.max() if n_valid > 0 else default

    @staticmethod
    def _valid_min(arr, default):
        arr_valid = arr[arr >= 0]
        n_valid = len(arr_valid)
        return arr_valid.min() if n_valid > 0 else default

    @staticmethod
    def _calc_distances(preds, gts, mask, units):
        """ preds, gts: bs x kpts x 2
            mask: bs x kpts
            units: bs x 2
        """
        batch_size, n_kpts = preds.shape[:2]
        mask = mask.copy()
        invalid = np.where((units <= 0).sum(1))[0]
        mask[invalid, :] = False

        dist_px = np.full((batch_size, n_kpts), -1.)  # bs x kpts
        dist_nm = np.full((batch_size, n_kpts), -1.)  # bs x kpts
        dist_px[mask] = np.linalg.norm((preds - gts)[mask], axis=-1)
        dist_nm[mask] = np.linalg.norm(((preds - gts) / units[:, None, :])[mask], axis=-1)
        return dist_px.T, dist_nm.T  # kpts x bs, kpts x bs

    @staticmethod
    def _accuracy_by_distance(dist, thr):  # dist: bs
        valid = dist != -1
        n_valid = valid.sum()
        return (dist[valid] < thr).sum() / n_valid if n_valid > 0 else -1

    @staticmethod
    def _distance_stats_in_pixels(dist):  # dist: bs or kpts
        valid = dist != -1
        n_valid = valid.sum()
        avg_dist = dist[valid].mean() if n_valid > 0 else -1
        max_dist = dist[valid].max() if n_valid > 0 else -1
        return avg_dist, max_dist
