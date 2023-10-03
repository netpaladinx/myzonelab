import numpy as np

from ...registry import POSE_POSTPROCESSOR
from ...consts import BBOX_SCALE_UNIT
from ...utils import apply_warp_to_coord, revert_coord
from .base_process import BaseProcess


@POSE_POSTPROCESSOR.register_class('process')
class PoseProcess(BaseProcess):
    @staticmethod
    def flip_heatmap_back(heatmaps_flipped, flip_pairs):
        """ heatmaps_flipped (np.ndarray): bs x kpts x height x width
            flip_pairs (list(tuple))
        """
        heatmaps_flipped_back = heatmaps_flipped.copy()
        for left, right in flip_pairs:  # mirrored pairs
            heatmaps_flipped_back[:, left, ...] = heatmaps_flipped[:, right, ...]
            heatmaps_flipped_back[:, right, ...] = heatmaps_flipped[:, left, ...]
        heatmaps_flipped_back = heatmaps_flipped_back[..., ::-1]
        return heatmaps_flipped_back

    @staticmethod
    def revert_preds(preds, center, scale, heatmap_size, dark_udp=False):  # no rotation allowed
        """ preds: kpts x 2 or 4 or 5
        """
        scale = scale * BBOX_SCALE_UNIT
        src_size = (heatmap_size[0] - 1.0, heatmap_size[1] - 1.0) if dark_udp else heatmap_size
        preds = revert_coord(preds, src_size, scale, center)
        return preds

    @staticmethod
    def revert_preds_by_inverse(preds, mat_inv):
        """ preds: kpts x 2 or 4 or 5
        """
        preds[:, :2] = apply_warp_to_coord(preds[:, :2], mat_inv)
        return preds
