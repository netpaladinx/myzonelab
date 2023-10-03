from collections import OrderedDict

import numpy as np
import torch
from ...registry import REID_POSTPROCESSOR
from ...consts import IMAGENET_RGB_MEAN, IMAGENET_RGB_STD
from ...utils import to_numpy, to_img_np, inv_normalize
from .base_process import BaseProcess


@REID_POSTPROCESSOR.register_class('predict')
class ReIDPredict(BaseProcess):
    def __init__(self, default='predict_with_batch_dict', mean=IMAGENET_RGB_MEAN, std=IMAGENET_RGB_STD):
        super().__init__(default)
        self.mean = mean
        self.std = std

    def predict_with_batch_dict(self, output, batch_dict, features=None,
                                recon_pred=None, recon_gt=None, recon_mask=None):
        results = {}
        ann_ids = []
        fighter_gids = []
        batch_size = len(output)
        for i in range(batch_size):
            ann_ids.append(batch_dict['ann_id'][i])
            fighter_gids.append(batch_dict['fighter_gid'][i])
        results['ann_ids'] = ann_ids
        results['fighter_gids'] = fighter_gids

        output_results = to_numpy(output).reshape(batch_size, -1)
        results['output_results'] = output_results

        if features:
            features_results = OrderedDict()
            for name, feature in features.items():
                features_results[name] = to_numpy(feature).reshape(batch_size, -1)
            results['features_results'] = features_results

        if recon_pred is not None:
            if recon_mask is not None:
                recon_mask = torch.from_numpy(np.stack(recon_mask, 0)[:, None]).to(recon_pred)
                recon_pred *= recon_mask
                recon_gt *= recon_mask
            results.update(self.get_recon_results(recon_pred, recon_gt))

        return results

    def get_recon_results(self, recon_pred, recon_gt):
        recon_pred = inv_normalize(recon_pred, self.mean, self.std)
        recon_gt = inv_normalize(recon_gt, self.mean, self.std)
        recon_pred = to_img_np(recon_pred)
        recon_gt = to_img_np(recon_gt)
        return {'recon_pred': recon_pred, 'recon_gt': recon_gt}
