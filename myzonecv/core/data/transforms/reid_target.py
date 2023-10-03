import numpy as np
import torch

from ...registry import REID_TRANSFORMS
from ..datautils import get_pairwise_mask


@REID_TRANSFORMS.register_class('generate_cls_target')
class GenerateClsTarget:
    def __init__(self, max_class_id=-1):
        self.max_class_id = max_class_id

    def __call__(self, input_batch, dataset, step):
        for input_dict in input_batch:
            if self.max_class_id > 0:
                assert input_dict['fighter_gid'] <= self.max_class_id
            input_dict['cls_target'] = np.array(input_dict['fighter_gid']).astype(int)

        return input_batch


@REID_TRANSFORMS.register_class('generate_ml_masks')
class GenerateMLMasks:
    def __init__(self):
        pass

    def __call__(self, input_batch, dataset, step):
        grps, mems = self._check_id_list([input_dict['fighter_gid'] for input_dict in input_batch])
        mask_pos = get_pairwise_mask(grps, mems, non_diagonal=True)
        mask_neg = get_pairwise_mask(grps, mems, inter_groups=True)
        mask_tri = get_pairwise_mask(grps * mems, mems - 1, members_b=(grps - 1) * mems)
        input_batch[0]['mask_pos'] = torch.as_tensor(mask_pos)
        input_batch[0]['mask_neg'] = torch.as_tensor(mask_neg)
        input_batch[0]['mask_tri'] = torch.as_tensor(mask_tri)
        input_batch[0]['size_params'] = (grps, mems)
        return input_batch

    def _check_id_list(self, id_list):
        grps = 0
        mems = 0
        prev_mems = -1
        prev_id = -1
        ids = set()
        for id in id_list:
            if prev_id == -1:
                mems = 1
                grps = 1
                ids.add(id)
            elif prev_id != id:
                assert prev_mems == -1 or mems == prev_mems
                assert id not in ids
                prev_mems = mems
                mems = 1
                grps += 1
                ids.add(id)
            else:
                mems += 1
            prev_id = id
        return grps, mems
