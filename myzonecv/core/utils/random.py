import os
import random

import numpy as np
import torch

from .dist import get_dist_info


def set_random_seed(seed, deterministic=False, use_rank_shift=False):
    if use_rank_shift:
        rank, _ = get_dist_info()
        seed += rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
