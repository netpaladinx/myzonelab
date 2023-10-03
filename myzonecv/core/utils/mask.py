import numpy as np


def get_pairwise_mask(groups, members, members_b=None, inter_groups=False, non_diagonal=False, upper_triangular=False, strictly_upper=False):
    if members_b is None:
        members_b = members
    group_indices = np.repeat(np.arange(groups), members)
    group_indices_b = np.repeat(np.arange(groups), members_b)
    mask = group_indices[:, None] == group_indices_b
    if inter_groups:
        mask = np.logical_not(mask)
    if non_diagonal or upper_triangular:
        nrow, ncol = mask.shape
        row_indices = np.arange(nrow) * 1. / nrow
        col_indices = np.arange(ncol) * 1. / ncol
        if non_diagonal:
            m = row_indices[:, None] != col_indices
        else:
            if strictly_upper:
                m = row_indices[:, None] < col_indices
            else:
                m = row_indices[:, None] <= col_indices
        mask = mask & m
    return mask
