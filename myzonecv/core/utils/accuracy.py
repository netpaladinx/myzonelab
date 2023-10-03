import numpy as np


def compute_cmc(dist_mat, query_sizes, gallery_sizes, repeated_times=100, rank_k=(1, 3, 5, 10), seed=0):
    """ Cumulative matching characteristic.

        dist_mat: N x M, where N is #query-samples and M is #gallery-samples
        query_sizes: (n1, n2, ..., nK), where N = n1 + n2 + ... + nK with K query groups
        gallery_sizes: (m1, m2, ..., mL), where M = m1 + m2 + ... + mL with L gallery groups

        First min(K, L) groups are the corresponding groups between query and gallery.

        return: acc, where acc[i] means the averaged rank-i accuracy
    """
    K = len(query_sizes)
    nvec = np.array(query_sizes)
    cum_nvec = nvec.cumsum()

    L = len(gallery_sizes)
    mvec = np.array(gallery_sizes)
    cum_mvec = mvec.cumsum()

    rng = np.random.default_rng(seed)

    accs = []
    for _ in range(repeated_times):
        query_samples = np.floor(nvec * rng.random(K)).astype(int)
        query_samples[1:] += cum_nvec[:-1]

        gallery_samples = np.floor(mvec * rng.random(L)).astype(int)
        gallery_samples[1:] += cum_mvec[:-1]

        sampled_dist = dist_mat[query_samples][:, gallery_samples]
        sorted_indices = np.argsort(sampled_dist)
        match = (sorted_indices == np.arange(K)[:, None])
        acc = (match.sum(axis=0) * 1.0 / K).cumsum()
        accs.append(acc)

    acc = np.mean(accs, 0)
    return [acc[k] for k in rank_k if k < len(acc)]


def compute_ap(rc, pr, method='interp'):
    rc = np.concatenate(([0.], rc, [rc[-1] + 0.01]))
    pr = np.concatenate(([1.], pr, [0.]))
    # adjusted to precision envelop
    pr = np.flip(np.maximum.accumulate(np.flip(pr)))
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, rc, pr), x)  # integrate
    elif method == 'continuous':
        i = np.where(rc[1:] != rc[:-1])[0]  # points where x axis changes
        ap = np.sum((rc[i + 1] - rc[i]) * pr[i + 1])  # area under curve
    else:
        raise ValueError(f"Invalid method {method}")
    return ap, rc, pr


def compute_map(dist_mat, query_sizes, gallery_sizes, query_is_gallery=False):
    """ Mean average precision.

        dist_mat: N x M, where N is #query-samples and M is #gallery-samples
        query_sizes: (n1, n2, ..., nK), where N = n1 + n2 + ... + nK with K query groups
        gallery_sizes: (m1, m2, ..., mL), where M = m1 + m2 + ... + mL with L gallery groups

        First min(K, L) groups are the corresponding groups between query and gallery.
    """
    K = len(query_sizes)
    nvec = np.array(query_sizes)

    L = len(gallery_sizes)
    mvec = np.array(gallery_sizes)

    query_labels = np.repeat(np.arange(K), nvec)
    gallery_labels = np.repeat(np.arange(L), mvec)

    sorted_indices = np.argsort(dist_mat)
    match = (query_labels[:, None] == gallery_labels[sorted_indices])
    if query_is_gallery:
        match = match[:, 1:]
    tp_mat = np.cumsum(match, axis=1).astype(float)
    fp_mat = np.cumsum(np.logical_not(match), axis=1).astype(float)
    aps = []
    for i, (tp_row, fp_row) in enumerate(zip(tp_mat, fp_mat)):
        q_label = query_labels[i]
        n_gt = mvec[q_label] - 1 if query_is_gallery else mvec[q_label]
        rec = tp_row / (n_gt + np.spacing(1))
        prec = tp_row / (tp_row + fp_row + np.spacing(1))
        ap, rec, prec = compute_ap(rec, prec)
        aps.append(ap)

    mean_ap = np.mean(aps)
    return mean_ap
