import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from myzonecv.core.utils import read_numpy


def draw_embeddings(data, save_path, labels=None, method='tsne', proj_dims=2, figsize=(16, 16), cmap='tab10', alpha=0.8, fontsize='large', markerscale=1):
    if isinstance(data, np.ndarray):
        assert data.ndim == 2 and labels is not None
    elif isinstance(data, str):
        labels, data = read_numpy(data)
    classes, cnts = np.unique(labels, return_counts=True)

    if method == 'tsne':
        tsne = TSNE(proj_dims, verbose=1)
        data_proj = tsne.fit_transform(data)
    elif method == 'pca':
        pca = PCA(n_components=proj_dims)
        pca.fit(data)
        data_proj = pca.transform(data)

    cmap = cm.get_cmap(cmap)

    if proj_dims == 2:
        fig, ax = plt.subplots(figsize=figsize)
    elif proj_dims == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
    else:
        raise ValueError(f"Invalid proj_dims: {proj_dims}")

    for i, cls in enumerate(classes):
        indices = labels == cls
        if proj_dims == 2:
            ax.scatter(data_proj[indices, 0], data_proj[indices, 1], c=np.array(cmap(i)).reshape(1, 4), label=f'#{cls}: {cnts[i]}', alpha=alpha)
        elif proj_dims == 3:
            ax.scatter(data_proj[indices, 0], data_proj[indices, 1], data_proj[indices, 2], c=np.array(cmap(i)).reshape(1, 4), label=f'#{cls}: {cnts[i]}', alpha=alpha)
        else:
            raise ValueError(f"Invalid proj_dims: {proj_dims}")
    ax.legend(fontsize=fontsize, markerscale=markerscale)
    plt.savefig(save_path)
