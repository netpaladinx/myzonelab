import numpy as np
import cv2

from ..datautils import npf
from ...registry import POSE_TRANSFORMS


@POSE_TRANSFORMS.register_class('generate_heatmaps')
class GenerateHeatmaps:
    def __init__(self, sigma=2, kernel=(11, 11), radius_factor=3, encode='default', use_keypoint_weights=False, laplace_factors=(0.5, 0.8), visibility_thr=0.):
        """ encode: 'default', 'unbiased', 'megvii', 'dark_udp', 'laplace'
        """
        assert encode in ('default', 'unbiased', 'megvii', 'dark_udp', 'laplace')
        self.sigma = sigma
        self.kernel = kernel
        self.radius_factor = radius_factor
        self.encode = encode
        self.use_keypoint_weights = use_keypoint_weights
        self.laplace_shrink = min(max(0.1, laplace_factors[0]), 1)
        self.laplace_weight = min(max(0, laplace_factors[1]), 1)
        self.visibility_thr = visibility_thr

    def __call__(self, input_dict, dataset, step):
        kpts = input_dict['kpts']
        input_size = dataset.input_size
        heatmap_size = dataset.heatmap_size
        keypoint_weights = dataset.keypoint_weights if self.use_keypoint_weights else None

        heatmaps, weights = self.encode_kpts(kpts, input_size, heatmap_size, keypoint_weights)  # n_kpts x h x w, n_kpts x 1

        input_dict['target_heatmaps'] = heatmaps
        input_dict['target_weights'] = weights
        return input_dict

    def encode_kpts(self, kpts, input_size, heatmap_size, keypoint_weights):
        if kpts.ndim == 3:  # N x n_kpts x 3
            rets = [self._encode_kpts(kpts[i], input_size, heatmap_size, keypoint_weights) for i in range(len(kpts))]
            heatmaps, weights = zip(*rets)
            heatmaps = np.stack(heatmaps, 0)  # N x n_kpts x h x w
            weights = np.stack(weights, 0)  # N x n_kpts x 1
        else:
            heatmaps, weights = self._encode_kpts(kpts, input_size, heatmap_size, keypoint_weights)  # n_kpts x h x w, n_kpts x 1
        return heatmaps, weights

    def _encode_kpts(self, kpts, input_size, heatmap_size, keypoint_weights):
        if self.encode == 'default':
            heatmaps, weights = self.default_encode(kpts, input_size, heatmap_size, self.sigma, keypoint_weights)
        elif self.encode == 'unbiased':
            heatmaps, weights = self.unbiased_encode(kpts, input_size, heatmap_size, self.sigma, keypoint_weights)
        elif self.encode == 'megvii':
            heatmaps, weights = self.megvii_encode(kpts, input_size, heatmap_size, self.kernel, keypoint_weights)
        elif self.encode == 'dark_udp':
            heatmaps, weights = self.dark_udp_encode(kpts, input_size, heatmap_size, self.sigma, keypoint_weights)
        elif self.encode == 'laplace':
            heatmaps, weights = self.laplace_encode(kpts, input_size, heatmap_size, self.sigma, keypoint_weights)
        return heatmaps, weights

    def _convert_visibility_to_weight(self, v):
        return 1. if v > self.visibility_thr else 0.

    def default_encode(self, kpts, input_size, heatmap_size, sigma, kpts_weights):
        n_kpts = len(kpts)
        input_w, input_h = input_size
        heatmap_w, heatmap_h = heatmap_size
        heatmaps = np.zeros((n_kpts, heatmap_h, heatmap_w))
        weights = np.zeros((n_kpts, 1))
        stride = (input_w / heatmap_w, input_h / heatmap_h)
        radius = sigma * self.radius_factor  # 3-sigma rule
        size = 2 * radius + 1

        for i in range(n_kpts):
            weights[i] = self._convert_visibility_to_weight(kpts[i, 2])
            mu_x = int(kpts[i, 0] / stride[0] + 0.5)  # center of heatmap corner elements aligned to input border corners
            mu_y = int(kpts[i, 1] / stride[1] + 0.5)
            x0, y0 = int(mu_x - radius), int(mu_y - radius)
            x1, y1 = int(mu_x + radius), int(mu_y + radius)
            if x0 > heatmap_w - 1 or y0 > heatmap_h - 1 or x1 < 0 or y1 < 0:
                weights[i] = 0

            if weights[i] > 0:
                x = np.arange(size)
                y = x[:, None]
                cx = cy = size // 2
                gau = np.exp(- ((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))  # the center value is 1
                gau_x0 = max(0, -x0)
                gau_y0 = max(0, -y0)
                gau_x1 = min(x1, heatmap_w - 1) - x0
                gau_y1 = min(y1, heatmap_h - 1) - y0
                hmap_x0 = max(0, x0)
                hmap_y0 = max(0, y0)
                hmap_x1 = min(x1, heatmap_w - 1)
                hmap_y1 = min(y1, heatmap_h - 1)
                heatmaps[i, hmap_y0:(hmap_y1 + 1), hmap_x0:(hmap_x1 + 1)] = gau[gau_y0:(gau_y1 + 1), gau_x0:(gau_x1 + 1)]

        if kpts_weights:
            weights = weights * npf(kpts_weights).reshape(-1, 1)

        return heatmaps, weights

    def unbiased_encode(self, kpts, input_size, heatmap_size, sigma, kpts_weights):
        n_kpts = len(kpts)
        input_w, input_h = input_size
        heatmap_w, heatmap_h = heatmap_size
        heatmaps = np.zeros((n_kpts, heatmap_h, heatmap_w))
        weights = np.zeros((n_kpts, 1))
        stride = (input_w / heatmap_w, input_h / heatmap_h)
        radius = sigma * self.radius_factor  # 3-sigma rule

        for i in range(n_kpts):
            weights[i] = self._convert_visibility_to_weight(kpts[i, 2])
            mu_x = kpts[i, 0] / stride[0]
            mu_y = kpts[i, 1] / stride[1]
            x0, y0 = mu_x - radius, mu_y - radius
            x1, y1 = mu_x + radius, mu_y + radius
            if x0 > heatmap_w - 1 or y0 > heatmap_h - 1 or x1 < 0 or y1 < 0:
                weights[i] = 0

            if weights[i] > 0:
                x = np.arange(heatmap_w)
                y = np.arange(heatmap_h)[:, None]
                heatmaps[i] = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2))

        if kpts_weights:
            weights = weights * npf(kpts_weights).reshape(-1, 1)

        return heatmaps, weights

    def megvii_encode(self, kpts, input_size, heatmap_size, kernel, kpts_weights):
        n_kpts = len(kpts)
        input_w, input_h = input_size
        heatmap_w, heatmap_h = heatmap_size
        heatmaps = np.zeros((n_kpts, heatmap_h, heatmap_w))
        weights = np.zeros((n_kpts, 1))
        stride = (input_w / heatmap_w, input_h / heatmap_h)

        for i in range(n_kpts):
            weights[i] = self._convert_visibility_to_weight(kpts[i, 2])
            mu_x = int(kpts[i, 0] / stride[0])  # border corner of heatmap corner elements aligned to input border corners
            mu_y = int(kpts[i, 1] / stride[1])
            if mu_x > heatmap_w - 1 or mu_y > heatmap_h - 1 or mu_x < 0 or mu_y < 0:
                weights[i] = 0

            if weights[i] > 0:
                heatmaps[i, mu_y, mu_x] = 1
                heatmaps[i] = cv2.GaussianBlur(heatmaps[i], kernel, 0)
                maxval = heatmaps[i, mu_y, mu_x]
                heatmaps[i] = heatmaps[i] / maxval

        if kpts_weights:
            weights = weights * npf(kpts_weights).reshape(-1, 1)

        return heatmaps, weights

    def dark_udp_encode(self, kpts, input_size, heatmap_size, sigma, kpts_weights):
        n_kpts = len(kpts)
        input_w, input_h = input_size
        heatmap_w, heatmap_h = heatmap_size
        heatmaps = np.zeros((n_kpts, heatmap_h, heatmap_w))
        weights = np.zeros((n_kpts, 1))
        stride = (input_w - 1) / (heatmap_w - 1), (input_h - 1) / (heatmap_h - 1)
        radius = sigma * self.radius_factor  # 3-sigma rule
        size = 2 * radius + 1

        for i in range(n_kpts):
            weights[i] = self._convert_visibility_to_weight(kpts[i, 2])
            mu_x = int(kpts[i, 0] / stride[0] + 0.5)  # snap to grid lines instead of pixels
            mu_y = int(kpts[i, 1] / stride[1] + 0.5)
            x0, y0 = int(mu_x - radius), int(mu_y - radius)
            x1, y1 = int(mu_x + radius), int(mu_y + radius)
            if x0 > heatmap_w - 1 or y0 > heatmap_h - 1 or x1 < 0 or y1 < 0:
                weights[i] = 0

            if weights[i] > 0:
                x = np.arange(size)
                y = x[:, None]
                mu_x_real = kpts[i, 0] / stride[0]
                mu_y_real = kpts[i, 1] / stride[1]
                cx = cy = size // 2
                cx = cx + mu_x_real - mu_x
                cy = cy + mu_y_real - mu_y
                gau = np.exp(- ((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
                gau_x0 = max(0, -x0)
                gau_y0 = max(0, -y0)
                gau_x1 = min(x1, heatmap_w - 1) - x0
                gau_y1 = min(y1, heatmap_h - 1) - y0
                hmap_x0 = max(0, x0)
                hmap_y0 = max(0, y0)
                hmap_x1 = min(x1, heatmap_w - 1)
                hmap_y1 = min(y1, heatmap_h - 1)
                heatmaps[i, hmap_y0:(hmap_y1 + 1), hmap_x0:(hmap_x1 + 1)] = gau[gau_y0:(gau_y1 + 1), gau_x0:(gau_x1 + 1)]

        if kpts_weights:
            weights = weights * npf(kpts_weights).reshape(-1, 1)

        return heatmaps, weights

    def laplace_encode(self, kpts, input_size, heatmap_size, sigma, kpts_weights):
        n_kpts = len(kpts)
        input_w, input_h = input_size
        heatmap_w, heatmap_h = heatmap_size
        heatmaps = np.zeros((n_kpts, heatmap_h, heatmap_w))
        weights = np.zeros((n_kpts, 1))
        stride = (input_w / heatmap_w, input_h / heatmap_h)

        if not isinstance(sigma, (list, tuple)):
            gau_sigma = sigma
            lap_sigma = sigma * self.laplace_shrink
        else:  # sigma = (3, 1.5)
            gau_sigma, lap_sigma = sigma

        radius = gau_sigma * self.radius_factor  # 3-sigma rule
        size = 2 * radius + 1

        for i in range(n_kpts):
            weights[i] = self._convert_visibility_to_weight(kpts[i, 2])
            mu_x = int(kpts[i, 0] / stride[0] + 0.5)  # center of heatmap corner elements aligned to input border corners
            mu_y = int(kpts[i, 1] / stride[1] + 0.5)
            x0, y0 = int(mu_x - radius), int(mu_y - radius)
            x1, y1 = int(mu_x + radius), int(mu_y + radius)
            if x0 > heatmap_w - 1 or y0 > heatmap_h - 1 or x1 < 0 or y1 < 0:
                weights[i] = 0

            if weights[i] > 0:
                x = np.arange(size)
                y = x[:, None]
                cx = cy = size // 2
                gau = np.exp(- ((x - cx)**2 + (y - cy)**2) / (2 * gau_sigma**2))  # the center value is 1
                lap = np.exp(-((x - cx)**2 + (y - cy)**2)**0.5 * (2**0.5) / lap_sigma)
                lap = self.laplace_weight * lap + (1 - self.laplace_weight) * gau
                lap_x0 = max(0, -x0)
                lap_y0 = max(0, -y0)
                lap_x1 = min(x1, heatmap_w - 1) - x0
                lap_y1 = min(y1, heatmap_h - 1) - y0
                hmap_x0 = max(0, x0)
                hmap_y0 = max(0, y0)
                hmap_x1 = min(x1, heatmap_w - 1)
                hmap_y1 = min(y1, heatmap_h - 1)
                heatmaps[i, hmap_y0:(hmap_y1 + 1), hmap_x0:(hmap_x1 + 1)] = lap[lap_y0:(lap_y1 + 1), lap_x0:(lap_x1 + 1)]

        if kpts_weights:
            weights = weights * npf(kpts_weights).reshape(-1, 1)

        return heatmaps, weights
