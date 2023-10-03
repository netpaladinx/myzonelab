import os.path as osp
from collections import deque, defaultdict

import numpy as np

from ....utils import list2str, plot_hist, mkdir, npf
from .coco_consts import KEYPOINT_SIGMAS, KEYPOINT_NAMES


class COCOConfidence:
    def __init__(self, stream_id, summary_dir=None, verbose=True):
        self.stream_id = stream_id
        self.summary_dir = summary_dir
        self.verbose = verbose

        self.eval_results = []
        self.summary_path = None
        if self.summary_dir is not None:
            if not osp.isdir(self.summary_dir):
                mkdir(self.summary_dir, exist_ok=True)
            self.summary_path = osp.join(self.summary_dir, f'{list2str(stream_id)}_confidence.txt')

    def eval_frame(self, frame_no, frame_result):
        bbox_conf = []
        bbox_xywh = []
        kpts_conf = []
        kpts_conf_min = []

        for ann in frame_result:
            if 'bbox' in ann and len(ann['bbox']) >= 5:
                bbox_conf.append(ann['bbox'][4])
                bbox_xywh += ann['bbox'][:4]

            if 'keypoints' in ann:
                kpts_confs = npf(ann['keypoints']).reshape(-1, 3)[:, 2]
                kpts_confs = kpts_confs[kpts_confs > 0]
                kpts_conf.append(kpts_confs.mean())
                kpts_conf_min.append(kpts_confs.min())

        log_str = f"stream: {self.stream_id}, frame: {frame_no}"
        if bbox_conf:
            log_str += f", bbox_conf: {list2str(bbox_conf, tmpl='{:.5f}')}"
            log_str += f", bbox_xywh: {list2str(bbox_xywh, tmpl='{:.5f}')}"
        if kpts_conf:
            log_str += f", kpts_conf: {list2str(kpts_conf, tmpl='{:.5f}')}"
        if kpts_conf_min:
            log_str += f", kpts_conf_min: {list2str(kpts_conf_min, tmpl='{:.5f}')}"

        if self.verbose:
            print(log_str)

        if self.summary_path:
            with open(self.summary_path, 'a+') as fout:
                fout.write(log_str + '\n')

        self.eval_results.append({'frame_index': frame_no,
                                  'bbox_conf': bbox_conf,
                                  'bbox_xywh': bbox_xywh,
                                  'kpts_conf': kpts_conf,
                                  'kpts_conf_min': kpts_conf_min})

    def summarize(self):
        pass


def compute_match(scores, thr=0.):
    n_rows, n_cols = scores.shape
    col_idx, row_idx = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    mask = scores > thr
    scores = scores[mask]
    col_idx = col_idx[mask]
    row_idx = row_idx[mask]
    sort_idx = scores.argsort()[::-1]
    col_mat = col_idx[sort_idx]
    row_mat = row_idx[sort_idx]
    mat = []
    while len(col_mat) > 0 and len(row_mat) > 0:
        ci = col_mat[0]
        ri = row_mat[0]
        mat.append((ri, ci))
        mask = (col_mat != ci) & (row_mat != ri)
        col_mat = col_mat[mask]
        row_mat = row_mat[mask]
    return mat


class COCOSmoothness:
    def __init__(self, stream_id, max_len=3, osk_thr=0.01, summary_dir=None, verbose=True):
        self.stream_id = stream_id
        self.max_len = max_len
        assert osk_thr >= 0
        self.osk_thr = osk_thr
        self.summary_dir = summary_dir
        self.verbose = verbose

        self.result_queue = deque()
        self.last_frame = -1
        self.eval_results = []
        self.summary_path = None
        if self.summary_dir is not None:
            if not osp.isdir(self.summary_dir):
                mkdir(self.summary_dir, exist_ok=True)
            self.summary_path = osp.join(self.summary_dir, f'{list2str(stream_id)}_smooth.txt')

    def __len__(self):
        return len(self.result_queue)

    def push(self, frame_no, frame_result):
        if self.last_frame == -1:
            self.last_frame = frame_no
        else:
            assert self.last_frame + 1 == frame_no
            self.last_frame += 1

        self.result_queue.append(frame_result)
        while len(self) > self.max_len:
            self.result_queue.popleft()

        if len(self) == self.max_len:
            self.eval_results.append(self.evaluate_queue())

    def evaluate_queue(self):
        mid = len(self) // 2
        left, right = 0, len(self) - 1
        left_res = self.result_queue[left]
        right_res = self.result_queue[right]
        left += 1
        right -= 1
        while left < mid:
            left_res = self.compute_average_result(left_res, self.result_queue[left])
            left += 1
        while right > mid:
            right_res = self.compute_average_result(right_res, self.result_queue[right])
            right -= 1
        mid_res = self.compute_average_result(left_res, right_res)
        pred_res = self.result_queue[mid]
        smooth_res = self.compute_smoothness(pred_res, mid_res)

        frame_index = self.last_frame - (len(self) - mid)
        log_str = (f"stream: {self.stream_id}, frame: {frame_index}, "
                   f"avg_osk: {list2str(smooth_res['avg_osk'], tmpl='{:.5f}')}, "
                   f"worst_osk: {list2str(smooth_res['worst_osk'], tmpl='{:.5f}')}, "
                   f"avg_pdiff: {list2str(smooth_res['avg_pdiff'], tmpl='{:.5f}')}, "
                   f"worst_pdiff: {list2str(smooth_res['worst_pdiff'], tmpl='{:.5f}')}")

        if self.verbose:
            print(log_str)

        if self.summary_path:
            with open(self.summary_path, 'a+') as fout:
                fout.write(log_str + '\n')

        return {'frame_index': frame_index, **smooth_res}

    @staticmethod
    def _get_area(ann):
        return ann['area'] if 'area' in ann else np.prod(ann['bbox'][2:4])

    @staticmethod
    def _get_points(ann):
        if 'points' in ann:
            return npf(ann['points']).reshape(-1, 3).T  # 3 x n_points
        elif 'keypoints' in ann:
            return npf(ann['keypoints']).reshape(-1, 3).T  # 3 x n_kpts
        elif 'bbox' in ann:
            x0, y0, w, h = ann['bbox'][:4]
            cx, cy = x0 + w / 2, y0 + h / 2
            return npf([[cx, cy, 1, w, h, 1]]).reshape(-1, 3).T  # 3 x 2
        else:
            raise ValueError("Cannot find 'points', 'keypoints', or 'bbox' in ann")

    @staticmethod
    def _get_vars(n):
        if n == len(KEYPOINT_SIGMAS):
            sigmas = npf(KEYPOINT_SIGMAS)
        else:
            sigmas = np.empty(n)
            sigmas.fill(npf(KEYPOINT_SIGMAS).mean())

        return (sigmas * 2) ** 2

    @staticmethod
    def osk_fn(x1, y1, area1, x2, y2, area2, mask=None):
        vars = COCOSmoothness._get_vars(len(x1))
        dx, dy = x1 - x2, y1 - y2
        err = (dx**2 + dy**2) / vars / (area1 + area2)
        osk = np.exp(-err)
        if mask is not None:
            osk[np.logical_not(mask)] = 0
            osk_masked = osk[mask]
        else:
            osk_masked = osk
        if len(osk_masked) == 0:
            avg_osk, min_osk = 0, 0
        else:
            avg_osk, min_osk = osk_masked.mean(), osk_masked.min()
        return osk, avg_osk, min_osk

    @staticmethod
    def pixeldiff_fn(x1, y1, area1, x2, y2, area2, mask=None):
        dx, dy = x1 - x2, y1 - y2
        pdiff = (dx**2 + dy**2)**0.5
        if mask is not None:
            pdiff[np.logical_not(mask)] = -1
            pdiff_masked = pdiff[mask]
        else:
            pdiff_masked = pdiff
        if len(pdiff_masked) == 0:
            avg_pdiff, max_pdiff = -1, -1
        else:
            avg_pdiff, max_pdiff = pdiff_masked.mean(), pdiff_masked.max()
        return pdiff, avg_pdiff, max_pdiff

    def compute_metric(self, res1, res2, metric_fn=None):
        avg_osk, worst_osk, points_osk = [], {}, {}
        avg_metric, worst_metric, points_metric = {}, {}, {}
        n1, n2 = len(res1), len(res2)
        for ri, ann1 in enumerate(res1):
            x1, y1, v1 = self._get_points(ann1)
            area1 = self._get_area(ann1)
            for ci, ann2 in enumerate(res2):
                x2, y2, v2 = self._get_points(ann2)
                area2 = self._get_area(ann2)

                mask = (v1 > 0) & (v2 > 0)
                n_valid = np.count_nonzero(mask)
                if n_valid == 0:
                    avg_osk.append(-1)
                else:
                    o, avg_o, worst_o = self.osk_fn(x1, y1, area1, x2, y2, area2, mask)
                    avg_osk.append(avg_o)
                    worst_osk[ri, ci] = worst_o
                    points_osk[ri, ci] = o
                    if callable(metric_fn):
                        m, avg_m, worst_m = metric_fn(x1, y1, area1, x2, y2, area2, mask)
                        avg_metric[ri, ci] = avg_m
                        worst_metric[ri, ci] = worst_m
                        points_metric[ri, ci] = m

        avg_osk = npf(avg_osk).reshape(n1, n2)  # n1 x n2
        match = compute_match(avg_osk, thr=self.osk_thr)  # n_matches x 2, (ri, ci)
        avg_osk = [avg_osk[ri, ci] for ri, ci in match]
        worst_osk = [worst_osk[ri, ci] for ri, ci in match]
        points_osk = [points_osk[ri, ci] for ri, ci in match]

        if avg_metric and worst_metric and points_metric:
            avg_metric = [avg_metric[ri, ci] for ri, ci in match]
            worst_metric = [worst_metric[ri, ci] for ri, ci in match]
            points_metric = [points_metric[ri, ci] for ri, ci in match]
            return match, avg_osk, worst_osk, points_osk, avg_metric, worst_metric, points_metric

        return match, avg_osk, worst_osk, points_osk

    def compute_average_result(self, res1, res2):
        new_points = []
        new_area = []
        mat = self.compute_metric(res1, res2)[0]
        for ri, ci in mat:
            ann1, ann2 = res1[ri], res2[ci]
            x1, y1, v1 = self._get_points(ann1)
            x2, y2, v2 = self._get_points(ann2)
            x = (x1 + x2) / 2.
            y = (y1 + y2) / 2.
            v = np.minimum(v1, v2)
            area1 = self._get_area(ann1)
            area2 = self._get_area(ann2)
            new_points.append(np.stack([x, y, v], axis=1))
            new_area.append((area1 + area2) / 2.)
        return [{'points': points, 'area': area} for points, area in zip(new_points, new_area)]

    def compute_smoothness(self, res1, res2):
        _, avg_osk, worst_osk, points_osk, avg_pdiff, worst_pdiff, points_pdiff = \
            self.compute_metric(res1, res2, self.pixeldiff_fn)
        return {'avg_osk': avg_osk, 'points_osk': points_osk, 'worst_osk': worst_osk,
                'avg_pdiff': avg_pdiff, 'worst_pdiff': worst_pdiff, 'points_pdiff': points_pdiff}

    def summarize(self):
        sum_osk, n_osk = 0, 0
        sum_pdiff, n_pdiff = 0, 0
        points_osk = defaultdict(list)
        points_pdiff = defaultdict(list)
        avg_osk, worst_osk = [], []
        avg_pdiff, worst_pdiff = [], []
        for res in self.eval_results:
            for osk in res['points_osk']:
                for i, o in enumerate(osk):
                    if o > 0:
                        points_osk[i].append(o)
                        sum_osk += o
                        n_osk += 1
            for pdiff in res['points_pdiff']:
                for i, pd in enumerate(pdiff):
                    if pd != -1:
                        points_pdiff[i].append(pd)
                        sum_pdiff += pd
                        n_pdiff += 1
            for osk in res['avg_osk']:
                avg_osk.append(osk)
            for osk in res['worst_osk']:
                worst_osk.append(osk)
            for pdiff in res['avg_pdiff']:
                avg_pdiff.append(pdiff)
            for pdiff in res['worst_pdiff']:
                worst_pdiff.append(pdiff)

        path_stem = osp.splitext(self.summary_path)[0]
        osk_options = {'log': True, 'range': (0, 1)}
        pdiff_options = {'log': True, 'range': (0, 100)}

        for i, x in points_osk.items():
            save_path = osp.join(path_stem + f'_osk_kpt{i}.jpg')
            plot_hist(x, 100, save_path, title=f'osk_kpt{i}', **osk_options)

        for i, x in points_pdiff.items():
            save_path = osp.join(path_stem + f'_pdiff_kpt{i}.jpg')
            plot_hist(x, 100, save_path, title=f'pdiff_kpt{i}', **pdiff_options)

        save_path = osp.join(path_stem + f'_avg_osk.jpg')
        plot_hist(avg_osk, 100, save_path, title=f'avg_osk', **osk_options)

        save_path = osp.join(path_stem + f'_worst_osk.jpg')
        plot_hist(worst_osk, 100, save_path, title=f'worst_osk', **osk_options)

        save_path = osp.join(path_stem + f'_avg_pdiff.jpg')
        plot_hist(avg_pdiff, 100, save_path, title=f'avg_pdiff', **pdiff_options)

        save_path = osp.join(path_stem + f'_worst_pdiff.jpg')
        plot_hist(worst_pdiff, 100, save_path, title=f'worst_pdiff', **pdiff_options)

        log_str = f"final_avg_osk: {sum_osk / n_osk}, final_avg_pdiff: {sum_pdiff / n_pdiff}"
        if self.verbose:
            print(log_str)
        if self.summary_path:
            with open(self.summary_path, 'a+') as fout:
                fout.write(log_str + '\n')


class COCOStableness:
    def __init__(self, multiple=10, summary_dir=None, summary_name=None, verbose=False):
        self.multiple = multiple
        self.summary_dir = summary_dir
        self.summary_name = summary_name
        self.verbose = verbose

        self.eval_results = []
        self.summary_path = None
        if self.summary_dir is not None:
            if not osp.isdir(self.summary_dir):
                mkdir(self.summary_dir, exist_ok=True)
            filename = f'{self.summary_name}_stable.txt' if self.summary_name else 'stable.txt'
            self.summary_path = osp.join(self.summary_dir, filename)

    def evaluate_step(self, results, batch):
        ann_kpts = defaultdict(list)
        for ann_id, kpts in zip(results['ann_ids'], results['kpts_results']):
            ann_kpts[tuple(ann_id)].append(kpts)

        ann_ids = batch['ann_id']
        img_ids = batch['img_id']
        log_str = ""
        for img_id, ann_id in zip(img_ids, ann_ids):
            kpts_std, avg_kpts_std = self.compute_stableness(ann_kpts[tuple(ann_id)])
            res = {'img_id': img_id, 'ann_id': ann_id, 'kpts_std': kpts_std, 'avg_kpts_std': avg_kpts_std}
            self.eval_results.append(res)
            log_str += (f"img_id: {list2str(img_id)}, ann_id: {list2str(ann_id)}, "
                        f"kpts_std: {list2str(kpts_std, tmpl='{:.5f}')}, avg_kpts_std: {list2str(avg_kpts_std, tmpl='{:.5f}')}\n")

        if self.verbose:
            print(log_str)

        if self.summary_path:
            with open(self.summary_path, 'a+') as fout:
                fout.write(log_str)

    def compute_stableness(self, kpts_list):
        kpts = npf(kpts_list)[..., :2]  # n x n_kpts x 2
        avg_kpts = np.mean(kpts, 0)  # n_kpts x 2
        kpts_std = np.mean(np.sum((kpts - avg_kpts) ** 2, -1), 0) ** 0.5  # n_kpts
        avg_kpts_std = kpts_std.mean()
        return kpts_std, avg_kpts_std

    def summarize(self):
        kpts_std_list = []
        avg_kpts_std_list = []
        for res in self.eval_results:
            kpts_std_list.append(res['kpts_std'])
            avg_kpts_std_list.append(res['avg_kpts_std'])
        kpts_std = npf(kpts_std_list).T  # n_kpts x n
        avg_kpts_std = npf(avg_kpts_std_list)  # n

        path_stem = osp.splitext(self.summary_path)[0]
        options = {'log': True}

        for i, x in enumerate(kpts_std):
            save_path = osp.join(path_stem + f'_stable_kpt{i}.jpg')
            plot_hist(x, 100, save_path, title=f'stable_kpt{i}', **options)

        save_path = osp.join(path_stem + f'_avg_stable.jpg')
        plot_hist(avg_kpts_std, 100, save_path, title=f'avg_stable', **options)

        per_kpt_avg_std = kpts_std.mean(1)
        all_kpt_avg_std = avg_kpts_std.mean()
        log_str = 'Stableness\n'
        log_str += "final_per_kpt_avg_std: \n"
        log_str += ',\n'.join([f'{KEYPOINT_NAMES[i]}: {a:.5f}' for i, a in enumerate(per_kpt_avg_std)])
        log_str += f"\nfinal_all_kpt_avg_std: {all_kpt_avg_std:.5f}"

        print(log_str)
        if self.summary_path:
            with open(self.summary_path, 'a+') as fout:
                fout.write(log_str + '\n')
