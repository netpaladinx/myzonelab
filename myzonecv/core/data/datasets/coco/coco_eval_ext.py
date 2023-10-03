import os.path as osp
import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import cv2

from ....utils import mkdir, list2str
from .coco_utils import bbox_img
from .coco_visualize import draw_anns


def to_numpy(a):
    if isinstance(a, np.ndarray):
        return a
    elif isinstance(a, (list, tuple)):
        return np.array(a, dtype=float)
    else:
        raise TypeError(f"Only accept np.ndarray, list or tuple, but got {type(a)}")


class COCOEval_DataAnalysis:
    def __init__(self, coco_eval, out_dir, score_name='score'):
        self.coco_eval = coco_eval
        self.out_dir = out_dir
        self.score_name = score_name
        self.eval_kpts_key = ((1,), 0, 0)

        mkdir(self.out_dir, exist_ok=True)

    def _process_eval_results(self, eval_results):
        img_tids = []
        ann_tids = []
        ann_scores = []
        ann_scores_min = []
        ann_metrics = []
        ann_extra_metrics = []
        for res in eval_results:
            img_tid = res['img_tid']
            for i, ann_tid in enumerate(res['res_ann_tids']):
                score = res['res_ann_scores'][i]
                score_min = res['res_ann_scores_min'][i]
                metric = res['metrics'][i]
                extra_metrics = {k: v[i] for k, v in res['extra_metrics'].items()} if res['extra_metrics'] else None

                img_tids.append(img_tid)
                ann_tids.append(ann_tid)
                ann_scores.append(score)
                ann_scores_min.append(score_min)
                ann_metrics.append(metric)
                ann_extra_metrics.append(extra_metrics)
        return {
            'img_tids': img_tids,
            'ann_tids': ann_tids,
            'ann_scores': ann_scores,
            'ann_scores_min': ann_scores_min,
            'ann_metrics': ann_metrics,
            'ann_extra_metrics': ann_extra_metrics
        }

    def _plot_scores_to_metrics(self,
                                scores,
                                scores_min,
                                metrics,
                                metric_range=None,
                                metric_unit=None,
                                metric_length=None,
                                metric_name=None,
                                log_counts=False):
        self._plot_heatmap2d(self.out_dir,
                             scores,
                             metrics,
                             range_x=(0, 1),
                             range_y=metric_range,
                             unit_x=0.001,
                             unit_y=metric_unit,
                             length_y=metric_length,
                             log_counts=log_counts,
                             label_x=self.score_name,
                             label_y=metric_name,
                             name=f'{self.score_name} to {metric_name}')
        self._plot_heatmap2d(self.out_dir,
                             scores_min,
                             metrics,
                             range_x=(0, 1),
                             range_y=metric_range,
                             unit_x=0.001,
                             unit_y=metric_unit,
                             length_y=metric_length,
                             log_counts=log_counts,
                             label_x=f'min_{self.score_name}',
                             label_y=metric_name,
                             name=f'min_{self.score_name} to {metric_name}')

    def analyze(self, metric_meta, extra_metrics_meta=None):
        if self.coco_eval.eval_kpts and self.coco_eval.kpts_eval_results:
            eval_results = self.coco_eval.kpts_eval_results[self.eval_kpts_key]
            items_results = self._process_eval_results(eval_results)

            self._plot_scores_to_metrics(items_results['ann_scores'],
                                         items_results['ann_scores_min'],
                                         items_results['ann_metrics'],
                                         metric_range=metric_meta.get('metric_range'),
                                         metric_unit=metric_meta.get('metric_unit'),
                                         metric_length=metric_meta.get('metric_length'),
                                         metric_name=metric_meta.get('metric_name', 'metric'),
                                         log_counts=metric_meta.get('log_counts', False))

            if extra_metrics_meta:
                for metric_name in extra_metrics_meta.keys():
                    if metric_name not in items_results['ann_extra_metrics'][0]:
                        continue

                    item_metrics = [res[metric_name] for res in items_results['ann_extra_metrics']]

                    self._plot_scores_to_metrics(items_results['ann_scores'],
                                                 items_results['ann_scores_min'],
                                                 item_metrics,
                                                 metric_range=extra_metrics_meta[metric_name].get('metric_range'),
                                                 metric_unit=extra_metrics_meta[metric_name].get('metric_unit'),
                                                 metric_length=extra_metrics_meta[metric_name].get('metric_length'),
                                                 metric_name=extra_metrics_meta[metric_name].get('metric_name', 'metric'),
                                                 log_counts=extra_metrics_meta[metric_name].get('log_counts', False))

    @staticmethod
    def _plot_heatmap2d(out_dir, values_x, values_y,
                        range_x=None, range_y=None,
                        unit_x=None, unit_y=None,
                        length_x=None, length_y=None,
                        log_counts=False,
                        label_x=None, label_y=None,
                        name='heatmap2d'):
        values_x = to_numpy(values_x)
        values_y = to_numpy(values_y)
        assert len(values_x) == len(values_y)
        min_x, max_x = values_x.min(), values_x.max()
        min_y, max_y = values_y.min(), values_y.max()

        filtered = np.ones(len(values_x)).astype(np.bool_)
        if range_x is None:
            range_x = (min_x, max_x)
        else:
            filtered = filtered & (values_x >= range_x[0]) & (values_x <= range_x[1])
        if range_y is None:
            range_y = (min_y, max_y)
        else:
            filtered = filtered & (values_y >= range_y[0]) & (values_y <= range_y[1])

        values_x = values_x[filtered]
        values_y = values_y[filtered]
        min_x, max_x, avg_x, std_x = values_x.min(), values_x.max(), values_x.mean(), values_x.std()
        min_y, max_y, avg_y, std_y = values_y.min(), values_y.max(), values_y.mean(), values_y.std()

        if unit_x is None:
            length_x = length_x or 10
            unit_x = (range_x[1] - range_x[0]) / length_x
        else:
            length_x = int((range_x[1] - range_x[0]) / unit_x)
            remainder = (range_x[1] - range_x[0]) - unit_x * length_x
            if remainder > 0:
                pad = (unit_x - remainder) / 2
                range_x = (range_x[0] - pad, range_x[1] + pad)
        if unit_y is None:
            length_y = length_y or 10
            unit_y = (range_y[1] - range_y[0]) / length_y
        else:
            length_y = int((range_y[1] - range_y[0]) / unit_y)
            remainder = (range_y[1] - range_y[0]) - unit_y * length_y
            if remainder > 0:
                pad = (unit_y - remainder) / 2
                range_y = (range_y[0] - pad, range_y[1] + pad)
                length_y += 1

        indices_x = ((values_x - range_x[0]) / unit_x).astype(np.int32)
        indices_y = ((values_y - range_y[0]) / unit_y).astype(np.int32)
        indices = np.stack([indices_x, indices_y], axis=-1)  # indices[i]: (xi, yi)
        indices, counts = np.unique(indices, return_counts=True, axis=0)
        if log_counts:
            counts = np.log(counts + 1)

        grid_y, grid_x = np.meshgrid(np.linspace(range_y[0], range_y[1], length_y + 1),
                                     np.linspace(range_x[0], range_x[1], length_x + 1))
        grid_c = np.zeros((length_x + 1, length_y + 1))
        grid_c[indices[:, 0], indices[:, 1]] = counts
        vmin, vmax = counts.min(), counts.max()

        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        c = ax.pcolormesh(grid_x, grid_y, grid_c, cmap='Blues', vmin=vmin, vmax=vmax)
        ax.set_title(name + f'\n({label_x}: {avg_x:.5f}$\pm${std_x:.5f}, {label_y}: {avg_y:.5f}$\pm${std_y:.5f})')
        if label_x:
            ax.set_xlabel(label_x)
        if label_y:
            ax.set_ylabel(label_y)
        fig.colorbar(c, ax=ax)
        fig.savefig(osp.join(out_dir, f'{name}.png'), dpi=250)
        plt.close()


class COCOEval_Visualization:
    _default_random_policy = {(0, 0.8): 20,
                              (0.8, 0.85): 40,
                              (0.85, 0.9): 100,
                              (0.9, 0.95): 30,
                              (0.95, 1): 10}

    _default_worst_policy = {(0, 0.8): -1}

    def __init__(self, coco_eval, out_dir, select_policy=None):
        self.coco_eval = coco_eval
        self.out_dir = out_dir
        self.eval_kpts_key = ((1,), 0, 0)

        select_policy = select_policy or 'random_select'
        if isinstance(select_policy, str):
            assert select_policy in ('worst_select', 'random_select')
            if select_policy == 'worst_select':
                self.select_policy = self._default_worst_policy
            elif select_policy == 'random_select':
                self.select_policy = self._default_random_policy

        mkdir(self.out_dir, exist_rm=True)

    def _process_eval_results(self, eval_results):
        results = []
        for res in eval_results:
            img_tid = res['img_tid']
            gt_ann_tids = res['gt_ann_tids']
            res_ann_tids = res['res_ann_tids']
            metrics = res['metrics']
            extra_metrics = res['extra_metrics']
            assert len(gt_ann_tids) == len(res_ann_tids) == len(metrics)
            assert all(g == r for g, r in zip(gt_ann_tids, res_ann_tids))

            dct = {
                'img_tid': img_tid,
                'ann_tids': [],
                'gt_anns': [],
                'res_anns': [],
                'metrics': [],
                'extra_metrics': defaultdict(list),
                'min_metric': -1,
            }

            for i, ann_tid in enumerate(gt_ann_tids):
                metric = metrics[i]
                gt_ann = self.coco_eval.coco_gt.get_ann(ann_tid)
                res_ann = self.coco_eval.coco_res.get_ann(ann_tid)

                dct['ann_tids'].append(ann_tid)
                dct['gt_anns'].append(gt_ann)
                dct['res_anns'].append(res_ann)
                dct['metrics'].append(metric)

                for k, v in extra_metrics.items():
                    dct['extra_metrics'][k].append(v[i])

                if dct['min_metric'] == -1 or dct['min_metric'] > metric:
                    dct['min_metric'] = metric

            results.append(dct)

        return results

    @staticmethod
    def _random_select(items_results, select_policy):
        new_items_results = []
        for rng, n in select_policy.items():
            items = [it for it in items_results if rng[0] <= it['min_metric'] < rng[1]]
            random.shuffle(items)
            new_items_results.extend(items[:n] if n >= 0 else items)
        return new_items_results

    @staticmethod
    def _imwrite(path, image, convert=False):
        try:
            if convert:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, image)
        except Exception:
            print(f"Error occurred when processing image (shape: {image.shape}) to path '{path}'")
            pass

    def _draw_gt_and_res(self, item_res):
        coco_data = self.coco_eval.coco_gt
        img_tid = item_res['img_tid']
        ann_tids = item_res['ann_tids']
        gt_anns = item_res['gt_anns']
        res_anns = item_res['res_anns']
        metrics = item_res['metrics']
        min_metric = item_res['min_metric']

        for ann_tid, gt_ann, res_ann, metric in zip(ann_tids, gt_anns, res_anns, metrics):
            bbox = gt_ann['bbox']

            img_gt = draw_anns([gt_ann], img_tid, coco_data, draw_bbox=False, draw_kpts=True, draw_seg=False)
            img_gt = bbox_img(img_gt, bbox)

            img_res = draw_anns([res_ann], img_tid, coco_data, draw_bbox=False, draw_kpts=True, draw_seg=False)
            img_res = bbox_img(img_res, bbox)

            img = np.concatenate([img_gt, img_res], axis=1)
            file_name = f'{min_metric:.5f}_{metric:.5f}_{list2str(img_tid)}_{list2str(ann_tid)}_gt-pred.jpg'
            file_path = osp.join(self.out_dir, file_name)
            self._imwrite(file_path, img, convert=True)

    def visualize(self):
        if self.coco_eval.eval_kpts and self.coco_eval.kpts_eval_results:
            eval_results = self.coco_eval.kpts_eval_results[self.eval_kpts_key]
            items_results = self._process_eval_results(eval_results)

            items_results = self._random_select(items_results, self.select_policy)

            for item_res in items_results:
                self._draw_gt_and_res(item_res)
