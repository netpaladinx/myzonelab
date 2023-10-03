import os.path as osp
from collections import OrderedDict
import random
import copy
import json

import numpy as np

from ....registry import DATASETS
from ....utils import get_if_is, get_if_eq, save_img
from ..base_dataset import BaseDataset
from .coco_data import COCOData
from .coco_utils import safe_bbox, bbox_center, bbox_scale, get_mask, size_tuple, npf
from .coco_consts import (KEYPOINT_NAMES, KEYPOINT_FLIP_PAIRS, KEYPOINT_UPPER_BODY, KEYPOINT_LOWER_BODY,
                          KEYPOINT_WEIGHTS, KEYPOINT_SIGMAS, SKELETON, BBOX_PADDING_RATIO)
from .coco_eval import COCOEval
from .coco_eval_ext import COCOEval_DataAnalysis, COCOEval_Visualization
from .coco_measure import COCOStableness
from .coco_visualize import draw_anns


@DATASETS.register_class('coco_pose')
class COCOPose(BaseDataset):
    def __init__(self, data_source=None, data_params=None, data_transforms=None, data_eval=None, name=None):
        super().__init__(data_source, data_params, data_transforms, data_eval, name)
        self.input_size = size_tuple(self.data_params['input_size'])
        self.input_channels = self.data_params['input_channels']
        self.heatmap_size = self.data_params['heatmap_size']
        self.heatmap_channels = len(KEYPOINT_NAMES)
        self.input_aspect_ratio = self.input_size[0] / self.input_size[1]
        self.bbox_padding_ratio = self.data_params.get('bbox_padding_ratio', BBOX_PADDING_RATIO)
        self.use_keypoint_weights = self.data_params.get('use_keypoint_weights', False)
        self.keypoint_weights = self.data_params.get('keypoint_weights', KEYPOINT_WEIGHTS)
        self.keypoint_sigmas = self.data_params.get('keypoint_sigmas', KEYPOINT_SIGMAS)
        self.use_segmentation_input = self.data_params.get('use_segmentation_input', False)
        self.segmentation_resolution = self.data_params.get('segmentation_resolution')
        self.eval_threshold = self.data_params.get('eval_threshold', 0.2)
        self.eval_multiple = self.data_params.get('eval_multiple', 1)
        self.eval_index = self.data_params.get('eval_index', None)
        self.keypoint_info = {
            'keypoints': KEYPOINT_NAMES,
            'num_keypoints': len(KEYPOINT_NAMES),
            'keypoint_name2id': OrderedDict([(i, name) for i, name in enumerate(KEYPOINT_NAMES)]),
            'upper_body_ids': KEYPOINT_UPPER_BODY,
            'lower_body_ids': KEYPOINT_LOWER_BODY,
            'flip_pairs_ids': KEYPOINT_FLIP_PAIRS
        }
        self.skeleton_info = {
            'links': SKELETON,
            'n_links': len(SKELETON)
        }

        self.coco_data = COCOData(**data_source)
        self.input_data = self.load_input_data()
        self.input_indices = list(range(len(self.input_data)))
        if self.shuffle:
            random.shuffle(self.input_indices)

        self.stableness = None

    def load_input_data(self):
        input_data = []
        for ann in self.coco_data.annotations:
            img_id = ann['image_id']
            img = self.coco_data.get_img(img_id)
            img_h, img_w = img.get('height'), img.get('width')
            bbox = ann.get('bbox')
            kpts = ann.get('keypoints')

            input_dict = {
                'ann_id': ann['id'].tid,
                'img_id': img_id.tid,
                'flip_pairs': self.keypoint_info['flip_pairs_ids']
            }

            if bbox is not None:
                bbox = safe_bbox(bbox, img_h, img_w)
                center = bbox_center(bbox)
                scale = bbox_scale(bbox, self.input_aspect_ratio, self.bbox_padding_ratio)
                input_dict['bbox'] = npf(bbox)
                input_dict['center'] = npf(center)
                input_dict['scale'] = npf(scale)

            if kpts is not None:
                input_dict['kpts'] = npf(kpts).reshape(-1, 3)
                if all(v == 0 for v in kpts[2::3]):
                    continue

            if self.use_segmentation_input:
                input_dict['mask'] = get_mask(ann, img, self.coco_data)

            input_data.append(input_dict)

        print(f'{len(input_data)} input items loaded (one item one annotation)')
        return input_data

    def get_unprocessed_item(self, idx):
        idx = self.input_indices[idx]
        input_dict = copy.deepcopy(self.input_data[idx])
        input_dict['img'] = self.coco_data.read_img(input_dict['img_id'])  # h x w x c
        return input_dict

    @property
    def image_ids(self):  # list(img_tid)
        return [img_id.tid for img_id in self.coco_data.get_img_ids()]

    @property
    def selected(self):  # set((img_tid, ann_tid))
        return set([(input_dict['img_id'], input_dict['ann_id']) for input_dict in self.input_data])

    def evaluate_step(self, results, batch, work_dir=None, summary_path=None, ctx=None):
        work_dir = get_if_is(ctx, 'work_dir', work_dir, None)
        summary_path = get_if_is(ctx, 'summary_path', summary_path, None)

        if self.eval_multiple > 1:
            if self.stableness is None:
                assert work_dir is not None and summary_path is not None
                summary_dir = osp.join(work_dir, 'stableness')
                summary_name = osp.splitext(osp.basename(summary_path))[0]
                self.stableness = COCOStableness(self.eval_multiple, summary_dir, summary_name)
            self.stableness.evaluate_step(results, batch)

    def summarize(self, **kwargs):
        if self.stableness:
            self.stableness.summarize()

    def _process_all_results(self, all_results):
        ann_results = []
        for res_dict in all_results:
            kpts_results = res_dict['kpts_results']
            bbox_results = res_dict['bbox_results']
            ann_ids = res_dict['ann_ids']

            for i, ann_id in enumerate(ann_ids):
                input_bbox = bbox_results[i][0:4].tolist()
                bbox_score = bbox_results[i][4].item()
                kpts = kpts_results[i].flatten().tolist()
                kpts_score = kpts_results[i][:, 2]
                kpts_score = kpts_score[kpts_score > self.eval_threshold]
                kpts_score_avg = kpts_score.mean().item() if len(kpts_score) > 0 else 0
                kpts_score_min = kpts_score.min().item() if len(kpts_score) > 0 else 0

                ann_results.append({
                    'id': ann_id,
                    'input_bbox': input_bbox,
                    'bbox_score': bbox_score,
                    'keypoints': kpts,
                    'keypoints_score': kpts_score_avg,
                    'keypoints_score_min': kpts_score_min,
                    'score': bbox_score * kpts_score_avg
                })
        return ann_results

    def _process_multiple_all_results(self, all_results):
        multiple = len([ann_id for ann_id in all_results[0]['ann_ids'] if ann_id == all_results[0]['ann_ids'][0]])
        ann_results_list = [[] for _ in range(multiple)]
        for res_dict in all_results:
            kpts_results = res_dict['kpts_results']
            bbox_results = res_dict['bbox_results']
            ann_ids = res_dict['ann_ids']
            id_set = OrderedDict([(tuple(ann_id), 1) for ann_id in ann_ids])
            assert len(ann_ids) == multiple * len(id_set)
            ann_ids = id_set.keys()

            for i, ann_id in enumerate(ann_ids):
                for j in range(multiple):
                    k = i * multiple + j
                    input_bbox = bbox_results[k][0:4].tolist()
                    bbox_score = bbox_results[k][4].item()
                    kpts = kpts_results[k].flatten().tolist()
                    kpts_score = kpts_results[k][:, 2]
                    kpts_score = kpts_score[kpts_score > self.eval_threshold]
                    kpts_score_avg = kpts_score.mean().item() if len(kpts_score) > 0 else 0
                    kpts_score_min = kpts_score.min().item() if len(kpts_score) > 0 else 0

                    ann_results_list[j].append({
                        'id': ann_id,
                        'input_bbox': input_bbox,
                        'bbox_score': bbox_score,
                        'keypoints': kpts,
                        'keypoints_score': kpts_score_avg,
                        'keypoints_score_min': kpts_score_min,
                        'score': bbox_score * kpts_score_avg
                    })
        return ann_results_list

    def _compute_average_over_multiple(self, ann_results_list):
        average_ann_results = []
        n_anns = len(ann_results_list[0])
        for i in range(n_anns):
            id = [ann_results[i]['id'] for ann_results in ann_results_list]
            input_bbox = [ann_results[i]['input_bbox'] for ann_results in ann_results_list]
            bbox_score = [ann_results[i]['bbox_score'] for ann_results in ann_results_list]
            kpts = [ann_results[i]['keypoints'] for ann_results in ann_results_list]
            kpts_score = [ann_results[i]['keypoints_score'] for ann_results in ann_results_list]
            score = [ann_results[i]['score'] for ann_results in ann_results_list]
            assert all(id_ == id[0] for id_ in id)
            input_bbox = npf(input_bbox).mean(0).tolist()
            bbox_score = npf(bbox_score).mean().item()
            kpts = npf(kpts).reshape(len(kpts), -1, 3)
            weight = kpts[..., 2:3].copy()
            weight[weight < np.median(weight, 0)] = 0
            kpts = (kpts * weight).sum(0) / np.maximum(weight.sum(0), np.spacing(1))
            kpts = kpts.flatten().tolist()
            kpts_score = npf(kpts_score).mean().item()
            score = npf(score).mean().item()
            average_ann_results.append({
                'id': id[0],
                'input_bbox': input_bbox,
                'bbox_score': bbox_score,
                'keypoints': kpts,
                'keypoints_score': kpts_score,
                'score': score
            })
        return average_ann_results

    def _evaluate_ann_results_impl(self, coco_eval, plot_dir, summary_path, analysis_dir, visualize_dir, select_policy, kpts_mask=None):
        coco_eval.evaluate(kpts_mask=kpts_mask)
        coco_eval.accumulate(plot_dir=plot_dir)
        results = coco_eval.summarize()['kpts']
        results = OrderedDict([(res['name'], res['mean_value']) for res in results])

        if summary_path:
            with open(summary_path, 'w') as fout:
                json.dump(results, fout)

        if analysis_dir:
            data_analysis = COCOEval_DataAnalysis(coco_eval, analysis_dir, score_name='confidence')
            data_analysis.analyze(metric_meta={'metric_range': (0, 1),
                                               'metric_unit': 0.001,
                                               'metric_name': 'oks',
                                               'log_counts': True},
                                  extra_metrics_meta={'avg_dist': {'metric_range': (0, 100),
                                                                   'metric_length': 100,
                                                                   'metric_name': 'avg_dist',
                                                                   'log_counts': True},
                                                      'max_dist': {'metric_range': (0, 100),
                                                                   'metric_length': 100,
                                                                   'metric_name': 'max_dist',
                                                                   'log_counts': True}})

        if visualize_dir:
            visualization = COCOEval_Visualization(coco_eval, visualize_dir, select_policy=select_policy)
            visualization.visualize()

        return results

    def _evaluate_ann_results(self, ann_results, plot_dir=None, summary_path=None, analysis_dir=None,
                              visualize_dir=None, select_policy=None, hard_level=0):
        coco_res = self.coco_data.from_results(ann_results, keep_ann_id=True)
        coco_eval = COCOEval(self.coco_data, coco_res, eval_kpts=True, keep_ann_id=True, gt_selected=self.selected, hard_level=hard_level)
        results = self._evaluate_ann_results_impl(coco_eval, plot_dir, summary_path, analysis_dir, visualize_dir, select_policy)
        extra_results = []
        for i, name in enumerate(KEYPOINT_NAMES):
            if summary_path:
                p0, p1 = osp.splitext(summary_path)
                s_path = f'{p0}_{name}{p1}'
            else:
                s_path = None
            p_dir = f'{plot_dir}_{name}' if plot_dir else None
            a_dir = f'{analysis_dir}_{name}' if analysis_dir else None
            v_dir = f'{visualize_dir}_{name}' if visualize_dir else None
            kpts_mask = np.zeros(len(KEYPOINT_NAMES))
            kpts_mask[i] = 1
            res = self._evaluate_ann_results_impl(coco_eval, p_dir, s_path, a_dir, v_dir, select_policy, kpts_mask=kpts_mask)
            res['keypoint'] = name
            extra_results.append(res)
        results['extra_results'] = extra_results
        return results

    def evaluate_all(self, all_results, plot_dir=None, summary_path=None, analysis_dir=None,
                     visualize_dir=None, select_policy=None, hard_level=0, ctx=None):
        plot_dir = get_if_is(ctx, 'plot_dir', plot_dir, None)
        summary_path = get_if_is(ctx, 'summary_path', summary_path, None)
        analysis_dir = get_if_is(ctx, 'analysis_dir', analysis_dir, None)
        visualize_dir = get_if_is(ctx, 'visualize_dir', visualize_dir, None)
        select_policy = get_if_is(ctx, 'select_policy', select_policy, None)
        hard_level = get_if_eq(ctx, 'eval_hard_level', hard_level, 0)

        # stableness evaluation
        if self.eval_multiple > 1:
            eval_results = {}
            ann_results_list = self._process_multiple_all_results(all_results)
            for i, ann_results in enumerate(ann_results_list):
                if self.eval_index is None or i == self.eval_index or i in self.eval_index:
                    print(f'===== {i} =====')
                    eval_results[i] = self._evaluate_ann_results(ann_results,
                                                                 plot_dir=plot_dir,
                                                                 summary_path=summary_path)

            average_ann_results = self._compute_average_over_multiple(ann_results_list)
            print(f'===== Average =====')
            eval_results['average'] = self._evaluate_ann_results(average_ann_results,
                                                                 plot_dir=plot_dir,
                                                                 summary_path=summary_path)

        # regular evaluation
        else:
            ann_results = self._process_all_results(all_results)
            eval_results = self._evaluate_ann_results(ann_results,
                                                      plot_dir=plot_dir,
                                                      summary_path=summary_path,
                                                      analysis_dir=analysis_dir,
                                                      visualize_dir=visualize_dir,
                                                      select_policy=select_policy,
                                                      hard_level=hard_level)
        return eval_results

    def dump_all(self, all_results, dump_dir=None, res_ann_file=None, ctx=None):
        dump_dir = get_if_is(ctx, 'dump_dir', dump_dir, None)
        res_ann_file = get_if_is(ctx, 'res_ann_file', res_ann_file, None)

        ann_results = self._process_all_results(all_results)
        coco_res = self.coco_data.from_results(ann_results, keep_ann_id=True)
        if res_ann_file:
            coco_res.dump_data(res_ann_file)

        for img in coco_res.images:
            img_id = img['id']
            file_name = img['file_name']
            anns = [coco_res.get_ann(ann_id) for ann_id in coco_res.get_ann_ids(img_ids=[img_id])]
            img = draw_anns(anns, img_id, coco_res, draw_bbox=False, draw_kpts=True, draw_seg=False)
            file_path = osp.join(dump_dir, file_name)
            save_img(img, file_path, use_cv2=True, convert=True)
