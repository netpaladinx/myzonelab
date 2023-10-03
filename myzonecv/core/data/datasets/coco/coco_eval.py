import os.path as osp
from collections import defaultdict, OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from . import mask as mask_utils
from .coco_utils import get_rle, npf
from .coco_consts import (KEYPOINT_SIGMAS, MAX_DETECTIONS_PER_IMG,
                          EVAL_AREA_LABELS, EVAL_AREA_RANGES, EVAL_SCORE_THRES, EVAL_RECALL_THRES, EVAL_OKS_HARD_FACTORS)


class COCOEval:
    def __init__(self, coco_gt, coco_res,
                 eval_bbox=False, eval_seg=False, eval_kpts=False,
                 keep_ann_id=False, gt_selected=None, hard_level=0, ignore_category=False):
        self.coco_gt = coco_gt
        self.coco_res = coco_res
        self.eval_bbox = eval_bbox
        self.eval_seg = eval_seg
        self.eval_kpts = eval_kpts
        self.keep_ann_id = keep_ann_id
        self.gt_selected = gt_selected  # set((img_tid, ann_tid))
        self.hard_level = hard_level
        self.ignore_category = ignore_category

        self.area_ranges = OrderedDict([(area_label, area_rng) for area_label, area_rng in zip(EVAL_AREA_LABELS, EVAL_AREA_RANGES)])
        self.max_detections = MAX_DETECTIONS_PER_IMG
        self.score_thres = EVAL_SCORE_THRES
        self.recall_thres = EVAL_RECALL_THRES

        self.img_tids = None  # intersection of img tids between coco_gt and coco_res
        self.cat_tids = None  # intersection of cat tids between coco_gt and coco_res
        self.gt_anns = defaultdict(list)  # (cat_tid, img_tid) => list(ann)
        self.res_anns = defaultdict(list)  # (cat_tid, img_tid) => list(ann

        self.bbox_eval_results = None
        self.seg_eval_results = None
        self.kpts_eval_results = None
        self.bbox_accu_results = None
        self.seg_accu_results = None
        self.kpts_accu_results = None
        self.bbox_summary = None
        self.seg_summary = None
        self.kpts_summary = None

        self.prepare()

    def prepare(self):
        if self.keep_ann_id:
            for key in self.max_detections:
                self.max_detections[key] = [-1]

        self.img_tids = sorted(set(self.coco_gt.img_id_reg.tids) & set(self.coco_res.img_id_reg.tids))

        if not self.ignore_category:
            self.cat_tids = sorted(set(self.coco_gt.cat_id_reg.tids) & set(self.coco_res.cat_id_reg.tids))
            gt_anns = self.coco_gt.get_anns(self.coco_gt.get_ann_ids(self.img_tids, self.cat_tids))
            res_anns = self.coco_res.get_anns(self.coco_res.get_ann_ids(self.img_tids, self.cat_tids))
        else:
            self.cat_tids = [(1,)]  # person category
            gt_anns = self.coco_gt.get_anns(self.coco_gt.get_ann_ids(self.img_tids))
            res_anns = self.coco_res.get_anns(self.coco_res.get_ann_ids(self.img_tids))

        for ann in gt_anns:
            img_tid = ann['image_id'].tid
            cat_tid = ann['category_id'].tid if not self.ignore_category else (1,)
            if self.gt_selected:
                ann_tid = ann['id'].tid
                if (img_tid, ann_tid) not in self.gt_selected:
                    continue
            self.gt_anns[cat_tid, img_tid].append(ann)
        for ann in res_anns:
            img_tid = ann['image_id'].tid
            cat_tid = ann['category_id'].tid if not self.ignore_category else (1,)
            self.res_anns[cat_tid, img_tid].append(ann)

        for cat_tid in self.cat_tids:
            for img_tid in self.img_tids:
                gt_anns = self.gt_anns[cat_tid, img_tid]    # empty list if not exists
                res_anns = self.res_anns[cat_tid, img_tid]  # empty list if not exists

                if self.keep_ann_id:
                    gt_ann_tids = [ann['id'].tid for ann in gt_anns]
                    res_ann_tids = [ann['id'].tid for ann in res_anns]
                    ann_tids = set(gt_ann_tids) & set(res_ann_tids)
                    if len(ann_tids) == 0:
                        del self.gt_anns[cat_tid, img_tid]
                        del self.res_anns[cat_tid, img_tid]
                    else:
                        self.gt_anns[cat_tid, img_tid] = sorted([ann for ann in gt_anns if ann['id'].tid in ann_tids],
                                                                key=lambda a: a['id'].tid)
                        self.res_anns[cat_tid, img_tid] = sorted([ann for ann in res_anns if ann['id'].tid in ann_tids],
                                                                 key=lambda a: a['id'].tid)

    @staticmethod
    def get_bbox_score(ann, default=None, score_type=None):
        suffix = '_min' if score_type == 'min' else ''
        return ann.get(f'bbox_score{suffix}', ann.get(f'score{suffix}', default))

    @staticmethod
    def get_seg_score(ann, default=None, score_type=None):
        suffix = '_min' if score_type == 'min' else ''
        return ann.get(f'segmentation_score{suffix}', ann.get(f'score{suffix}', default))

    @staticmethod
    def get_kpts_score(ann, default=None, score_type=None):
        suffix = '_min' if score_type == 'min' else ''
        return ann.get(f'keypoints_score{suffix}', ann.get(f'score{suffix}', default))

    def get_score(self, ann, eval_type=None, default=None, score_type=None):
        if eval_type == 'bbox':
            return self.get_bbox_score(ann, default, score_type)
        elif eval_type == 'seg':
            return self.get_seg_score(ann, default, score_type)
        elif eval_type == 'kpts':
            return self.get_kpts_score(ann, default, score_type)

        suffix = '_min' if score_type == 'min' else ''
        return ann.get(f'score{suffix}', default)

    @staticmethod
    def compute_oks(gt_anns, res_anns, hard_factor=1., return_dist=False, kpts_mask=None):
        vars = (npf(KEYPOINT_SIGMAS) * 2)**2
        avg_dist, max_dist = [], []
        oks = []
        for gt_ann in gt_anns:
            gt_kpts = npf(gt_ann['keypoints'])
            gt_x, gt_y, gt_v = gt_kpts[0::3], gt_kpts[1::3], gt_kpts[2::3]
            if kpts_mask is not None:
                gt_v = gt_v * kpts_mask
            gt_n_kpts = np.count_nonzero(gt_v > 0)

            for res_ann in res_anns:
                res_kpts = npf(res_ann['keypoints'])
                res_x, res_y, res_v = res_kpts[0::3], res_kpts[1::3], res_kpts[2::3]
                if kpts_mask is not None:
                    res_v = res_v * kpts_mask
                res_n_kpts = np.count_nonzero(res_v > 0)
                if gt_n_kpts == 0 or res_n_kpts == 0:
                    avg_dist.append(-1)
                    max_dist.append(-1)
                    oks.append(-1)
                else:
                    dx = gt_x - res_x
                    dy = gt_y - res_y

                    d = (dx**2 + dy**2)**0.5
                    d = d[gt_v > 0]
                    avg_dist.append(d.mean())
                    max_dist.append(d.max())

                    e = (dx**(2 * hard_factor) + dy**(2 * hard_factor)) / vars / (gt_ann['area'] + np.spacing(1)) / 2.
                    e = e[gt_v > 0]
                    s = np.sum(np.exp(-e)) / len(e)
                    oks.append(s)
        osk = npf(oks).reshape(len(gt_anns), -1)  # np.ndarray (n_gt, n_res)
        avg_dist = npf(avg_dist).reshape(len(gt_anns), -1)
        max_dist = npf(max_dist).reshape(len(gt_anns), -1)
        if return_dist:
            return osk, (avg_dist, max_dist)
        else:
            return osk

    def evaluate_single(self, eval_type, cat_tid, img_tid, max_dets_idx, keep_ann_id=False, **kwargs):
        assert eval_type in ('bbox', 'seg', 'kpts')
        gt_anns = self.gt_anns[cat_tid, img_tid]
        res_anns = self.res_anns[cat_tid, img_tid]
        max_dets = self.max_detections[eval_type][max_dets_idx]

        if len(gt_anns) == 0 or len(res_anns) == 0:
            return

        if all([self.get_score(ann, eval_type) is not None for ann in res_anns]):
            res_anns = sorted(res_anns, key=lambda a: -self.get_score(a, eval_type))

        if max_dets > 0 and max_dets < len(res_anns):
            res_anns = res_anns[:max_dets]

        if keep_ann_id:
            gt_tids = set([gt_ann['id'].tid for gt_ann in gt_anns])
            res_tids = set([res_ann['id'].tid for res_ann in res_anns])
            assert gt_tids == res_tids
            tid2order = {res_ann['id'].tid: i for i, res_ann in enumerate(res_anns)}
            gt_anns = sorted([gt_ann for gt_ann in gt_anns], key=lambda a: tid2order[a['id'].tid])

        if eval_type == 'seg':
            ious = []
            if keep_ann_id:
                for gt_ann, res_ann in zip(gt_anns, res_anns):
                    gt_rles = [get_rle(gt_ann, self.coco_gt)]
                    res_rles = [get_rle(res_ann, self.coco_res)]
                    iou = mask_utils.iou(gt_rles, res_rles, is_rle=True)[0, 0]
                    ious.append(iou)
                ious = npf(ious)  # np.ndarray (n_gt)
            else:
                gt_rles = [get_rle(ann, self.coco_gt) for ann in gt_anns]
                res_rles = [get_rle(ann, self.coco_res) for ann in res_anns]
                ious = mask_utils.iou(gt_rles, res_rles, is_rle=True)
                ious = np.ascontiguousarray(ious)  # np.ndarray (n_gt x n_res)
            metrics = ious
            extra_metrics = {}

        elif eval_type == 'bbox':
            ious = []
            if keep_ann_id:
                for gt_ann, res_ann in zip(gt_anns, res_anns):
                    gt_rles = [gt_ann['bbox'][:4]]
                    res_rles = [res_ann['bbox'][:4]]
                    iou = mask_utils.iou(gt_rles, res_rles, is_rle=False)[0, 0]
                    ious.append(iou)
                ious = npf(ious)  # np.ndarray (n_gt)
            else:
                gt_rles = [ann['bbox'][:4] for ann in gt_anns]
                res_rles = [ann['bbox'][:4] for ann in res_anns]
                ious = mask_utils.iou(gt_rles, res_rles, is_rle=False)
                ious = np.ascontiguousarray(ious)  # np.ndarray (n_gt x n_res)
            metrics = ious
            extra_metrics = {}

        elif eval_type == 'kpts':
            oks, avg_dist, max_dist = [], [], []
            kpts_mask = kwargs.get('kpts_mask')
            gt_anns = [gt_ann for gt_ann in gt_anns if np.count_nonzero(npf(gt_ann['keypoints'][2::3]) > 0) > 0]
            if keep_ann_id:
                assert len(gt_anns) == len(res_anns)
                for gt_ann, res_ann in zip(gt_anns, res_anns):
                    o, (a_d, m_d) = self.compute_oks([gt_ann], [res_ann],
                                                     hard_factor=EVAL_OKS_HARD_FACTORS[self.hard_level], return_dist=True, kpts_mask=kpts_mask)
                    oks.append(o[0, 0])
                    avg_dist.append(a_d[0, 0])
                    max_dist.append(m_d[0, 0])
                oks = npf(oks)  # np.ndarray (n_gt)
                avg_dist = npf(avg_dist)
                max_dist = npf(max_dist)
            else:
                oks, (avg_dist, max_dist) = self.compute_oks(gt_anns, res_anns,
                                                             hard_factor=EVAL_OKS_HARD_FACTORS[self.hard_level], return_dist=True, kpts_mask=kpts_mask)  # np.ndarray (n_gt x n_res)
            metrics = oks
            extra_metrics = {'avg_dist': avg_dist, 'max_dist': max_dist}

        for area_rng_idx, (area_rng_label, area_rng) in enumerate(self.area_ranges.items()):
            picked_gt_indices = [i for i, ann in enumerate(gt_anns) if area_rng[0] <= ann['area'] < area_rng[1]]
            picked_res_indices = picked_gt_indices if keep_ann_id else [
                i for i, ann in enumerate(res_anns)
                if ('area' in ann and area_rng[0] <= ann['area'] < area_rng[1]) or 'area' not in ann]

            picked_gt_anns = [gt_anns[i] for i in picked_gt_indices]
            picked_res_anns = [res_anns[i] for i in picked_res_indices]

            result = {
                'img_tid': img_tid,
                'cat_tid': cat_tid,
                'area_rng_idx': area_rng_idx,
                'area_rng_label': area_rng_label,
                'area_rng': area_rng,
                'max_dets_idx': max_dets_idx,
                'max_dets': max_dets,
                'gt_ann_tids': [gt_ann['id'].tid for gt_ann in picked_gt_anns],
                'res_ann_tids': [res_ann['id'].tid for res_ann in picked_res_anns],
                'res_ann_scores': [self.get_score(res_ann, eval_type, default=0.) for res_ann in picked_res_anns],
                'res_ann_scores_min': [self.get_score(res_ann, eval_type, default=0., score_type='min') for res_ann in picked_res_anns],
                'metrics': None,
                'extra_metrics': None,
                'gt_match_by_gid': None,
                'res_match_by_gid': None
            }

            if picked_gt_indices and picked_res_indices:
                picked_metrics = metrics[picked_gt_indices] if keep_ann_id else metrics[picked_gt_indices][:, picked_res_indices]
                result['metrics'] = picked_metrics

                if extra_metrics:
                    picked_extra_metrics = {name: ext_met[picked_gt_indices] if keep_ann_id else ext_met[picked_gt_indices][:, picked_res_indices]
                                            for name, ext_met in extra_metrics.items()}
                    result['extra_metrics'] = picked_extra_metrics

                match_results = self.match_annotations(picked_metrics, picked_gt_anns, picked_res_anns, keep_ann_id=keep_ann_id)
                result.update(match_results)

            yield result

    def match_annotations(self, metrics, gt_anns, res_anns, keep_ann_id=False):
        n_thr = len(self.score_thres)
        n_gt = len(gt_anns)
        n_res = len(res_anns)
        gt_match = np.ones((n_thr, n_gt)) * -1
        res_match = np.ones((n_thr, n_res)) * -1

        for i, thr in enumerate(self.score_thres):
            for res_idx, res_ann in enumerate(res_anns):
                cur_best = min(thr, 1 - 1e-10)
                cur_best_gt_idx = -1

                if keep_ann_id:
                    gt_idx = res_idx
                    if cur_best <= metrics[gt_idx]:
                        cur_best = metrics[gt_idx]
                        cur_best_gt_idx = gt_idx
                else:
                    for gt_idx, _ in enumerate(gt_anns):
                        if gt_match[i, gt_idx] >= 0:
                            continue
                        if cur_best > metrics[gt_idx, res_idx]:
                            continue
                        cur_best = metrics[gt_idx, res_idx]
                        cur_best_gt_idx = gt_idx

                if cur_best_gt_idx == -1:
                    continue

                res_match[i, res_idx] = gt_anns[cur_best_gt_idx]['id'].gid
                gt_match[i, cur_best_gt_idx] = res_ann['id'].gid

        return {
            'gt_match_by_gid': gt_match,   # n_thr x n_gt_of_this_img
            'res_match_by_gid': res_match  # n_thr x n_res_of_this_img
        }

    def evaluate(self, **kwargs):
        if self.eval_bbox:
            self.bbox_eval_results = defaultdict(list)
            for cat_tid in self.cat_tids:
                for img_tid in self.img_tids:
                    for max_dets_idx in range(len(self.max_detections['bbox'])):
                        for result in self.evaluate_single('bbox', cat_tid, img_tid, max_dets_idx, keep_ann_id=self.keep_ann_id, **kwargs):
                            area_rng_idx = result['area_rng_idx']
                            self.bbox_eval_results[cat_tid, area_rng_idx, max_dets_idx].append(result)

        if self.eval_seg:
            self.seg_eval_results = defaultdict(list)
            for cat_tid in self.cat_tids:
                for img_tid in self.img_tids:
                    for max_dets_idx in range(len(self.max_detections['seg'])):
                        for result in self.evaluate_single('seg', cat_tid, img_tid, max_dets_idx, keep_ann_id=self.keep_ann_id, **kwargs):
                            area_rng_idx = result['area_rng_idx']
                            self.seg_eval_results[cat_tid, area_rng_idx, max_dets_idx].append(result)

        if self.eval_kpts:
            self.kpts_eval_results = defaultdict(list)
            for cat_tid in self.cat_tids:
                for img_tid in self.img_tids:
                    for max_dets_idx in range(len(self.max_detections['kpts'])):
                        for result in self.evaluate_single('kpts', cat_tid, img_tid, max_dets_idx, keep_ann_id=self.keep_ann_id, **kwargs):
                            area_rng_idx = result['area_rng_idx']
                            self.kpts_eval_results[cat_tid, area_rng_idx, max_dets_idx].append(result)

    def accumulate_results(self, eval_type, results, plot_dir=None):
        n_scores = len(self.score_thres)
        n_recall = len(self.recall_thres)
        n_cats = len(self.cat_tids)
        n_arngs = len(self.area_ranges)
        n_maxdets = len(self.max_detections[eval_type])
        precisions = -np.ones((n_cats, n_arngs, n_maxdets, n_scores, n_recall))  # n_cats x n_arngs x n_maxdets x n_scores x n_recall
        recalls = -np.ones((n_cats, n_arngs, n_maxdets, n_scores))               # n_cats x n_arngs x n_maxdets x n_scores
        scores = -np.ones((n_cats, n_arngs, n_maxdets, n_scores, n_recall))      # n_cats x n_arngs x n_maxdets x n_scores x n_recall

        if plot_dir:
            n_plot_pnts = 1000
            plot_px = np.linspace(0, 1, n_plot_pnts)
            plot_py = np.zeros((n_cats, n_plot_pnts))
            plot_pr = np.zeros((n_cats, n_plot_pnts))
            plot_rc = np.zeros((n_cats, n_plot_pnts))
            plot_f1 = np.zeros((n_cats, n_plot_pnts))
            plot_ap = np.zeros((n_cats, n_scores))

        for cat_i, cat_tid in enumerate(self.cat_tids):
            for area_rng_idx in range(n_arngs):
                for max_dets_idx in range(n_maxdets):
                    max_dets = self.max_detections[eval_type][max_dets_idx]
                    res_imgs = results[cat_tid, area_rng_idx, max_dets_idx]
                    res_imgs = [res for res in res_imgs if res['res_match_by_gid'] is not None]
                    if res_imgs:
                        res_scores = np.concatenate([res['res_ann_scores'][:max_dets] if max_dets > 0 else res['res_ann_scores']
                                                     for res in res_imgs])  # n_all_res (trunc by max_dets)
                        sorted_indices = np.argsort(-res_scores, kind='mergesort')
                        sorted_res_scores = res_scores[sorted_indices]
                        res_match = np.concatenate([res['res_match_by_gid'][:, :max_dets] if max_dets > 0 else res['res_match_by_gid']
                                                   for res in res_imgs], axis=1)[:, sorted_indices]  # n_thr x n_all_res (sorted by score in axis 1)
                        n_gt = sum([len(res['gt_ann_tids']) for res in res_imgs])

                        tps = res_match >= 0
                        fps = res_match < 0
                        tp_sum = np.cumsum(tps, axis=1).astype(np.float32)  # n_thr x n_all_res
                        fp_sum = np.cumsum(fps, axis=1).astype(np.float32)  # n_thr x n_all_res

                        for score_i, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                            n_res = len(tp)
                            rc = tp / (n_gt + np.spacing(1))     # n_all_res
                            pr = tp / (fp + tp + np.spacing(1))  # n_all_res

                            if plot_dir and area_rng_idx == 0 and max_dets_idx == 0:
                                pl_pr = np.interp(-plot_px, -sorted_res_scores, pr, left=1)
                                pl_rc = np.interp(-plot_px, -sorted_res_scores, rc, left=0)
                                pl_f1 = 2 * pl_pr * pl_rc / (pl_pr + pl_rc + np.spacing(0))
                                pl_ap, pl_pr2, pl_rc2 = compute_avg_precision(rc, pr)
                                pl_py = np.interp(plot_px, pl_rc2, pl_pr2)
                                plot_ap[cat_i, score_i] = pl_ap
                                if score_i == 0:  # only record mAP@0.5
                                    plot_py[cat_i] = pl_py
                                    plot_pr[cat_i] = pl_pr
                                    plot_rc[cat_i] = pl_rc
                                    plot_f1[cat_i] = pl_f1

                            pr = pr.tolist()
                            for i in range(n_res - 1, 0, -1):
                                if pr[i] > pr[i - 1]:
                                    pr[i - 1] = pr[i]

                            recalls[cat_i, area_rng_idx, max_dets_idx, score_i] = rc[-1] if n_res else 0

                            rec_precision = np.zeros((n_recall,), dtype=np.float32)
                            rec_score = np.zeros((n_recall,), dtype=np.float32)
                            indices = np.searchsorted(rc, npf(self.recall_thres, dtype=np.float32), side='left')
                            for ri, pi in enumerate(indices):
                                if pi < len(pr):
                                    rec_precision[ri] = pr[pi]
                                    rec_score[ri] = sorted_res_scores[pi]

                            precisions[cat_i, area_rng_idx, max_dets_idx, score_i, :] = rec_precision
                            scores[cat_i, area_rng_idx, max_dets_idx, score_i, :] = rec_score

        if plot_dir:
            plot_precision_recall_curve(plot_px, plot_py, plot_ap, plot_dir)
            plot_metric_confidence_curve(plot_px, plot_f1, plot_dir, ylabel='F1')
            plot_metric_confidence_curve(plot_px, plot_pr, plot_dir, ylabel='Precision')
            plot_metric_confidence_curve(plot_px, plot_rc, plot_dir, ylabel='Recall')

        return {
            'n_score_thres': n_scores,
            'n_recall_thres': n_recall,
            'n_cat_tids': n_cats,
            'n_area_ranges': n_arngs,
            'n_max_dets': n_maxdets,
            'precisions': precisions,
            'recalls': recalls,
            'scores': scores
        }

    def accumulate(self, **kwargs):
        results = {}
        if self.eval_bbox and self.bbox_eval_results:
            self.bbox_accu_results = self.accumulate_results('bbox', self.bbox_eval_results, **kwargs)
            results['bbox'] = self.bbox_accu_results
        if self.eval_seg and self.seg_eval_results:
            self.seg_accu_results = self.accumulate_results('seg', self.seg_eval_results, **kwargs)
            results['seg'] = self.seg_accu_results
        if self.eval_kpts and self.kpts_eval_results:
            self.kpts_accu_results = self.accumulate_results('kpts', self.kpts_eval_results, **kwargs)
            results['kpts'] = self.kpts_accu_results
        return results

    def summarize_mean_result(self, eval_type, accu_results, ap_or_ar, score_type, score_thr=None, area_label=None, max_dets=None, name=None):
        title = 'Average Precision' if ap_or_ar == 'ap' else 'Average Recall'
        score = '{:0.2f}:{:0.2f}'.format(self.score_thres[0], self.score_thres[-1]) if score_thr is None else '{:0.2f}'.format(score_thr)

        area_rng_indices = [i for i, alab in enumerate(self.area_ranges.keys()) if alab == area_label or area_label is None]
        max_dets_indices = [i for i, mdet in enumerate(self.max_detections[eval_type]) if mdet == max_dets or max_dets is None]

        if ap_or_ar == 'ap':
            # n_cats x n_arngs x n_maxdets x n_scores x n_recall => n_scores x n_recall x n_cats x n_arngs x n_maxdets
            res = accu_results['precisions'].transpose((3, 4, 0, 1, 2))
            if score_thr is not None:
                res = res[np.where(score_thr == npf(self.score_thres))[0]]
            res = res[..., area_rng_indices, max_dets_indices]
        else:
            # n_cats x n_arngs x n_maxdets x n_scores => n_scores x n_cats x n_arngs x n_maxdets
            res = accu_results['recalls'].transpose((3, 0, 1, 2))
            if score_thr is not None:
                res = res[np.where(score_thr == npf(self.score_thres))[0]]
            res = res[..., area_rng_indices, max_dets_indices]

        if len(res[res > -1]) == 0:
            mean_value = -1
        else:
            mean_value = np.mean(res[res > -1])
        result_str = '{:<18} @[ {}={:<9} | area={:>6s} | max_dets={:>3d} ] = {:0.3f}'.format(
            title, score_type, score, area_label, max_dets, mean_value)
        print(result_str)
        return {'mean_value': mean_value, 'result': result_str, 'name': name}

    def summarize(self):
        results = defaultdict(list)
        if self.eval_bbox and self.bbox_accu_results:
            max_dets = self.max_detections['bbox'][0]
            self.bbox_summary = []
            self.bbox_summary.append(self.summarize_mean_result('bbox', self.bbox_accu_results, 'ap', 'iou', 0.5, 'all', max_dets, 'AP.5'))
            self.bbox_summary.append(self.summarize_mean_result('bbox', self.bbox_accu_results, 'ap', 'iou', 0.75, 'all', max_dets, 'AP.75'))
            self.bbox_summary.append(self.summarize_mean_result('bbox', self.bbox_accu_results, 'ap', 'iou', 0.9, 'all', max_dets, 'AP.9'))
            self.bbox_summary.append(self.summarize_mean_result('bbox', self.bbox_accu_results, 'ap', 'iou', 0.95, 'all', max_dets, 'AP.95'))
            self.bbox_summary.append(self.summarize_mean_result('bbox', self.bbox_accu_results, 'ap', 'iou', None, 'all', max_dets, 'AP'))
            self.bbox_summary.append(self.summarize_mean_result('bbox', self.bbox_accu_results, 'ap', 'iou', None, 'medium', max_dets, 'AP(M)'))
            self.bbox_summary.append(self.summarize_mean_result('bbox', self.bbox_accu_results, 'ap', 'iou', None, 'large', max_dets, 'AP(L)'))
            self.bbox_summary.append(self.summarize_mean_result('bbox', self.bbox_accu_results, 'ar', 'iou', 0.5, 'all', max_dets, 'AR.5'))
            self.bbox_summary.append(self.summarize_mean_result('bbox', self.bbox_accu_results, 'ar', 'iou', 0.75, 'all', max_dets, 'AR.75'))
            self.bbox_summary.append(self.summarize_mean_result('bbox', self.bbox_accu_results, 'ar', 'iou', None, 'all', max_dets, 'AR'))
            self.bbox_summary.append(self.summarize_mean_result('bbox', self.bbox_accu_results, 'ar', 'iou', None, 'medium', max_dets, 'AR(M)'))
            self.bbox_summary.append(self.summarize_mean_result('bbox', self.bbox_accu_results, 'ar', 'iou', None, 'large', max_dets, 'AR(L)'))
            results['bbox'] = self.bbox_summary

        if self.eval_seg and self.seg_accu_results:
            max_dets = self.max_detections['seg'][0]
            self.seg_summary = []
            self.seg_summary.append(self.summarize_mean_result('seg', self.seg_accu_results, 'ap', 'iou', 0.5, 'all', max_dets, 'AP.5'))
            self.seg_summary.append(self.summarize_mean_result('seg', self.seg_accu_results, 'ap', 'iou', 0.75, 'all', max_dets, 'AP.75'))
            self.seg_summary.append(self.summarize_mean_result('seg', self.seg_accu_results, 'ap', 'iou', 0.9, 'all', max_dets, 'AP.9'))
            self.seg_summary.append(self.summarize_mean_result('seg', self.seg_accu_results, 'ap', 'iou', 0.95, 'all', max_dets, 'AP.95'))
            self.seg_summary.append(self.summarize_mean_result('seg', self.seg_accu_results, 'ap', 'iou', None, 'all', max_dets, 'AP'))
            self.seg_summary.append(self.summarize_mean_result('seg', self.seg_accu_results, 'ap', 'iou', None, 'medium', max_dets, 'AP(M)'))
            self.seg_summary.append(self.summarize_mean_result('seg', self.seg_accu_results, 'ap', 'iou', None, 'large', max_dets, 'AP(L)'))
            self.seg_summary.append(self.summarize_mean_result('seg', self.seg_accu_results, 'ar', 'iou', 0.5, 'all', max_dets, 'AR.5'))
            self.seg_summary.append(self.summarize_mean_result('seg', self.seg_accu_results, 'ar', 'iou', 0.75, 'all', max_dets, 'AR.75'))
            self.seg_summary.append(self.summarize_mean_result('seg', self.seg_accu_results, 'ar', 'iou', None, 'all', max_dets, 'AR'))
            self.seg_summary.append(self.summarize_mean_result('seg', self.seg_accu_results, 'ar', 'iou', None, 'medium', max_dets, 'AR(M)'))
            self.seg_summary.append(self.summarize_mean_result('seg', self.seg_accu_results, 'ar', 'iou', None, 'large', max_dets, 'AR(L)'))
            results['seg'] = self.seg_summary

        if self.eval_kpts and self.kpts_accu_results:
            max_dets = self.max_detections['kpts'][0]
            self.kpts_summary = []
            self.kpts_summary.append(self.summarize_mean_result('kpts', self.kpts_accu_results, 'ap', 'iou', 0.5, 'all', max_dets, 'AP.5'))
            self.kpts_summary.append(self.summarize_mean_result('kpts', self.kpts_accu_results, 'ap', 'iou', 0.75, 'all', max_dets, 'AP.75'))
            self.kpts_summary.append(self.summarize_mean_result('kpts', self.kpts_accu_results, 'ap', 'iou', 0.9, 'all', max_dets, 'AP.9'))
            self.kpts_summary.append(self.summarize_mean_result('kpts', self.kpts_accu_results, 'ap', 'iou', 0.95, 'all', max_dets, 'AP.95'))
            self.kpts_summary.append(self.summarize_mean_result('kpts', self.kpts_accu_results, 'ap', 'iou', None, 'all', max_dets, 'AP'))
            self.kpts_summary.append(self.summarize_mean_result('kpts', self.kpts_accu_results, 'ap', 'iou', None, 'medium', max_dets, 'AP(M)'))
            self.kpts_summary.append(self.summarize_mean_result('kpts', self.kpts_accu_results, 'ap', 'iou', None, 'large', max_dets, 'AP(L)'))
            self.kpts_summary.append(self.summarize_mean_result('kpts', self.kpts_accu_results, 'ar', 'iou', 0.5, 'all', max_dets, 'AR.5'))
            self.kpts_summary.append(self.summarize_mean_result('kpts', self.kpts_accu_results, 'ar', 'iou', 0.75, 'all', max_dets, 'AR.75'))
            self.kpts_summary.append(self.summarize_mean_result('kpts', self.kpts_accu_results, 'ar', 'iou', None, 'all', max_dets, 'AR'))
            self.kpts_summary.append(self.summarize_mean_result('kpts', self.kpts_accu_results, 'ar', 'iou', None, 'medium', max_dets, 'AR(M)'))
            self.kpts_summary.append(self.summarize_mean_result('kpts', self.kpts_accu_results, 'ar', 'iou', None, 'large', max_dets, 'AR(L)'))
            results['kpts'] = self.kpts_summary

        return results


def bbox_nms(bboxes, thr):
    """ bboxes: cx, cy, w, h, score
    """
    bboxes = npf(bboxes)
    cx = bboxes[:, 0]
    cy = bboxes[:, 1]
    w = bboxes[:, 2]
    h = bboxes[:, 3]
    scores = bboxes[:, 4]
    x0 = cx - w * 0.5
    y0 = cy - h * 0.5
    x1 = cx + w * 0.5
    y1 = cy + h * 0.5
    areas = w * h

    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        x_max = np.maximum(x0[i], x0[order[1:]])
        y_max = np.maximum(y0[i], y0[order[1:]])
        x_min = np.minimum(x1[i], x1[order[1:]])
        y_min = np.minimum(y1[i], y1[order[1:]])

        w = np.maximum(0, x_min - x_max)
        h = np.maximum(0, y_min - y_max)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        idx = np.where(ovr <= thr)[0]
        order = order[idx + 1]

    return keep


def compute_oks(gt_kpts, res_kpts, gt_area, res_area, sigmas=KEYPOINT_SIGMAS, visbility_thr=0.0, return_dist=False):
    vars = (npf(sigmas) * 2)**2
    gt_x, gt_y, gt_v = gt_kpts[0::3], gt_kpts[1::3], gt_kpts[2::3]
    res_x, res_y, res_v = res_kpts[0::3], res_kpts[1::3], res_kpts[2::3]
    dx = gt_x - res_x
    dy = gt_y - res_y
    d = (dx**2 + dy**2)**0.5
    e = (dx**2 + dy**2) / vars / ((gt_area + res_area) / 2. + np.spacing(1)) / 2.
    idx = list(gt_v > visbility_thr) and list(res_v > visbility_thr)
    e = e[idx]
    d = d[idx]
    osk = np.sum(np.exp(-e)) / len(e) if len(e) > 0 else 0.
    if return_dist:
        avg_dist, max_dist = d.mean(), d.max()
        return osk, (avg_dist, max_dist)
    else:
        return osk


def kpts_nms(kpts, scores, areas, thr, sigmas=KEYPOINT_SIGMAS, visbility_thr=0.0):
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        oks = [compute_oks(kpts[i], kpts[j], areas[i], areas[j], sigmas, visbility_thr) for j in order[1:]]
        idx = np.where(oks <= thr)[0]
        order = order[idx + 1]

    return keep


def soft_kpts_nms(kpts, scores, areas, thr, max_keep=20, sigmas=KEYPOINT_SIGMAS, visbility_thr=0.0, soft_type='gaussian'):
    assert soft_type in ('gaussian', 'linear')
    order = scores.argsort()[::-1]
    scores = scores[order]
    keep = [0] * max_keep
    n_keep = 0
    while len(order) > 0:
        i = order[0]
        oks = [compute_oks(kpts[i], kpts[j], areas[i], areas[j], sigmas, visbility_thr) for j in order[1:]]

        order = order[1:]
        scores = scores[1:]
        if soft_type == 'gaussian':
            scores *= np.exp(-oks**2 / thr)
        elif soft_type == 'linear':
            idx = np.where(oks >= thr)[0]
            scores[idx] *= (1 - oks[idx])

        order2 = scores.argsort()[::-1]
        order = order[order2]
        scores = scores[order2]

        keep[n_keep] = i
        n_keep += 1

    keep = keep[:n_keep]
    return keep


def compute_avg_precision(rc, pr, method='interp'):
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
    return ap, pr, rc


def plot_precision_recall_curve(px, py, ap, save_dir):
    """ px: n_pnts
        py: n_cats x n_plot_pnts
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    n_cats = py.shape[0]
    for ci in range(n_cats):
        ax.plot(px, py[ci], linewidth=1,
                label=f'{ap[ci,0]:.3f} cat{ci},mAP@0.5')
        ax.text(0.05, 0.95, f'mAP@0.5:0.95 = {ap[ci].mean():.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(osp.join(save_dir, 'Precision-Recall_curve.png'), dpi=250)
    plt.close()


def plot_metric_confidence_curve(px, py, save_dir, xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    n_cats = py.shape[0]
    for ci in range(n_cats):
        ax.plot(px, py[ci], linewidth=1, label=f'cat{ci},mAP@0.5')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(osp.join(save_dir, f'{xlabel}-{ylabel}_curve.png'), dpi=250)
    plt.close()


class COCOCustomEval(COCOEval):
    def __init__(self,
                 coco_gt,
                 coco_res,
                 criterion=None,
                 criterion_type='iou',
                 keep_ann_id=False,
                 score_key='score',
                 default_score=0,
                 area_ranges=OrderedDict([('all', (0**2, 1e5**2))]),
                 max_detections=[-1],  # per cat per img
                 score_thres=np.linspace(0.5, 0.95, 10),
                 recall_thres=np.linspace(0, 1, 101),
                 gt_selected=None):
        self.coco_gt = coco_gt
        self.coco_res = coco_res
        assert callable(criterion)
        self.criterion = criterion
        self.criterion_type = criterion_type
        self.keep_ann_id = keep_ann_id
        self.score_key = score_key
        self.default_score = default_score
        self.area_ranges = area_ranges
        self.score_thres = score_thres
        self.recall_thres = recall_thres
        self.eval_type = 'custom'
        self.max_detections = {self.eval_type: max_detections}
        self.gt_selected = gt_selected

        self.img_tids = None
        self.cat_tids = None
        self.gt_anns = defaultdict(list)
        self.res_anns = defaultdict(list)

        self.eval_results = None
        self.accu_results = None
        self.summary = None

        self.prepare()

    def get_score(self, ann, score_type=None):
        suffix = '_min' if score_type == 'min' else ''
        return ann.get(self.score_key + suffix, self.default_score)

    def evaluate_single(self, cat_tid, img_tid, max_dets_idx, keep_ann_id=False):
        gt_anns = self.gt_anns[cat_tid, img_tid]  # ordered by ann tid if keep_ann_id
        res_anns = self.res_anns[cat_tid, img_tid]  # ordered by ann tid  if keep_ann_id
        max_dets = self.max_detections[self.eval_type][max_dets_idx]

        if len(gt_anns) == 0 or len(res_anns) == 0:
            return

        res_anns = sorted(res_anns, key=lambda a: -self.get_score(a))
        if max_dets > 0 and max_dets < len(res_anns):
            res_anns = res_anns[:max_dets]

        if keep_ann_id:
            gt_tids = set([gt_ann['id'].tid for gt_ann in gt_anns])
            res_tids = set([res_ann['id'].tid for res_ann in res_anns])
            assert gt_tids == res_tids
            tid2order = {res_ann['id'].tid: i for i, res_ann in enumerate(res_anns)}
            gt_anns = sorted([gt_ann for gt_ann in gt_anns], key=lambda a: tid2order[a['id'].tid])

        metrics = []
        if keep_ann_id:
            for gt_ann, res_ann in zip(gt_anns, res_anns):
                metric = self.criterion(self.coco_gt, gt_ann, self.coco_res, res_ann)
                metrics.append(metric)
            metrics = npf(metrics)  # np.ndarray (n_gt)
        else:
            metrics = self.criterion(self.coco_gt, gt_anns, self.coco_res, res_anns)  # np.ndarray (n_gt x n_res)

        for area_rng_idx, (area_rng_label, area_rng) in enumerate(self.area_ranges.items()):
            picked_gt_indices = [i for i, ann in enumerate(gt_anns) if area_rng[0] <= ann['area'] < area_rng[1]]
            picked_res_indices = picked_gt_indices if keep_ann_id else [
                i for i, ann in enumerate(res_anns)
                if ('area' in ann and area_rng[0] <= ann['area'] < area_rng[1]) or 'area' not in ann]

            picked_gt_anns = [gt_anns[i] for i in picked_gt_indices]
            picked_res_anns = [res_anns[i] for i in picked_res_indices]

            result = {
                'img_tid': img_tid,
                'cat_tid': cat_tid,
                'area_rng_idx': area_rng_idx,
                'area_rng_label': area_rng_label,
                'area_rng': area_rng,
                'max_dets_idx': max_dets_idx,
                'max_dets': max_dets,
                'gt_ann_tids': [ann['id'].tid for ann in picked_gt_anns],
                'res_ann_tids': [ann['id'].tid for ann in picked_res_anns],
                'res_ann_scores': [self.get_score(ann) for ann in picked_res_anns],
                'res_ann_scores_min': [self.get_score(ann, score_type='min') for ann in picked_res_anns],
                'metrics': None,
                'gt_match_by_gid': None,
                'res_match_by_gid': None
            }

            if picked_gt_indices:
                picked_metrics = metrics[picked_gt_indices] if keep_ann_id else metrics[picked_gt_indices][:, picked_res_indices]
                match_results = self.match_annotations(picked_metrics, picked_gt_anns, picked_res_anns, keep_ann_id=keep_ann_id)
                result['metrics'] = picked_metrics
                result.update(match_results)

            yield result

    def evaluate(self):
        self.eval_results = defaultdict(list)
        for cat_tid in self.cat_tids:
            for img_tid in self.img_tids:
                for max_dets_idx in range(len(self.max_detections[self.eval_type])):
                    for result in self.evaluate_single(cat_tid, img_tid, max_dets_idx, keep_ann_id=self.keep_ann_id):
                        area_rng_idx = result['area_rng_idx']
                        self.eval_results[cat_tid, area_rng_idx, max_dets_idx].append(result)

    def accumulate(self, **kwargs):
        self.accu_results = self.accumulate_results(self.eval_type, self.eval_results, **kwargs)
        return self.accu_results

    def summarize(self, score_thrs=(0.5, 0.75, None)):
        self.summary = []
        assert self.accu_results, f"Call accumulate() first to get `accu_results`"
        for ap_or_ar in ('ap', 'ar'):
            for score_thr in score_thrs:
                for area_label in self.area_ranges.keys():
                    for max_dets in self.max_detections[self.eval_type]:
                        name = ap_or_ar.upper() + (str(score_thr)[1:] if score_thr is not None else '')
                        if area_label != 'all':
                            name += f'({area_label})'
                        self.summary.append(self.summarize_mean_result(
                            self.eval_type, self.accu_results, ap_or_ar, self.criterion_type, score_thr, area_label, max_dets, name))
        return self.summary
