import os.path as osp
from collections import OrderedDict
import random
import copy

import numpy as np

from myzonecv.core.registry import DATASETS
from myzonecv.core.utils import get_if_is, load_numpy, save_img, get_logger
from myzonecv.core.data.datasets import BaseDataset, COCOCustomData, COCOCustomEval
from myzonecv.core.data.datautils import seg2mask, int_bbox, mask_area, size_tuple, bbox_center, bbox_scale, load_numpy, npf
from myzonecv.core.data.dataconsts import BBOX_PADDING_RATIO, PERSON_CAT_ID, KEYPOINT_INDEX2NAME
from myzonecv.core.data.transforms import Compose

logger = get_logger('coco_bbox_seg')


def process_res_ann(res_ann, ref_ann=None):
    res_ann = copy.deepcopy(res_ann)
    ref_ann = copy.deepcopy(ref_ann) if ref_ann else {}

    assert 'bbox' in res_ann and 'bbox_seg_mask' in res_ann
    if 'bbox' in ref_ann:
        assert tuple(res_ann['bbox']) == tuple(int_bbox(ref_ann['bbox']))

    res_ann.pop('id', None)
    res_ann.pop('image_id', None)
    res_ann.pop('category_id', None)
    for key, val in res_ann.items():
        ref_ann[key] = val

    return ref_ann


def get_seg_mask(ann, img_w, img_h, bbox=None):
    seg = ann['segmentation']
    seg_mask = seg2mask(seg, img_w, img_h)
    x0, y0, w, h = bbox if bbox is not None else int_bbox(ann['bbox'])
    bbox_seg_mask = seg_mask[y0:y0 + h, x0:x0 + w].copy()
    return seg_mask, bbox_seg_mask


def bbox_seg_iou(coco_gt, gt_ann, coco_res, res_ann, eps=1e-10):  # with keep_ann_id = True
    img = coco_gt.get_img(gt_ann['image_id'])
    img_h, img_w = img['height'], img['width']
    gt_bbox = int_bbox(gt_ann['bbox'])
    _, gt_bbox_seg_mask = get_seg_mask(gt_ann, img_w, img_h, bbox=gt_bbox)

    res_bbox = res_ann['bbox']
    res_bbox_seg_mask = res_ann['bbox_seg_mask']
    if isinstance(res_bbox_seg_mask, str):
        res_bbox_seg_mask = load_numpy(res_bbox_seg_mask)
    assert isinstance(res_bbox_seg_mask, np.ndarray)

    assert tuple(gt_bbox) == tuple(res_bbox)
    inter_mask = (gt_bbox_seg_mask * res_bbox_seg_mask)
    inter_area = mask_area(inter_mask)
    union_area = mask_area(gt_bbox_seg_mask) + mask_area(res_bbox_seg_mask) - inter_area
    return inter_area / max(union_area, eps)


@DATASETS.register_class('coco_bbox_seg')
class COCOBBoxSeg(BaseDataset):
    def __init__(self, data_source=None, data_params=None, data_transforms=None, data_eval=None, name=None):
        super().__init__(data_source, data_params, data_transforms, data_eval, name)
        self.input_size = size_tuple(self.data_params['input_size'])  # weight, height
        self.input_channels = self.data_params['input_channels']
        self.output_size = size_tuple(self.data_params['output_size'])
        self.input_aspect_ratio = self.input_size[0] / self.input_size[1]
        self.bbox_padding_ratio = data_params.get('bbox_padding_ratio', BBOX_PADDING_RATIO)

        self.coco_data = COCOCustomData(process_res_ann=process_res_ann, **data_source)
        self.input_data = self.load_input_data()
        self.input_indices = list(range(len(self.input_data)))
        if self.shuffle:
            random.shuffle(self.input_indices)

    def load_input_data(self):
        input_data = []
        for ann in self.coco_data.annotations:
            if ann['category_id'].tid != PERSON_CAT_ID:  # person only
                continue
            if not (self.data_mode == 'infer' or 'segmentation' in ann):
                continue

            bbox = int_bbox(ann['bbox'])
            center = bbox_center(bbox)
            scale = bbox_scale(bbox, self.input_aspect_ratio, self.bbox_padding_ratio)

            input_dict = {
                'ann_id': ann['id'].tid,
                'img_id': ann['image_id'].tid,
                'input_size': npf(self.input_size),
                'output_size': npf(self.output_size),
                'orig_bbox': npf(bbox),  # keep static during transformation
                'center': npf(center),
                'scale': npf(scale),
            }

            input_data.append(input_dict)

        logger.info(f'{len(input_data)} input items loaded (one item one annotation)')
        return input_data

    def get_unprocessed_item(self, idx):
        idx = self.input_indices[idx]
        input_dict = copy.deepcopy(self.input_data[idx])
        img = self.coco_data.read_img(input_dict['img_id'])  # h x w x c
        img_h, img_w = img.shape[:2]
        ann = self.coco_data.get_ann(input_dict['ann_id'])
        input_dict['img'] = img
        input_dict['orig_size'] = npf([img_w, img_h])
        if self.data_mode != 'infer':
            seg_mask, _ = get_seg_mask(ann, img_w, img_h, bbox=input_dict['orig_bbox'])
            input_dict['seg_mask'] = seg_mask
        return input_dict

    @property
    def image_ids(self):  # list(img_tid)
        return [img_id.tid for img_id in self.coco_data.get_img_ids()]

    @property
    def selected(self):  # set((img_tid, ann_tid))
        return set([(input_dict['img_id'], input_dict['ann_id']) for input_dict in self.input_data])

    def evaluate_all(self, all_results, plot_dir=None):
        ann_results = []

        for res_dict in all_results:
            seg_mask_results = res_dict['seg_mask_results']
            bbox_seg_mask_results = res_dict['bbox_seg_mask_results']
            bbox_results = res_dict['bbox_results']
            conf_scores = res_dict['conf_scores']
            ann_ids = res_dict['ann_ids']

            for i, ann_id in enumerate(ann_ids):
                ann_results.append({
                    'id': ann_id,
                    'seg_mask': seg_mask_results[i],
                    'bbox': bbox_results[i],
                    'bbox_seg_mask': bbox_seg_mask_results[i],
                    'score': conf_scores[i]
                })

        coco_res = self.coco_data.from_results(ann_results, keep_ann_id=True)
        coco_eval = COCOCustomEval(self.coco_data, coco_res, criterion=bbox_seg_iou, keep_ann_id=True, gt_selected=self.selected)
        coco_eval.evaluate()
        coco_eval.accumulate(plot_dir=plot_dir)
        results = coco_eval.summarize()
        results = OrderedDict([(res['name'], res['mean_value']) for res in results])
        return results

    def dump_all(self, all_results, dump_dir=None, ctx=None):
        dump_dir = get_if_is(ctx, 'dump_dir', dump_dir, None)
        for res_dict in all_results:
            seg_mask_results = res_dict['seg_mask_results']
            ann_ids = res_dict['ann_ids']

            for i, ann_id in enumerate(ann_ids):
                ann = self.coco_data.get_ann(ann_id)
                img = self.coco_data.get_img(ann['image_id'])
                fname_sp = img['file_name'].rsplit('.', 1)
                dump_file_name = f"{fname_sp[0]}_{str(ann['id'])}.{fname_sp[1]}"
                seg_mask = load_numpy(seg_mask_results[i])
                dump_path = osp.join(dump_dir, dump_file_name)
                save_img(seg_mask, dump_path)


def get_points(ann, vis_thr=0.):
    if 'keypoints' not in ann:
        return None

    kpts = npf(ann['keypoints']).reshape(-1, 3)
    nose = kpts[KEYPOINT_INDEX2NAME['nose']]
    left_eye = kpts[KEYPOINT_INDEX2NAME['left_eye']]
    right_eye = kpts[KEYPOINT_INDEX2NAME['right_eye']]
    left_shoulder = kpts[KEYPOINT_INDEX2NAME['left_shoulder']]
    right_shoulder = kpts[KEYPOINT_INDEX2NAME['right_shoulder']]
    left_hip = kpts[KEYPOINT_INDEX2NAME['left_hip']]
    right_hip = kpts[KEYPOINT_INDEX2NAME['right_hip']]

    points = []
    if all(v > vis_thr for v in (nose[2], left_eye[2], right_eye[2])):
        head = ((nose + left_eye + right_eye) / 3)[:2]
        points.append(head)
    if all(v > vis_thr for v in (left_shoulder[2], right_shoulder[2], left_hip[2], right_hip[2])):
        spine = ((left_shoulder + right_shoulder + left_hip + right_hip) / 4)[:2]
        points.append(spine)
    if all(v > vis_thr for v in (left_hip[2], right_hip[2])):
        pelvis = ((left_hip + right_hip) / 2)[:2]
        points.append(pelvis)

    return points


@DATASETS.register_class('coco_points_guided_bbox_seg')
class COCOPointsGuidedBBoxSeg(COCOBBoxSeg):
    def load_input_data(self):
        input_data = []
        for ann in self.coco_data.annotations:
            if ann['category_id'].tid != PERSON_CAT_ID:  # person only
                continue
            if not (self.data_mode == 'infer' or 'segmentation' in ann):
                continue
            points = get_points(ann)
            if len(points) == 0:
                continue

            bbox = int_bbox(ann['bbox'])
            center = bbox_center(bbox)
            scale = bbox_scale(bbox, self.input_aspect_ratio, self.bbox_padding_ratio)

            input_dict = {
                'ann_id': ann['id'].tid,
                'img_id': ann['image_id'].tid,
                'input_size': npf(self.input_size),
                'output_size': npf(self.output_size),
                'orig_bbox': npf(bbox),  # keep static during transformation
                'center': npf(center),
                'scale': npf(scale),
                'points': npf(points)  # n_points x 2, [x, y]
            }

            input_data.append(input_dict)

        print(f'{len(input_data)} input items loaded (one item one annotation)')
        return input_data
