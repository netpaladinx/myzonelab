import os.path as osp

import numpy as np

from ....registry import DATASETS
from ...datautils import size_tuple, resize_image
from ...dataloader import collate
from ...transforms import Collect
from ..stream_dataset import StreamDataset, MultiStreamDataset
from ..base_dataset import BaseJITDataset
from .coco_utils import safe_bbox, bbox_center, bbox_scale, npf
from .coco_consts import BBOX_PADDING_RATIO, KEYPOINT_FLIP_PAIRS
from .coco_visualize import draw_results
from .coco_measure import COCOConfidence, COCOSmoothness


class COCOStreamMixin:
    def merge_results(self, detect_results=None, pose_results=None):
        results = {}  # img_id => list(dict(bbox=, keypoints=))
        for img_id, xyxy, conf in zip(detect_results['img_ids'], detect_results['xyxy_results'], detect_results['conf_results']):
            img_id = tuple(img_id)
            results[img_id] = [{'bbox': [x0, y0, x1 - x0, y1 - y0, c]} for (x0, y0, x1, y1), c in zip(xyxy, conf)]
        if pose_results is not None:
            for ann_id, kpts in zip(pose_results['ann_ids'], pose_results['kpts_results']):
                img_id, ai = ann_id[:-1], ann_id[-1]
                results[img_id][ai]['keypoints'] = kpts
        return results

    def visualize(self, results, batch, display=False, sink_id=0, work_dir=None):
        if not display:
            return

        imgs = {tuple(img_id): img for img_id, img in zip(batch['frame_id'], batch['frame'])}
        frames = {}
        for img_id, res in results.items():
            img_id = tuple(img_id)
            img = draw_results(res, imgs[img_id])
            src_id = self.get_source_id(img_id)
            stream_id = (src_id, sink_id)
            frames[stream_id] = img

        self.display(frames, out_dir=work_dir)

    def evaluate_step(self, results, batch, sink_id=0, work_dir=None, verbose=True):
        for img_id, res in results.items():
            img_id = tuple(img_id)
            src_id = self.get_source_id(img_id)
            fno = self.get_frame_no(img_id)
            stream_id = (src_id, sink_id)

            if self.eval_confidence:
                if stream_id not in self.coco_conf:
                    assert work_dir is not None
                    self.coco_conf[stream_id] = COCOConfidence(stream_id, summary_dir=osp.join(work_dir, 'confidence'), verbose=verbose)
                self.coco_conf[stream_id].eval_frame(fno, res)

            if self.eval_smoothness:
                if stream_id not in self.coco_smooth:
                    assert work_dir is not None
                    self.coco_smooth[stream_id] = COCOSmoothness(stream_id, summary_dir=osp.join(work_dir, 'smoothness'), verbose=verbose)
                self.coco_smooth[stream_id].push(fno, res)

    def summarize(self):
        if self.eval_smoothness:
            for stream_id, coco_smooth in self.coco_smooth.items():
                print(f"Summarize smoothness on stream {stream_id}")
                coco_smooth.summarize()


@DATASETS.register_class('coco_stream')
class COCOStream(StreamDataset, COCOStreamMixin):
    # source id: stream source ID; sink id: inferring model ID

    def __init__(self, data_source=None, data_params=None, data_transforms=None, data_eval=None, name=None):
        super().__init__(data_source, data_params, data_transforms, data_eval, name)
        self.input_size = size_tuple(self.data_params['input_size'])  # w, h

        self.eval_confidence = self.data_eval.get('eval_confidence', False) if self.data_eval else False
        self.coco_conf = {}  # stream_id => COCOConfidence

        self.eval_smoothness = self.data_eval.get('eval_smoothness', False) if self.data_eval else False
        self.coco_smooth = {}  # stream_id => COCOSmoothness

    def get_unprocessed_item(self):
        quit, input_dict = super().get_unprocessed_item()
        if not quit:
            img_id = input_dict['frame_id']
            img = input_dict['frame']
            img_h, img_w = img.shape[:2]
            img, ratio, (border_top, border_bottom, border_left, border_right) = resize_image(img, self.input_size)
            input_dict['img'] = img
            input_dict['img_id'] = img_id
            input_dict['_revert_params'] = (ratio, (border_top, border_bottom, border_left, border_right), (img_w, img_h))
        return quit, input_dict

    def evaluate_step(self, results, batch, **kwargs):
        COCOStreamMixin.evaluate_step(self, results, batch, **kwargs)

    def summarize(self, **kwargs):
        COCOStreamMixin.summarize(self, **kwargs)


@DATASETS.register_class('coco_multi_stream')
class COCOMultiStream(MultiStreamDataset, COCOStreamMixin):
    def __init__(self, data_source=None, data_params=None, data_transforms=None, data_eval=None, name=None):
        super().__init__(data_source, data_params, data_transforms, data_eval, name)
        self.input_size = size_tuple(self.data_params['input_size'])  # w, h
        last_trans = self.data_transforms.transforms[-1]
        if isinstance(last_trans, Collect):
            assert last_trans.input_batching == 'concat' and last_trans.array_batching == 'concat'

        self.eval_confidence = self.data_eval.get('eval_confidence', False) if self.data_eval else False
        self.coco_conf = {}

        self.eval_smoothness = self.data_eval.get('eval_smoothness', False) if self.data_eval else False
        self.coco_smooth = {}

    def get_unprocessed_item(self):
        quit, input_dict = super().get_unprocessed_item()
        if not quit:
            input_dict['img'] = []
            input_dict['img_id'] = []  # needs array_batching = 'concat
            input_dict['_revert_params'] = []
            for img_id, img in zip(input_dict['frame_id'], input_dict['frame']):
                img_h, img_w = img.shape[:2]
                img, ratio, (border_top, border_bottom, border_left, border_right) = resize_image(img, self.input_size)
                input_dict['img'].append(img)
                input_dict['img_id'].append(img_id)  # img_id: (fno,) or (src_id, fno)
                input_dict['_revert_params'].append([ratio, (border_top, border_bottom, border_left, border_right), (img_w, img_h)])
            input_dict['img'] = np.stack(input_dict['img'], axis=0)  # needs input_batching = 'concat'
        return quit, input_dict

    def evaluate_step(self, results, batch, **kwargs):
        COCOStreamMixin.evaluate_step(self, results, batch, **kwargs)

    def summarize(self, **kwargs):
        COCOStreamMixin.summarize(self, **kwargs)


@DATASETS.register_class('coco_pose_jit')
class COCOPoseJit(BaseJITDataset):
    def __init__(self, data_params=None, data_transforms=None, data_eval=None, name=None):
        super().__init__(None, data_params, data_transforms, data_eval, name)
        self.input_size = size_tuple(self.data_params['input_size'])
        self.input_aspect_ratio = self.input_size[0] / self.input_size[1]
        self.bbox_padding_ratio = self.data_params.get('bbox_padding_ratio', BBOX_PADDING_RATIO)

    def get_input_batch(self, detect_batch, detect_results):
        images = {tuple(img_id): img for img_id, img in zip(detect_batch['frame_id'], detect_batch['frame'])}
        input_batch = []
        for img_id, xyxy in zip(detect_results['img_ids'], detect_results['xyxy_results']):
            img_id = tuple(img_id)
            img = images[img_id]
            img_h, img_w = img.shape[:2]
            for ai, (x0, y0, x1, y1) in enumerate(xyxy):
                ann_id = img_id + (ai,)
                bbox = safe_bbox([x0, y0, x1 - x0, y1 - y0], img_h, img_w)
                center = bbox_center(bbox)
                scale = bbox_scale(bbox, self.input_aspect_ratio, self.bbox_padding_ratio)

                input_dict = {
                    'ann_id': ann_id,
                    'img_id': img_id,
                    'img': img,
                    'bbox': npf(bbox),
                    'center': npf(center),
                    'scale': npf(scale),
                    'flip_pairs': KEYPOINT_FLIP_PAIRS
                }
                input_dict = self.data_transforms(input_dict)
                input_batch.append(input_dict)
        input_batch = collate(input_batch)
        return input_batch
