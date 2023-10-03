import os.path as osp
import random
from collections import OrderedDict
import copy

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from ....registry import DATASETS
from ..base_dataset import BaseDataset
from .coco_data import COCOData
from .coco_utils import resize_image, size_tuple, bbox2xyxy, xyxy2bbox, npf, npi
from .coco_visualize import get_color
from .coco_eval import COCOEval


@DATASETS.register_class('coco_detect')
class COCODetect(BaseDataset):
    def __init__(self, data_source=None, data_params=None, data_transforms=None, data_eval=None, name=None):
        super().__init__(data_source, data_params, data_transforms, data_eval, name)
        self.input_size = size_tuple(self.data_params['input_size'])  # w,h
        self.input_channels = self.data_params['input_channels']
        self.num_classes = self.data_params['num_classes']
        self.anchors = self.data_params['anchors']
        self.num_anchors = len(self.anchors[0]) // 2
        self.num_anchor_layers = len(self.anchors)
        self.strides = self.data_params['strides']
        self.person_only = self.data_params.get('person_only', False)
        self.image_position = self.data_params.get('image_position', 'center')

        self.coco_data = COCOData(**data_source)
        self.input_data = self.load_input_data()
        self.input_indices = list(range(len(self.input_data)))
        if self.shuffle:
            random.shuffle(self.input_indices)
        self.class_weights = self.compute_class_weights()

    def load_input_data(self):
        input_data = []
        for img in self.coco_data.images:
            img_id = img['id']
            ann_ids = sorted(img['annotation_ids'], key=lambda a: a.tid)
            anns = self.coco_data.get_anns(ann_ids)
            xyxy = [bbox2xyxy(ann['bbox']) for ann in anns]
            cls = [0 if self.person_only else ann['category_id'].tid[0] - 1 for ann in anns]

            if len(anns) == 0:
                continue

            input_dict = {
                'img_id': img_id.tid,
                'ann_ids': [ann_id.tid for ann_id in ann_ids],
                'orig_xyxy': npf(xyxy),
                'xyxy': npf(xyxy),  # row: x0, y0, x1, y1
                'cls': npi(cls)
            }

            input_data.append(input_dict)

        print(f'{len(input_data)} input items loaded (one item one image)')
        return input_data

    def get_unprocessed_item(self, idx):
        idx = self.input_indices[idx]
        input_dict = copy.deepcopy(self.input_data[idx])
        img = self.coco_data.read_img(input_dict['img_id'])  # h x w x c
        input_dict['orig_img'] = img
        img_h, img_w = img.shape[:2]
        img, ratio, (border_top, border_bottom, border_left, border_right) = \
            resize_image(img, self.input_size, position=self.image_position)
        input_dict['img'] = img
        input_dict['xyxy'][:, :4] *= ratio
        input_dict['xyxy'][:, [0, 2]] += border_left
        input_dict['xyxy'][:, [1, 3]] += border_top
        input_dict['_revert_params'] = (ratio, (border_top, border_bottom, border_left, border_right), (img_w, img_h))
        return input_dict

    def compute_class_weights(self):
        if self.person_only:
            weights = np.zeros((self.num_classes))
            weights[0] = 1.
        else:
            classes = np.concatenate([input_dict['cls'] for input_dict in self.input_data], 0)
            weights = np.bincount(classes, minlength=self.num_classes)
            weights[weights == 0] = 1
            weights = 1 / weights
            weights /= weights.sum()
        return weights

    @property
    def image_ids(self):  # list(img_tid)
        return [img_id.tid for img_id in self.coco_data.get_img_ids()]

    @property
    def selected(self):  # set((img_tid, ann_tid))
        return set([(input_dict['img_id'], ann_id) for input_dict in self.input_data for ann_id in input_dict['ann_ids']])

    def plot_labels(self, save_dir):
        xyxy = np.concatenate([item['xyxy'] for item in self.input_data], axis=0)
        cls = np.concatenate([item['cls'] for item in self.input_data], axis=0)
        n_cls = int(cls.max() + 1)
        x0, y0, x1, y1 = np.split(xyxy, xyxy.shape[1], axis=1)
        cxywh = np.concatenate(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), axis=1)

        dat = pd.DataFrame(cxywh, columns=['cx', 'cy', 'width', 'height'])

        # seaborn correlogram
        sn.pairplot(dat, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
        plt.savefig(osp.join(save_dir, 'labels_correlogram.jpg'), dpi=200)
        plt.close()

        # matplotlib labels
        matplotlib.use('svg')  # faster
        ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
        ax[0].hist(cls, bins=np.linspace(0, n_cls, n_cls + 1) - 0.5, rwidth=0.8)
        ax[0].set_ylabel('instances')
        ax[0].set_xlabel('classes')
        sn.histplot(dat, x='cx', y='cy', ax=ax[2], bins=50, pmax=0.9)
        sn.histplot(dat, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

        # rectangles
        img = self.coco_data.read_img(self.input_data[0]['img_id'])
        img_h, img_w = img.shape[:2]
        img = Image.fromarray(np.ones((img_h, img_w, 3), dtype=np.uint8) * 255)
        for c, (x0, y0, x1, y1) in zip(cls, xyxy):
            ImageDraw.Draw(img).rectangle(((x0, y0), (x1, y1)), width=1, outline=get_color(c))
        ax[1].imshow(img)
        ax[1].axis('off')

        for i in (0, 1, 2, 3):
            for s in ('top', 'right', 'left', 'bottom'):
                ax[i].spines[s].set_visible(False)

        plt.savefig(osp.join(save_dir, 'labels.jpg'), dpi=200)
        matplotlib.use('Agg')
        plt.close()

    def evaluate_all(self, all_results, plot_dir=None, ctx=None):
        plot_dir = ctx.get('plot_dir') if not plot_dir and ctx else plot_dir

        ann_results = []

        for res_dict in all_results:
            img_ids = res_dict['img_ids']
            xyxy_results = res_dict['xyxy_results']
            cls_results = res_dict['cls_results']
            conf_results = res_dict['conf_results']

            for i, img_id in enumerate(img_ids):
                xyxy = xyxy_results[i].tolist()
                cls = cls_results[i].tolist()
                conf = conf_results[i].tolist()

                for j in range(len(xyxy)):
                    ann_results.append({
                        'image_id': tuple(img_id),
                        'category_id': (int(cls[j]) + 1,),
                        'bbox': xyxy2bbox(xyxy[j]),  # xywh
                        'bbox_score': conf[j]
                    })

        coco_res = self.coco_data.from_results(ann_results)
        coco_eval = COCOEval(self.coco_data, coco_res, eval_bbox=True, gt_selected=self.selected)
        coco_eval.evaluate()
        coco_eval.accumulate(plot_dir=plot_dir)
        results = coco_eval.summarize()['bbox']
        results = OrderedDict([(res['name'], res['mean_value']) for res in results])
        return results
