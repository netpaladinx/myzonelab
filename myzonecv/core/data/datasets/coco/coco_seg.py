import copy
import random

import numpy as np
import cv2
from torch.utils.data import Dataset

from ....registry import DATASETS
from ..base_dataset import BaseDataset
from .coco_data import COCOData
from .coco_utils import size_tuple, seg2mask, resize_image


@DATASETS.register_class('coco_class_seg')
class COCOClassSeg(BaseDataset):
    default_class_names = (
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
        'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
        'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
        'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
        'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
        'floor-other', 'floor-stone', 'floor-tile', 'floor-wood',
        'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass',
        'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat',
        'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
        'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
        'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
        'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
        'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other',
        'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
        'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
        'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
        'window-blind', 'window-other', 'wood')

    default_palette = [[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
                       [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
                       [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
                       [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
                       [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
                       [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
                       [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
                       [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
                       [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
                       [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
                       [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
                       [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
                       [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
                       [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
                       [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
                       [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
                       [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
                       [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
                       [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
                       [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
                       [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
                       [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
                       [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
                       [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
                       [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
                       [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
                       [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
                       [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
                       [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
                       [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
                       [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
                       [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
                       [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
                       [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
                       [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
                       [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
                       [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
                       [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
                       [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
                       [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
                       [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
                       [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
                       [64, 192, 96], [64, 160, 64], [64, 64, 0]]

    def __init__(self, data_source=None, data_params=None, data_transforms=None, data_eval=None, name=None):
        super().__init__(data_source, data_params, data_transforms, data_eval, name)
        self.input_size = size_tuple(self.data_params['input_size'])
        self.input_channels = self.data_params['input_channels']
        self.class_names = self.data_params.get('class_names', self.default_class_names)
        self.num_classes = self.data_params.get('num_classes', len(self.class_names))
        self.remove_zero_index = self.data_params.get('remove_zero_index', False)
        self.person_only = self.data_params.get('person_only', False)

        self.coco_data = COCOData(**data_source)
        self.input_data = self.load_input_data()
        self.input_indices = list(range(len(self.input_data)))
        if self.shuffle:
            random.shuffle(self.input_indices)

    def load_input_data(self):
        input_data = []
        for img in self.coco_data.images:
            img_id = img['id']
            img = self.coco_data.get_img(img_id)
            img_h, img_w = img.get('height'), img.get('width')
            ann_ids = sorted(img['annotation_ids'], key=lambda a: a.tid)
            anns = self.coco_data.get_anns(ann_ids)
            anns = [ann for ann in anns if 'segmentation' in ann]
            seg_cls, seg_mask = self.get_segmentation(anns, img_w, img_h)

            if len(anns) == 0:
                continue

            input_dict = {
                'img_id': img_id.tid,
                'ann_ids': [ann_id.tid for ann_id in ann_ids],
                'seg_cls': seg_cls,
                'seg_mask': seg_mask,
            }

            input_data.append(input_dict)

        print(f'{len(input_data)} input items loaded (one item one image)')
        return input_data

    def get_unprocessed_item(self, idx):
        idx = self.input_indices[idx]
        input_dict = copy.deepcopy(self.input_data[idx])
        img = self.coco_data.read_img(input_dict['img_id'])  # h x w x c
        seg_cls = input_dict['seg_cls']
        seg_mask = input_dict['seg_mask']
        input_dict['orig_img'] = img
        input_dict['orig_seg_cls'] = seg_cls
        input_dict['orig_seg_mask'] = seg_mask
        img_h, img_w = img.shape[:2]
        img, ratio, (border_top, border_bottom, border_left, border_right) = resize_image(img, self.input_size)
        input_dict['img'] = img
        input_dict['_revert_params'] = (ratio, (border_top, border_bottom, border_left, border_right), (img_w, img_h))
        bg_val = -1 if self.remove_zero_index else 0
        input_dict['seg_cls'] = resize_image(seg_cls, self.input_size, border_value=bg_val, interpolation=cv2.INTER_NEAREST)[0]
        input_dict['seg_mask'] = resize_image(seg_mask, self.input_size, border_value=0)[0]
        return input_dict

    def get_segmentation(self, anns, img_w, img_h):
        seg_cls = np.zeros((img_h, img_w)).astype(np.int32)
        if self.remove_zero_index:
            seg_cls -= 1
        seg_mask = np.zeros((img_h, img_w, self.num_classes))
        for ann in anns:
            mask = seg2mask(ann['segmentation'], img_w, img_h)
            cat_id = ann['category_id'] - 1 if self.remove_zero_index else ann['category_id']
            seg_cls[mask > 0] = cat_id
            seg_mask[cat_id] = mask
        return seg_cls, seg_mask

    @property
    def image_ids(self):  # list(img_tid)
        return [img_id.tid for img_id in self.coco_data.get_img_ids()]

    @property
    def selected(self):  # set((img_tid, ann_tid))
        return set([(input_dict['img_id'], ann_id) for input_dict in self.input_data for ann_id in input_dict['ann_ids']])

    def evaluate_all(self, all_results, plot_dir=None):
        pass


@DATASETS.register_class('coco_instance_seg')
class COCOInstanceSeg(Dataset):
    pass
