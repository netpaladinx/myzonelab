from collections import defaultdict
import json
import glob
import os.path as osp
import numpy as np

import myzonecv.apis as APIs


def load_data(ann_file):
    with open(ann_file) as fin:
        return json.load(fin)


def parse_data(data):
    imgs = data['images']
    anns = data['annotations']
    id2img = {img['id']: img for img in imgs}
    img2anns = defaultdict(list)
    for ann in anns:
        img_id = ann['image_id']
        img2anns[img_id].append(ann)
    return id2img, img2anns


dataset_dir = './workspace/data_zoo/trainval/20220331_FightFlow_RetrainingData_Error_0-350_Revised'
ann_file = osp.join(dataset_dir, 'annotations.json')
data = load_data(ann_file)
id2img, img2anns = parse_data(data)

img_id, img = next(iter(id2img.items()))
anns = img2anns[img_id]

config = 'pose_hrnet_w32_coco_256x192_v2'
checkpoint = osp.abspath('./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-experimental_20220605T060104.pth')

batch, model = APIs.tools.test_pose_single(img, anns, dataset_dir=dataset_dir, config=config, checkpoint=checkpoint)
print(batch)
