import os.path as osp
import json

import numpy as np

import myzonecv.apis.tools as T


def load_data(ann_file):
    with open(ann_file) as fin:
        return json.load(fin)


data_dir = './workspace/data_zoo/trainval/FighterIDs_Base/train'
data = load_data(osp.join(data_dir, 'annotations.json'))


def show_fighter(data, fighter_id, show_number=8 * 8):
    id2img = {img['id']: img for img in data['images']}
    anns = [ann for ann in data['annotations'] if ann['fighter_id'] == fighter_id]
    anns = np.random.choice(anns, size=show_number, replace=False)
    img_paths = [osp.join(data_dir, 'images', id2img[ann['image_id']]['file_name']) for ann in anns]
    img_grid = T.make_image_grid(img_paths, (100, 100), 8, inner_padding=2)
    T.show_image(img_grid)


show_fighter(data, 1)
