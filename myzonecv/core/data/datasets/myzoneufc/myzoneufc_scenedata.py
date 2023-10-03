import os
import os.path as osp
from collections import defaultdict
import json
import copy
import shutil

import numpy as np
import skimage
import skimage.io as io
import skimage.color as color
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from ....utils import tolist, get_logger
from ...datautils import bbox_area
from ..coco.coco_data import IDRegistry, ID, sorted_by_id, tuple_id
from ..coco import mask as mask_utils
from ..coco.coco_utils import seg2mask
from .myzoneufc_consts import NONFIGHTER_LABEL

logger = get_logger('myzoneufc_scenedata')


class MyZoneUFCSceneData:
    """
    Holds annotation files for generate image-based training dataset

    Main keys in annotation files:
    {
        "info": {
            "description" (str),
            "version" (str),
            "date_created" (str)
        },
        "categories": [{
            "id" (int),
            "name" (str: lower)
        }],
        "fights": [{
            "id" (int),
            "name" (str: lower): f"{date}_{person_name}-vs-{person_name}" or "{person_name}-vs-{person_name}" if date is unknown,
            "date" (str): "<yyyymmdd>"
        }],
        "images": [{
            "id" (int): f"{date}{fight_id:02d}{camera_id}{cat_id}{frame_index:06d}" (18 digits)
            "file_name" (str),
            "width" (int),
            "height" (int),
            "category_id" (int),
            "fight_id" (int),
            "camera_id" (int): 1 ~ 8
        }]
    }
    """

    def __init__(self,
                 ann_file=None,      # str or list(str)
                 img_dir=None,       # str or list(str)
                 drop_prob=0.,       # randomly skip images according to the probability
                 drop_imgs=None,     # skip specified images
                 share_cat_ids=True,      # share cat id space across multiple annotation files
                 share_fight_ids=True,    # share fight id space across multiple annotation files
                 post_init_process=None,  # custom function called at the end of init()
                 process_res_ann=None     # custom function called for processing each res_ann
                 ):
        self.ann_file = ann_file
        self.img_dir = img_dir
        self.drop_prob = drop_prob
        self.drop_imgs = set(drop_imgs) if drop_imgs is not None else None
        self.share_cat_ids = share_cat_ids
        self.share_fight_ids = share_fight_ids

        # internal data structure
        self.dataset = None   # dict as data container to store list data of 'categories', 'images', 'fights'
        self.cats = None      # dict that maps cat id to cat dict
        self.fights = None    # dict that maps fight id to fight dict
        self.imgs = None      # dict that maps img id to img dict
        self.cat_id_reg = None      # registry that holds all cat id
        self.fight_id_reg = None    # registry that holds all fight id
        self.img_id_reg = None      # registry that holds all img id

        # load data from files to fill internal data structure
        if ann_file is not None and img_dir is not None:
            self.load_data(ann_file, img_dir)

        if callable(post_init_process):
            post_init_process(self)

        if callable(process_res_ann):
            self._process_res_ann = process_res_ann

    def load_data(self, ann_file, img_dir):
        ann_files = tolist(ann_file)
        img_dirs = tolist(img_dir)

        if ann_files and img_dirs:
            if len(ann_files) > 1 and len(img_dirs) == 1:
                img_dirs = img_dir * len(ann_files)  # share the same img_dir by multiple ann_files
            elif len(ann_files) == 1 and len(img_dirs) > 1:
                ann_files = ann_files * len(img_dirs)  # share the same ann_file by mutiple img_dirs
            assert len(ann_files) == len(img_dirs), "Numbers of annotation files and image dirs do not match"
            for f, d in zip(ann_files, img_dirs):
                assert osp.isfile(f), f"{f} is not a file"
                assert osp.isdir(d), f"{d} is not a directory"

            (dataset, cats, imgs, fights,
             cat_id_reg, img_id_reg, fight_id_reg) = self._load_data(ann_files, img_dirs)

            self.ann_file = ann_files
            self.img_dir = img_dirs
            self.dataset = dataset
            self.cats = cats
            self.fights = fights
            self.imgs = imgs
            self.cat_id_reg = cat_id_reg
            self.fight_id_reg = fight_id_reg
            self.img_id_reg = img_id_reg
        else:
            logger.warning("No annotation files or image directories are specified")

    def _load_data(self, ann_files, img_dirs):
        single_source = len(ann_files) == 1
        dataset = defaultdict(list)
        cats = dict()
        fights = dict()
        imgs = dict()
        cat_id_reg = IDRegistry('category', exist_ok=(not single_source) and self.share_cat_ids)
        fight_id_reg = IDRegistry('fight', exist_ok=(not single_source) and self.share_fight_ids)
        img_id_reg = IDRegistry('image')

        for i, (ann_file, img_dir) in enumerate(zip(ann_files, img_dirs)):
            logger.info(f"Loading data from {ann_file}:")

            with open(ann_file) as fin:
                loaded_dict = json.load(fin)
                loaded_cats = loaded_dict.get('categories', [])
                loaded_fights = loaded_dict.get('fights', [])
                loaded_imgs = loaded_dict.get('images', [])

            cat_src_idx = None if single_source or self.share_cat_ids else i
            self._load_cats(cat_src_idx, loaded_cats, dataset, cats, cat_id_reg)

            fight_src_idx = None if single_source or self.share_fight_ids else i
            self._load_fights(fight_src_idx, loaded_fights, dataset, fights, fight_id_reg)

            img_src_idx = None if single_source else i
            self._load_imgs(img_src_idx, loaded_imgs, dataset, imgs, img_id_reg,
                            self.drop_prob, self.drop_imgs, img_dir)

        return (dataset, cats, imgs, fights, cat_id_reg, img_id_reg, fight_id_reg)

    def _load_cats(self, src_idx, loaded_cats, dataset, cats, cat_id_reg):
        data = []
        for cat in sorted_by_id(loaded_cats):
            cat_id = (cat['id'],) if src_idx is None else (src_idx, cat['id'])

            cat_id = cat_id_reg.gen(*cat_id)
            if cat_id:
                cat['orig_id'] = cat['id']
                cat['id'] = cat_id
                cat['image_ids'] = set()
                cats[cat_id] = cat
                data.append(cat)

        dataset['categories'] += data
        logger.info(f"{len(data)} categories loaded")

    def _load_fights(self, src_idx, loaded_fights, dataset, fights, fight_id_reg):
        data = []
        for fight in sorted_by_id(loaded_fights):
            fight_id = (fight['id'],) if src_idx is None else (src_idx, fight['id'])
            fight_id = fight_id_reg.gen(*fight_id, label=fight['name'])
            if fight_id:
                fight['orig_id'] = fight['id']
                fight['id'] = fight_id
                fight['image_ids'] = set()
                fights[fight_id] = fight
                data.append(fight)

        dataset['fights'] += data
        logger.info(f"{len(data)} fights loaded")

    def _load_imgs(self, src_idx, loaded_imgs, dataset, imgs, img_id_reg,
                   drop_prob, drop_imgs, img_dir,
                   cat_src_idx, cats, cat_id_reg,
                   fight_src_idx, fights, fight_id_reg):
        n_imgs = len(loaded_imgs)
        if drop_prob > 0:
            rands = np.random.rand(n_imgs)

        data = []
        for i, img in enumerate(sorted_by_id(loaded_imgs)):
            if drop_prob > 0 and rands[i] < drop_prob:
                continue

            img_id = (img['id'],) if src_idx is None else (src_idx, img['id'])
            if drop_imgs is not None:
                id_tuple = tuple(img_id_reg.parse(img_id))
                if id_tuple in drop_imgs:
                    continue

            cat_id = (img['category_id'], )  # todo

            img_id = img_id_reg.gen(*img_id)
            if img_id:
                img['orig_id'] = img['id']
                img['id'] = img_id
                img['file_path'] = osp.join(img_dir, img['file_name'])
                imgs[img_id] = img
                data.append(img)

        dataset['images'] += data
        logger.info(f"{len(data)} images loaded")
