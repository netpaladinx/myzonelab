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

logger = get_logger('myzoneufc_data')


class MyZoneUFCData:
    """
    Holds annotation files for generating various training dataset

    Main keys in annotation files:
    {
        "info": {
            "description" (str),
            "version" (str),
            "date_created" (str)
        },
        "categories": [{
            "id" (int),
            "name" (str: lower),
            "supercategory" (str: lower),
            "keypoints" (list(str)),
            "skeleton" (list(tuple(int)))
        }],
        "images": [{
            "id" (int),
            "file_name" (str),
            "width" (int),
            "height" (int),
        }],
        "fights": [{
            "id" (int),
            "name" (str: lower): "<date>_<person-name>_vs_<person-name>" or "<person-name>_vs_<person-name>" if date is unknown,
            "date" (str): "<yyyymmdd>"
        }],
        "fighters": [{
            "id" (int),  # treat fighters in different fights as different fighters even if they are the same person
            "fight_id" (int),
            "person_name" (str: lower): "<person-name>" or "non-fighter"
        }],
        "attributes": [{
            "id" (int),
            "name" (str: lower),
            "value_set" (list),
            "is_global" (bool)
        }],
        "annotations": [{
            "id" (int),
            "category_id" (int),
            "image_id" (int),
            "fight_id" (int),
            "fighter_id" (int),
            "bbox" (list),  # xywh
            "area" (float),
            "segmentation", # encoded mask
            "mask_file" (str), # file that stores binary mask
            "keypoints" (list(int) or list(float)),
            "num_keypoints" (int),
            "camera_label" (str): "1" ~ "8",
            "camera_index" (int): 0 ~ 7,
            "attributes": [{  # optional
                "attribute_id" (int),
                "value" (any),
                "segmentation", # encoded mask of this attribute
                "mask_file", # file that stores binary mask of this attribute
            }],
            "opponent_ann_id" (int),  # annotation id of the opponent apprearing at the same image
            --
            "extracted_feature"
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
                 share_fighter_ids=True,  # share fighter id space across multiple annotation files
                 share_attr_ids=True,     # share attr id space across multiple annotation files
                 post_init_process=None,  # custom function called at the end of init()
                 process_res_ann=None     # custom function called for processing each res_ann
                 ):
        self.ann_file = ann_file
        self.img_dir = img_dir
        self.drop_prob = drop_prob
        self.drop_imgs = set(drop_imgs) if drop_imgs is not None else None
        self.share_cat_ids = share_cat_ids
        self.share_fight_ids = share_fight_ids
        self.share_fighter_ids = share_fighter_ids
        self.share_attr_ids = share_attr_ids

        # internal data structure
        self.dataset = None   # dict as data container to store list data of 'categories', 'images', 'fights', 'fighters', 'attributes', 'annotations'
        self.cats = None      # dict that maps cat id to cat dict
        self.imgs = None      # dict that maps img id to img dict
        self.fights = None    # dict that maps fight id to fight dict
        self.fighters = None  # dict that maps fighter id to fighter dict
        self.attrs = None     # dict that maps attribute id to attr dict
        self.anns = None      # dict that maps ann id to ann dict
        self.cat_id_reg = None      # registry that holds all cat id
        self.img_id_reg = None      # registry that holds all img id
        self.fight_id_reg = None    # registry that holds all fight id
        self.fighter_id_reg = None  # registry that holds all fighter id
        self.attr_id_reg = None     # registry that holds all attr id
        self.ann_id_reg = None      # registry that holds all ann id

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

            (dataset, cats, imgs, fights, fighters, attrs, anns,
             cat_id_reg, img_id_reg, fight_id_reg, fighter_id_reg, attr_id_reg, ann_id_reg) = self._load_data(ann_files, img_dirs)

            self.ann_file = ann_files
            self.img_dir = img_dirs
            self.dataset = dataset
            self.cats = cats
            self.imgs = imgs
            self.fights = fights
            self.fighters = fighters
            self.attrs = attrs
            self.anns = anns
            self.cat_id_reg = cat_id_reg
            self.img_id_reg = img_id_reg
            self.fight_id_reg = fight_id_reg
            self.fighter_id_reg = fighter_id_reg
            self.attr_id_reg = attr_id_reg
            self.ann_id_reg = ann_id_reg
        else:
            logger.warning("No annotation files or image directories are specified")

    def _load_data(self, ann_files, img_dirs):
        single_source = len(ann_files) == 1
        dataset = defaultdict(list)
        cats = dict()
        imgs = dict()
        fights = dict()
        fighters = dict()
        attrs = dict()
        anns = dict()
        cat_id_reg = IDRegistry('category', exist_ok=(not single_source) and self.share_cat_ids)
        img_id_reg = IDRegistry('image')
        fight_id_reg = IDRegistry('fight', exist_ok=(not single_source) and self.share_fight_ids)
        fighter_id_reg = IDRegistry('fighter', exist_ok=(not single_source) and self.share_fighter_ids)
        attr_id_reg = IDRegistry('attribute', exist_ok=(not single_source) and self.share_attr_ids)
        ann_id_reg = IDRegistry('annotation')

        for i, (ann_file, img_dir) in enumerate(zip(ann_files, img_dirs)):
            logger.info(f"Loading data from {ann_file}:")

            with open(ann_file) as fin:
                loaded_dict = json.load(fin)
                loaded_cats = loaded_dict.get('categories', [])
                loaded_imgs = loaded_dict.get('images', [])
                loaded_fights = loaded_dict.get('fights', [])
                loaded_fighters = loaded_dict.get('fighters', [])
                loaded_attrs = loaded_dict.get('attributes', [])
                loaded_anns = loaded_dict.get('annotations', [])

            cat_src_idx = None if single_source or self.share_cat_ids else i
            self._load_cats(cat_src_idx, loaded_cats, dataset, cats, cat_id_reg)

            img_src_idx = None if single_source else i
            self._load_imgs(img_src_idx, loaded_imgs, dataset, imgs, img_id_reg,
                            self.drop_prob, self.drop_imgs, img_dir)

            fight_src_idx = None if single_source or self.share_fight_ids else i
            self._load_fights(fight_src_idx, loaded_fights, dataset, fights, fight_id_reg)

            fighter_src_idx = None if single_source or self.share_fighter_ids else i
            self._load_fighters(fighter_src_idx, loaded_fighters, dataset, fighters, fighter_id_reg,
                                fight_src_idx, fights, fight_id_reg)

            attr_src_idx = None if single_source or self.share_attr_ids else i
            self._load_attrs(attr_src_idx, loaded_attrs, dataset, attrs, attr_id_reg)

            ann_src_idx = None if single_source else i
            self._load_anns(ann_src_idx, loaded_anns, dataset, anns, ann_id_reg,
                            cat_src_idx, cats, cat_id_reg,
                            img_src_idx, imgs, img_id_reg,
                            fight_src_idx, fights, fight_id_reg,
                            fighter_src_idx, fighters, fighter_id_reg,
                            attr_src_idx, attrs, attr_id_reg, img_dir)

        return (dataset, cats, imgs, fights, fighters, attrs, anns,
                cat_id_reg, img_id_reg, fight_id_reg, fighter_id_reg, attr_id_reg, ann_id_reg)

    def _load_cats(self, src_idx, loaded_cats, dataset, cats, cat_id_reg):
        data = []
        for cat in sorted_by_id(loaded_cats):
            cat_id = (cat['id'],) if src_idx is None else (src_idx, cat['id'])

            cat_id = cat_id_reg.gen(*cat_id)
            if cat_id:
                cat['orig_id'] = cat['id']
                cat['id'] = cat_id
                cat['image_ids'] = set()
                cat['annotation_ids'] = set()
                cats[cat_id] = cat
                data.append(cat)

        dataset['categories'] += data
        logger.info(f"{len(data)} categories loaded")

    def _load_imgs(self, src_idx, loaded_imgs, dataset, imgs, img_id_reg,
                   drop_prob, drop_imgs, img_dir):
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

            img_id = img_id_reg.gen(*img_id)
            if img_id:
                img['orig_id'] = img['id']
                img['id'] = img_id
                img['annotation_ids'] = set()
                img['file_path'] = osp.join(img_dir, img['file_name'])
                imgs[img_id] = img
                data.append(img)

        dataset['images'] += data
        logger.info(f"{len(data)} images loaded")

    def _load_fights(self, src_idx, loaded_fights, dataset, fights, fight_id_reg):
        data = []
        for fight in sorted_by_id(loaded_fights):
            fight_id = (fight['id'],) if src_idx is None else (src_idx, fight['id'])
            fight_id = fight_id_reg.gen(*fight_id, label=fight['name'])
            if fight_id:
                fight['orig_id'] = fight['id']
                fight['id'] = fight_id
                fight['image_ids'] = set()
                fight['annotation_ids'] = set()
                fight['fighter_ids'] = set()
                fights[fight_id] = fight
                data.append(fight)

        dataset['fights'] += data
        logger.info(f"{len(data)} fights loaded")

    def _load_fighters(self, src_idx, loaded_fighters, dataset, fighters, fighter_id_reg,
                       fight_src_idx, fights, fight_id_reg):
        data = []
        for fighter in sorted_by_id(loaded_fighters):
            fight_id = (fighter['fight_id'],) if fight_src_idx is None else (fight_src_idx, fighter['fight_id'])
            fight_id = fight_id_reg.get(fight_id)
            if not fight_id:
                continue

            fighter_id = (fighter['id'],) if src_idx is None else (src_idx, fighter['id'])
            fighter_id = fighter_id_reg.gen(*fighter_id, label=f"{fighter['fight_id']}_{fighter['person_name']}")
            if fighter_id:
                fighter['orig_id'] = fighter['id']
                fighter['orig_fight_id'] = fighter['fight_id']
                fighter['id'] = fighter_id
                fighter['fight_id'] = fight_id
                fighter['image_ids'] = set()
                fighter['annotation_ids'] = set()
                fighters[fighter_id] = fighter
                data.append(fighter)
                fights[fight_id]['fighter_ids'].add(fighter_id)

        dataset['fighters'] += data
        logger.info(f"{len(data)} fighters loaded")

    def _load_attrs(self, src_idx, loaded_attrs, dataset, attrs, attr_id_reg):
        data = []
        for attr in sorted_by_id(loaded_attrs):
            attr_id = (attr['id'],) if src_idx is None else (src_idx, attr['id'])
            attr_id = attr_id_reg.gen(*attr_id, label=attr['name'])
            if attr_id:
                attr['orig_id'] = attr['id']
                attr['id'] = attr_id
                attr['image_ids'] = set()
                attr['annotation_ids'] = set()
                attrs[attr_id] = attr
                data.append(attr)

        dataset['attributes'] += data
        logger.info(f"{len(data)} attributes loaded")

    def _load_anns(self, src_idx, loaded_anns, dataset, anns, ann_id_reg,
                   cat_src_idx, cats, cat_id_reg,
                   img_src_idx, imgs, img_id_reg,
                   fight_src_idx, fights, fight_id_reg,
                   fighter_src_idx, fighters, fighter_id_reg,
                   attr_src_idx, attrs, attr_id_reg, img_dir):
        data = []
        for ann in sorted_by_id(loaded_anns):
            cat_id = (ann['category_id'],) if cat_src_idx is None else (cat_src_idx, ann['category_id'])
            cat_id = cat_id_reg.get(cat_id)
            img_id = (ann['image_id'],) if img_src_idx is None else (img_src_idx, ann['image_id'])
            img_id = img_id_reg.get(img_id)

            if not (cat_id and img_id):
                continue

            if 'fight_id' in ann:
                fight_id = (ann['fight_id'],) if fight_src_idx is None else (fight_src_idx, ann['fight_id'])
                fight_id = fight_id_reg.get(fight_id)
            else:
                fight_id = None

            if 'fighter_id' in ann:
                fighter_id = (ann['fighter_id'],) if fighter_src_idx is None else (fighter_src_idx, ann['fighter_id'])
                fighter_id = fighter_id_reg.get(fighter_id)
            else:
                fighter_id = None

            ann_id = (ann['id'],) if src_idx is None else (src_idx, ann['id'])
            ann_id = ann_id_reg.gen(*ann_id)
            if ann_id:
                ann['orig_id'] = ann['id']
                ann['orig_category_id'] = ann['category_id']
                ann['orig_image_id'] = ann['image_id']
                ann['id'] = ann_id
                ann['category_id'] = cat_id
                ann['image_id'] = img_id

                if 'fight_id' in ann:
                    ann['orig_fight_id'] = ann['fight_id']
                    ann['fight_id'] = fight_id
                if 'fighter_id' in ann:
                    ann['orig_fighter_id'] = ann['fighter_id']
                    ann['fighter_id'] = fighter_id

                orig_attr_ids = []
                attr_ids = []
                attrs = []
                for attr in ann.get('attributes', []):
                    attr_id = (attr['attribute_id'],) if attr_src_idx is None else (attr_src_idx, attr['attribute_id'])
                    attr_id = attr_id_reg.get(attr_id)
                    if attr_id:
                        orig_attr_ids.append(attr['attribute_id'])
                        attr_ids.append(attr_id)
                        attr['attribute_id'] = attr_id
                        attrs.append(attr)
                ann['orig_attr_ids'] = orig_attr_ids
                ann['attributes'] = attrs

                mask_file = ann.get('mask_file')
                if mask_file:
                    ann['mask_path'] = osp.join(img_dir, '../masks', mask_file)

                anns[ann_id] = ann
                data.append(ann)
                cats[cat_id]['image_ids'].add(img_id)
                cats[cat_id]['annotation_ids'].add(ann_id)
                imgs[img_id]['annotation_ids'].add(ann_id)

                if fight_id is not None:
                    fights[fight_id]['image_ids'].add(img_id)
                    fights[fight_id]['annotation_ids'].add(ann_id)
                if fighter_id is not None:
                    fighters[fighter_id]['image_ids'].add(img_id)
                    fighters[fighter_id]['annotation_ids'].add(ann_id)

                for attr_id in attr_ids:
                    attrs[attr_id]['image_ids'].add(img_id)
                    attrs[attr_id]['annotation_ids'].add(ann_id)

        dataset['annotations'] += data
        logger.info(f"{len(data)} annotations loaded")

    def dump_data(self, ann_file, img_dir):
        if self.dataset:
            os.makedirs(osp.dirname(ann_file), exist_ok=True)
            os.makedirs(img_dir, exist_ok=True)
            dataset = copy.deepcopy(self.dataset)

            for cat in dataset['categories']:
                cat['id'] = str(cat['id'])
                cat.pop('image_ids')
                cat.pop('annotation_ids')

            for img in dataset['images']:
                img['id'] = str(img['id'])
                img.pop('annotation_ids')
                img_path = img.pop('file_path')
                if img_path and osp.isfile(img_path):
                    dst_path = osp.join(img_dir, osp.basename(img_path))
                    if not osp.exists(dst_path):
                        shutil.copy(img_path, img_dir)

            for fight in dataset['fights']:
                fight['id'] = str(fight['id'])
                fight.pop('image_ids')
                fight.pop('annotation_ids')
                fight.pop('fighter_ids')

            for fighter in dataset['fighters']:
                fighter['id'] = str(fighter['id'])
                fighter['fight_id'] = str(fighter['fight_id'])
                fighter.pop('image_ids')
                fighter.pop('annotation_ids')

            for attr in dataset['attributes']:
                attr['id'] = str(attr['id'])
                attr.pop('image_ids')
                attr.pop('annotation_ids')

            for ann in dataset['annotations']:
                ann['id'] = str(ann['id'])
                ann['category_id'] = str(ann['category_id'])
                ann['image_id'] = str(ann['image_id'])
                if 'fight_id' in ann:
                    ann['fight_id'] = str(ann['fight_id'])
                if 'fighter_id' in ann:
                    ann['fighter_id'] = str(ann['fighter_id'])
                for attr in ann['attributes']:
                    attr['attribute_id'] = str(attr['attribute_id'])
                mask_path = ann.pop('mask_path')
                if mask_path and osp.isfile(mask_path):
                    mask_dir = osp.join(img_dir, '../masks')
                    dst_path = osp.join(mask_dir, osp.basename(mask_path))
                    if not osp.exists(dst_path):
                        shutil.copy(mask_path, mask_dir)

            with open(ann_file, 'w') as fout:
                json.dump(dataset, fout)

    def read_img(self, img_or_id_or_path_or_dict):
        if isinstance(img_or_id_or_path_or_dict, np.ndarray):
            img = img_or_id_or_path_or_dict
        else:
            if isinstance(img_or_id_or_path_or_dict, str):
                img_path = img_or_id_or_path_or_dict
            else:
                if isinstance(img_or_id_or_path_or_dict, dict):
                    img_dict = img_or_id_or_path_or_dict
                else:
                    if isinstance(img_or_id_or_path_or_dict, ID):
                        img_id = img_or_id_or_path_or_dict
                    else:
                        if isinstance(img_or_id_or_path_or_dict, int):
                            img_gid = img_or_id_or_path_or_dict
                            img_id = self.img_id_reg.get(img_gid)
                        elif isinstance(img_or_id_or_path_or_dict, (tuple, list)):
                            img_tid = tuple(img_or_id_or_path_or_dict)
                            img_id = self.img_id_reg.get(img_tid)
                        else:
                            raise TypeError(f"Invalid img_or_id_or_path_or_dict: {img_or_id_or_path_or_dict}")
                    img_dict = self.imgs[img_id]
                img_path = img_dict['file_path']
            assert osp.isfile(img_path), f"img_path {img_path} is not a file"
            img = io.imread(img_path)  # 0 ~ 255 np.uint8 (RGB)
        if img.ndim == 2:
            img = color.gray2rgb(img)
        return img

    def get_mask(self, ann_or_id_or_path_or_np, as_bool=True):
        if isinstance(ann_or_id_or_path_or_np, np.ndarray):  # mask array
            mask = ann_or_id_or_path_or_np
        else:
            mask_path, seg, ann_dict = None, None, None
            if isinstance(ann_or_id_or_path_or_np, str):  # mask file
                mask_path = ann_or_id_or_path_or_np
            else:
                if isinstance(ann_or_id_or_path_or_np, dict):  # ann dict
                    ann_dict = ann_or_id_or_path_or_np
                else:
                    if isinstance(ann_or_id_or_path_or_np, ID):
                        ann_id = ann_or_id_or_path_or_np
                    else:
                        if isinstance(ann_or_id_or_path_or_np, int):
                            ann_gid = ann_or_id_or_path_or_np
                            ann_id = self.ann_id_reg.get(ann_gid)
                        elif isinstance(ann_or_id_or_path_or_np, (tuple, list)):
                            ann_tid = ann_or_id_or_path_or_np
                            ann_id = self.ann_id_reg.get(ann_tid)
                        else:
                            raise TypeError(f"Invalid ann_or_id_or_path_or_np: {ann_or_id_or_path_or_np}")
                    ann_dict = self.anns[ann_id]
                mask_path = ann_dict.get('mask_path')
                seg = ann_dict.get('segmentation')
            if mask_path:
                assert osp.isfile(mask_path), f"mask_path {mask_path} is not a file"
                mask = skimage.img_as_float(io.imread(mask_path))  # 0 ~ 1 np.float64
            elif seg and ann_dict:
                img = self.imgs[ann_dict['image_id']]
                img_width = img['width']
                img_height = img['height']
                mask = seg2mask(seg, img_width, img_height)
            else:
                return None
        if as_bool:
            mask = mask > 0.5  # np.bool
        return mask

    def list_categories(self):
        return [self.cats[cat_id] for cat_id in self.cat_id_reg.ids]

    def list_images(self):
        return [self.imgs[img_id] for img_id in self.img_id_reg.ids]

    def list_fights(self):
        return [self.fights[fight_id] for fight_id in self.fight_id_reg.ids]

    def list_fighters(self):
        return [self.fighters[fighter_id] for fighter_id in self.fighter_id_reg.ids]

    def list_attributes(self):
        return [self.attrs[attr_id] for attr_id in self.attr_id_reg.ids]

    def list_annotations(self):
        return [self.anns[ann_id] for ann_id in self.ann_id_reg.ids]

    def get_cat_ids(self, cat_ids=[], cat_names=[], supcat_names=[]):
        cat_ids = self.cat_id_reg.getn(cat_ids)
        cat_names = [cat_names] if not isinstance(cat_names, (list, tuple)) else cat_names
        supcat_names = [supcat_names] if not isinstance(supcat_names, (list, tuple)) else supcat_names

        cat_ids = self.cat_id_reg.ids if not cat_ids else sorted(cat_ids, key=lambda a: a.gid)
        cat_ids = [cat_id for cat_id in cat_ids
                   if (len(cat_names) == 0 or self.cats[cat_id]['name'] in cat_names)
                   and (len(supcat_names) == 0 or self.cats[cat_id]['supercategory'] in supcat_names)]
        return cat_ids

    def get_img_ids(self, img_ids=[], cat_ids=[], fight_ids=[], fighter_ids=[], attr_ids=[]):
        img_ids = self.img_id_reg.getn(img_ids)
        cat_ids = self.cat_id_reg.getn(cat_ids)
        fight_ids = self.fight_id_reg.getn(fight_ids)
        fighter_ids = self.fighter_id_reg.getn(fighter_ids)
        attr_ids = self.attr_id_reg.getn(attr_ids)

        img_ids = set(img_ids)
        if len(cat_ids) > 0:
            ids_ = set([img_id for cat_id in cat_ids for img_id in self.cats[cat_id]['image_ids']])
            img_ids = img_ids & ids_ if img_ids else ids_
        if len(fight_ids) > 0:
            ids_ = set([img_id for fight_id in fight_ids for img_id in self.fights[fight_id]['image_ids']])
            img_ids = img_ids & ids_ if img_ids else ids_
        if len(fighter_ids) > 0:
            ids_ = set([img_id for fighter_id in fighter_ids for img_id in self.fighters[fighter_id]['image_ids']])
            img_ids = img_ids & ids_ if img_ids else ids_
        if len(attr_ids) > 0:
            ids_ = set([img_id for attr_id in attr_ids for img_id in self.attrs[attr_id]['image_ids']])
            img_ids = img_ids & ids_ if img_ids else ids_

        img_ids = self.img_id_reg.ids if not img_ids else sorted(img_ids, key=lambda a: a.gid)
        return img_ids

    def get_fight_ids(self, fight_ids=[], fight_names=[], fight_dates=[]):
        fight_ids = self.fight_id_reg.getn(fight_ids)
        fight_names = [fight_names] if not isinstance(fight_names, (list, tuple)) else fight_names
        fight_dates = [fight_dates] if not isinstance(fight_dates, (list, tuple)) else fight_dates

        fight_ids = self.fight_id_reg.ids if not fight_ids else sorted(fight_ids, key=lambda a: a.gid)
        fight_ids = [fight_id for fight_id in fight_ids
                     if (len(fight_names) == 0 or self.fights[fight_id]['name'] in fight_names)
                     and (len(fight_dates) == 0 or self.fights[fight_id]['date'] in fight_dates)]
        return fight_ids

    def get_fighter_ids(self, fighter_ids=[], fight_ids=[]):
        fighter_ids = self.fighter_id_reg.getn(fighter_ids)
        fight_ids = self.fight_id_reg.getn(fight_ids)

        fighter_ids = set(fighter_ids)
        if len(fight_ids) > 0:
            ids_ = set([fighter_id for fight_id in fight_ids for fighter_id in self.fights[fight_id]['fighter_ids']])
            fighter_ids = fighter_ids & ids_ if fighter_ids else ids_

        fighter_ids = self.fighter_id_reg.ids if not fighter_ids else sorted(fighter_ids, key=lambda a: a.gid)
        return fighter_ids

    def get_attr_ids(self, attr_ids=[], attr_names=[], is_global=None):
        attr_ids = self.attr_id_reg.getn(attr_ids)
        attr_names = [attr_names] if not isinstance(attr_names, (list, tuple)) else attr_names

        attr_ids = self.attr_id_reg.ids if not attr_ids else sorted(attr_ids, key=lambda a: a.gid)
        attr_ids = [attr_id for attr_id in attr_ids
                    if (len(attr_names) == 0 or self.attrs[attr_id]['name'] in attr_names)
                    and (is_global is None or is_global == self.attrs[attr_id]['is_global'])]
        return attr_ids

    def get_ann_ids(self, img_ids=[], cat_ids=[], fight_ids=[], fighter_ids=[], attr_ids=[], min_area=None, max_area=None):  # todo
        img_ids = self.img_id_reg.getn(img_ids)
        cat_ids = self.cat_id_reg.getn(cat_ids)
        fight_ids = self.fight_id_reg.getn(fight_ids)
        fighter_ids = self.fighter_id_reg.getn(fighter_ids)
        attr_ids = self.attr_id_reg.getn(attr_ids)

        ann_ids = set()
        if len(img_ids) > 0:
            ids_ = set([ann_id for img_id in img_ids for ann_id in self.imgs[img_id]['annotation_ids']])
            ann_ids = ann_ids & ids_ if ann_ids else ids_
        if len(cat_ids) > 0:
            ids_ = set([ann_id for cat_id in cat_ids for ann_id in self.cats[cat_id]['annotation_ids']])
            ann_ids = ann_ids & ids_ if ann_ids else ids_
        if len(fight_ids) > 0:
            ids_ = set([ann_id for fight_id in fight_ids for ann_id in self.fights[fight_id]['annotation_ids']])
            ann_ids = ann_ids & ids_ if ann_ids else ids_
        if len(fighter_ids) > 0:
            ids_ = set([ann_id for fighter_id in fighter_ids for ann_id in self.fighters[fighter_id]['annotation_ids']])
            ann_ids = ann_ids & ids_ if ann_ids else ids_
        if len(attr_ids) > 0:
            ids_ = set([ann_id for attr_id in attr_ids for ann_id in self.attrs[attr_id]['annotation_ids']])
            ann_ids = ann_ids & ids_ if ann_ids else ids_

        ann_ids = self.ann_id_reg.ids if not ann_ids else sorted(ann_ids, key=lambda a: a.gid)
        ann_ids = [ann_id for ann_id in ann_ids
                   if ((min_area is None or self.anns[ann_id]['area'] >= min_area)
                       and (max_area is None or self.anns[ann_id]['area'] <= max_area))]
        return ann_ids

    def get_cat(self, cat_id=None, ann_id=None):
        if cat_id is None:
            assert ann_id is not None
            cat_id = self.get_ann(ann_id)['category_id']
        cat_id = self.cat_id_reg.get(cat_id)
        return self.cats[cat_id]

    def get_cats(self, cat_ids=[], ann_ids=[]):
        if len(cat_ids) == 0:
            assert len(ann_ids) > 0
            cat_ids = [self.get_ann(ann_id)['category_id'] for ann_id in ann_ids]
        cat_ids = self.cat_id_reg.getn(cat_ids, resort=False)
        return [self.cats[cat_id] for cat_id in cat_ids]

    def get_img(self, img_id=None, ann_id=None):
        if img_id is None:
            assert ann_id is not None
            img_id = self.get_ann(ann_id)['image_id']
        img_id = self.img_id_reg.get(img_id)
        return self.imgs[img_id]

    def get_imgs(self, img_ids=[], ann_ids=[]):
        if len(img_ids) == 0:
            assert len(ann_ids) > 0
            img_ids = [self.get_ann(ann_id)['image_id'] for ann_id in ann_ids]
        img_ids = self.img_id_reg.getn(img_ids, resort=False)
        return [self.imgs[img_id] for img_id in img_ids]

    def get_fight(self, fight_id=None, ann_id=None):
        if fight_id is None:
            assert ann_id is not None
            fight_id = self.get_ann(ann_id)['fight_id']
        fight_id = self.fight_id_reg.get(fight_id)
        return self.fights[fight_id]

    def get_fights(self, fight_ids=[], ann_ids=[]):
        if len(fight_ids) == 0:
            assert len(ann_ids) > 0
            fight_ids = [self.get_ann(ann_id)['fight_id'] for ann_id in ann_ids]
        fight_ids = self.fight_id_reg.getn(fight_ids, resort=False)
        return [self.fights[fight_id] for fight_id in fight_ids]

    def get_fighter(self, fighter_id=None, ann_id=None):
        if fighter_id is None:
            assert ann_id is not None
            fighter_id = self.get_ann(ann_id)['fighter_id']
        fighter_id = self.fighter_id_reg.get(fighter_id)
        return self.fighters[fighter_id]

    def get_fighters(self, fighter_ids=[], ann_ids=[]):
        if len(fighter_ids) == 0:
            assert len(ann_ids) > 0
            fighter_ids = [self.get_ann(ann_id)['fighter_id'] for ann_id in ann_ids]
        fighter_ids = self.fighter_id_reg.getn(fighter_ids, resort=False)
        return [self.fighters[fighter_id] for fighter_id in fighter_ids]

    def get_attr(self, attr_id):
        attr_id = self.attr_id_reg.get(attr_id)
        return self.attrs[attr_id]

    def get_attrs(self, attr_ids=[]):
        attr_ids = self.attr_id_reg.getn(attr_ids, resort=False)
        return [self.attrs[attr_id] for attr_id in attr_ids]

    def get_ann(self, ann_id):
        ann_id = self.ann_id_reg.get(ann_id)
        return self.anns[ann_id]

    def get_anns(self, ann_ids=[]):
        ann_ids = self.ann_id_reg.getn(ann_ids, resort=False)
        return [self.anns[ann_id] for ann_id in ann_ids]

    def is_nonfighter(self, fighter_id):
        fighter = self.get_fighter(fighter_id)
        return fighter.get('person_name') == NONFIGHTER_LABEL

    def get_opponent(self, fighter_id):
        fighter_id = self.get_fighter_ids([fighter_id])[0]
        if self.is_nonfighter(fighter_id):
            return None

        fighter = self.fighters[fighter_id]
        fight = self.fights[fighter['fight_id']]
        opponent_id = [i for i in fight['fighter_ids'] if i != fighter_id and not self.is_nonfighter(i)]
        if len(opponent_id) == 1:
            return self.fighters[opponent_id[0]]
        elif len(opponent_id) == 0:
            logger.warning(f"Cannot find the opponent of fighter {fighter_id}")
        else:
            logger.warning(f"Find more than one opponents of fighter {fighter_id}: {opponent_id}")
        return None

    def from_results(self, ann_results, keep_ann_id=False, cat_ids=None, img_ids=None):
        if isinstance(ann_results, str) and osp.isfile(ann_results):
            ann_results = json.load(open(ann_results))
            if isinstance(ann_results, dict):
                ann_results = ann_results['annotations']

        myzoneufc_data = MyZoneUFCData()

        dataset = defaultdict(list)
        cat_data = []
        img_data = []
        fight_data = []
        fighter_data = []
        attr_data = []
        ann_data = []

        cats = dict()
        imgs = dict()
        fights = dict()
        fighters = dict()
        attrs = dict()
        anns = dict()
        cat_id_reg = IDRegistry('category')
        img_id_reg = IDRegistry('image')
        fight_id_reg = IDRegistry('fight')
        fighter_id_reg = IDRegistry('fighter')
        attr_id_reg = IDRegistry('attribute')
        ann_id_reg = IDRegistry('annotation')

        if cat_ids is not None:
            for res_cat_id in cat_ids:
                res_cat_id = tuple_id(res_cat_id)
                self_cat_id = self.cat_id_reg.get(res_cat_id)
                if self_cat_id is None:
                    logger.warning(f"Unexpected category_id {res_cat_id}")
                    continue

                cat_id = cat_id_reg.get(res_cat_id)
                if not cat_id:
                    cat_id = cat_id_reg.gen(res_cat_id)
                    cat = copy.deepcopy(self.cats[self_cat_id])
                    cat['id'] = cat_id
                    cat['image_ids'] = set()
                    cat['annotation_ids'] = set()
                    cats[cat_id] = cat
                    cat_data.append(cat)

        if img_ids is not None:
            for res_img_id in img_ids:
                res_img_id = tuple_id(res_img_id)
                self_img_id = self.img_id_reg.get(res_img_id)
                if self_img_id is None:
                    logger.warning(f"Unexpected image_id {res_img_id}")
                    continue

                img_id = img_id_reg.get(res_img_id)
                if not img_id:
                    img_id = img_id_reg.gen(res_img_id)
                    img = copy.deepcopy(self.imgs[self_img_id])
                    img['id'] = img_id
                    img['image_ids'] = set()
                    img['annotation_ids'] = set()
                    imgs[img_id] = img
                    img_data.append(img)

        for i, res_ann in enumerate(sorted_by_id(ann_results)):
            res_ann_id = tuple_id(res_ann['id']) if 'id' in res_ann else (i + 1,)
            if keep_ann_id:
                self_ann_id = self.ann_id_reg.get(res_ann_id)
                assert self_ann_id is not None, f"Unexpected ann_id {res_ann_id}"
                self_ann = self.anns[self_ann_id]

            if 'category_id' in res_ann:
                res_cat_id = tuple_id(res_ann['category_id'])
            elif keep_ann_id:
                res_cat_id = tuple_id(self_ann['category_id'])
            else:
                raise ValueError(f"category_id not found for res_ann {res_ann}")
            self_cat_id = self.cat_id_reg.get(res_cat_id)
            if self_cat_id is None:
                logger.warning(f"Unexpected category_id {res_cat_id}")
                continue

            if 'image_id' in res_ann:
                res_img_id = tuple_id(res_ann['image_id'])
            elif keep_ann_id:
                res_img_id = tuple_id(self_ann['image_id'])
            else:
                raise ValueError(f"image_id not found for res_ann {res_ann}")
            self_img_id = self.img_id_reg.get(res_img_id)
            if self_img_id is None:
                logger.warning(f"Unexpected image_id {res_img_id}")
                continue

            if 'fight_id' in res_ann:
                res_fight_id = tuple_id(res_ann['fight_id'])
            elif keep_ann_id:
                res_fight_id = tuple_id(self_ann['fight_id']) if 'fight_id' in self_ann else None
            else:
                res_fight_id = None
            if res_fight_id is not None:
                self_fight_id = self.fight_id_reg.get(res_fight_id)
                if self_fight_id is None:
                    logger.warning(f"Unexpected fight_id {res_fight_id}")
                    continue

            if 'fighter_id' in res_ann:
                res_fighter_id = tuple_id(res_ann['fighter_id'])
            elif keep_ann_id:
                res_fighter_id = tuple_id(self_ann['fighter_id']) if 'fighter_id' in self_ann else None
            else:
                res_fighter_id = None
            if res_fighter_id is not None:
                self_fighter_id = self.fighter_id_reg.get(res_fighter_id)
                if self_fighter_id is None:
                    logger.warning(f"Unexpected fighter_id {res_fighter_id}")
                    continue

            res_attr_ids = []
            self_attr_ids = []
            attributes = []
            if 'attributes' in res_ann:
                attributes = res_ann['attributes']
                res_attr_ids = [tuple_id(attr['attribute_id']) for attr in attributes]
            elif keep_ann_id:
                attributes = self.anns[self_ann_id]['attributes']
                res_attr_ids = [tuple_id(attr['attribute_id']) for attr in attributes]
            for res_attr_id in res_attr_ids:
                self_attr_id = self.attr_id_reg.get(res_attr_id)
                if self_attr_id is None:
                    logger.warning(f"Unexpected attribute_id {res_attr_id}")
                    break
                self_attr_ids.append(self_attr_id)
            if len(res_attr_ids) != len(self_attr_ids):
                continue

            cat_id = cat_id_reg.get(res_cat_id)
            if cat_id:  # cat_id exists
                cat = cats[cat_id]
            else:  # new cat_id
                cat_id = cat_id_reg.gen(res_cat_id)
                cat = copy.deepcopy(self.cats[self_cat_id])
                cat['id'] = cat_id
                cat['image_ids'] = set()
                cat['annotation_ids'] = set()
                cats[cat_id] = cat
                cat_data.append(cat)

            img_id = img_id_reg.get(res_img_id)
            if img_id:  # img_id exists
                img = imgs[img_id]
            else:  # new img_id
                img_id = img_id_reg.gen(res_img_id)
                img = copy.deepcopy(self.imgs[self_img_id])
                img['id'] = img_id
                img['annotation_ids'] = set()
                imgs[img_id] = img
                img_data.append(img)

            if res_fight_id is not None:
                fight_id = fight_id_reg.get(res_fight_id)
                if fight_id:  # fight_id exists
                    fight = fights[fight_id]
                else:  # new fight_id
                    fight_id = fight_id_reg.gen(res_fight_id)
                    fight = copy.deepcopy(self.fights[self_fight_id])
                    fight['id'] = fight_id
                    fight['image_ids'] = set()
                    fight['annotation_ids'] = set()
                    fight['fighter_ids'] = set()
                    fights[fight_id] = fight
                    fight_data.append(fight)
            else:
                fight_id = None

            if res_fighter_id is not None:
                fighter_id = fighter_id_reg.get(res_fighter_id)
                if fighter_id:  # fighter_id exists
                    fighter = fighters[fighter_id]
                else:  # new fighter_id
                    fighter_id = fighter_id_reg.gen(res_fighter_id)
                    fighter = copy.deepcopy(self.fighters[self_fighter_id])
                    fighter['id'] = fighter_id
                    fighter['fight_id'] = fight_id
                    fighter['image_ids'] = set()
                    fighter['annotation_ids'] = set()
                    fighters[fighter_id] = fighter
                    fighter_data.append(fighter)
            else:
                fighter_id = None

            attr_ids = []
            attrs = []
            for res_attr_id, self_attr_id in zip(res_attr_ids, self_attr_ids):
                attr_id = attr_id_reg.get(res_attr_id)
                if attr_id:  # attr_id exists
                    attr = attrs[attr_id]
                else:  # new attr_id
                    attr_id = attr_id_reg.gen(res_attr_id)
                    attr = copy.deepcopy(self.attrs[self_attr_id])
                    attr['id'] = attr_id
                    attr['image_ids'] = set()
                    attr['annotation_ids'] = set()
                    attrs[attr_id] = attr
                    attr_data.append(attr)
                attr_ids.append(attr_id)
                attrs.append(attr)

            ann_id = ann_id_reg.get(res_ann_id)
            if ann_id:  # ann_id exists
                raise ValueError(f"Repeated ann {ann_id}")
            else:
                ann_id = ann_id_reg.gen(res_ann_id)
                if keep_ann_id:
                    ann = self._process_res_ann(res_ann, ref_ann=self.anns[self_ann_id])
                else:
                    ann = self._process_res_ann(res_ann)
                ann['id'] = ann_id
                ann['category_id'] = cat_id
                ann['image_id'] = img_id
                if fight_id is not None:
                    ann['fight_id'] = fight_id
                if fighter_id is not None:
                    ann['fighter_id'] = fighter_id
                ann['attributes'] = []
                for attr_id, attribute in zip(attr_ids, attributes):
                    attribute = copy.deepcopy(attribute)
                    attribute['attribute_id'] = attr_id
                    ann['attributes'].append(attribute)
                anns[ann_id] = ann
                ann_data.append(ann)

            cat['image_ids'].add(img_id)
            cat['annotation_ids'].add(ann_id)
            img['annotation_ids'].add(ann_id)
            if fight_id is not None:
                fight['image_ids'].add(img_id)
                fight['annotation_ids'].add(ann_id)
                fight['fighter_ids'].add(fighter_id)
            if fighter_id is not None:
                fighter['image_ids'].add(img_id)
                fighter['annotation_ids'].add(ann_id)
            for attr in attrs:
                attr['image_ids'].add(img_id)
                attr['annotation_ids'].add(ann_id)

        dataset = {
            'categories': sorted_by_id(cat_data),
            'images': sorted_by_id(img_data),
            'fights': sorted_by_id(fight_data),
            'fighters': sorted_by_id(fighter_data),
            'attributes': sorted_by_id(attr_data),
            'annotations': sorted_by_id(ann_data)
        }

        myzoneufc_data.img_dir = self.img_dir
        myzoneufc_data.dataset = dataset
        myzoneufc_data.cats = cats
        myzoneufc_data.imgs = imgs
        myzoneufc_data.fights = fights
        myzoneufc_data.fighters = fighters
        myzoneufc_data.attrs = attrs
        myzoneufc_data.anns = anns
        myzoneufc_data.cat_id_reg = cat_id_reg
        myzoneufc_data.img_id_reg = img_id_reg
        myzoneufc_data.fight_id_reg = fight_id_reg
        myzoneufc_data.fighter_id_reg = fighter_id_reg
        myzoneufc_data.attr_id_reg = attr_id_reg
        myzoneufc_data.ann_id_reg = ann_id_reg

        return myzoneufc_data

    @staticmethod
    def _process_res_ann(res_ann, ref_ann=None):
        res_ann = copy.deepcopy(res_ann)
        out_ann = copy.deepcopy(ref_ann) if ref_ann else {}

        res_area = res_ann.pop('area', None)
        res_bbox = res_ann.pop('bbox', None)
        res_seg = res_ann.pop('segmentation', None)
        res_kpts = res_ann.pop('keypoints', None)

        if res_bbox is None and res_seg is not None:
            res_bbox = mask_utils.toBbox(res_seg)

        if res_area is None and res_bbox is not None:
            res_area = bbox_area(res_bbox)

        if res_bbox is not None:
            out_ann['bbox'] = res_bbox
        if res_area is not None:
            out_ann['area'] = res_area
        if res_seg is not None:
            out_ann['segmentation'] = res_seg
        if res_kpts is not None:
            out_ann['keypoints'] = res_kpts

        for key, val in res_ann.items():
            out_ann[key] = val

        return out_ann
