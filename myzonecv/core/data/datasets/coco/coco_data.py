import os
import os.path as osp
from collections import OrderedDict, defaultdict
import json
import copy
import shutil
import warnings

import numpy as np
import skimage.io as io
import skimage.color as color
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from ....utils import print_progress, get_logger
from . import mask as mask_utils

logger = get_logger('coco_data')


def tuple_id(id):  # don't use global id here
    if isinstance(id, ID):
        tid = id.id_tuple
    elif isinstance(id, str) and id:
        id = id.split('.')
        tid = tuple([int(v) for v in id])
    elif isinstance(id, int):
        tid = (id,)
    else:
        tid = id
    tid = check_tuple_id(tid)
    return tid


def check_tuple_id(tid):
    assert isinstance(tid, (tuple, list))
    assert all([isinstance(item, int) for item in tid])
    return tuple(tid)


def sorted_by_id(a):
    def _get_key(item):
        id = item['id']
        tid = tuple_id(id)
        return tid

    if all(['id' in item for item in a]):
        return sorted(a, key=_get_key)
    else:
        return a


def get_interval(total_n, min_interval=1000, max_splits=100):
    split_n = total_n / max_splits
    interval = min_interval
    while interval < split_n:
        interval *= 10
    return interval


class ID:
    def __init__(self, id_tuple, registry, label=None):
        assert isinstance(id_tuple, tuple) and len(id_tuple) > 0
        self.id_tuple = id_tuple              # tuple id: used persistantly
        self.global_idx = registry.add(self)  # int id: used internally or for special purpose
        self.registry = registry
        self.label = label

    def __hash__(self):
        return hash(self.id_tuple)

    def __eq__(self, other):
        return self.id_tuple == other.id_tuple

    def __repr__(self):  # for debug
        s = '.'.join([str(i) for i in self.id_tuple]) if len(self.id_tuple) > 1 else str(self.id_tuple[0])
        if self.label:
            s = f'{s}, label={self.label}'
        return f'ID({s})'

    def __str__(self):  # for print
        return '.'.join([str(i) for i in self.id_tuple]) if len(self.id_tuple) > 1 else str(self.id_tuple[0])

    @property
    def gid(self):
        return self.global_idx

    @property
    def tid(self):
        return self.id_tuple

    def __copy__(self):
        cls = self.__class__
        obj = cls.__new__(cls)
        obj.__dict__.update(self.__dict__)
        return obj

    def __deepcopy__(self, memo):
        cls = self.__class__
        obj = cls.__new__(cls)
        memo[id(self)] = obj
        obj.__dict__.update({
            'id_tuple': copy.deepcopy(self.id_tuple),
            'global_idx': self.global_idx,
            'registry': self.registry
        })
        return obj


class IDRegistry:
    def __init__(self, name, exist_ok=False):
        self.name = name
        self.id_dict = OrderedDict()  # id_tuple => id_obj
        self.id_list = []             # global_idx => id_obj
        self.exist_ok = exist_ok

    @property
    def ids(self):
        return self.id_list

    @property
    def tids(self):
        return self.id_dict.keys()

    @property
    def size(self):
        return len(self.id_list)

    def parse(self, a):
        if isinstance(a, str) and a:
            a = a.split('.')
            if len(a) == 1:
                a = a[0]

        if isinstance(a, (list, tuple)):
            for i in a:
                for r in self.parse(i):
                    yield r
        else:
            try:
                a = int(a)
            except ValueError:
                pass
            else:
                yield a

    def gen(self, *args, label=None):
        id_tuple = tuple(self.parse(args))
        try:
            id_obj = ID(id_tuple, self, label=label)
        except ValueError as e:
            if self.exist_ok:
                return self.id_dict[id_tuple]
            raise e
        else:
            return id_obj

    def add(self, id_obj):
        assert isinstance(id_obj, ID)
        id_tuple = id_obj.id_tuple
        if id_tuple in self.id_dict:
            raise ValueError(f"{id_obj} is already in id registry {self.name}")

        global_idx = len(self.id_list)
        self.id_dict[id_tuple] = id_obj
        self.id_list.append(id_obj)
        return global_idx

    def get(self, id_or_tid_or_gid):
        if isinstance(id_or_tid_or_gid, ID):
            id_obj = id_or_tid_or_gid
            return self.id_dict.get(id_obj.tid)

        if isinstance(id_or_tid_or_gid, (tuple, list)):
            tid = tuple(id_or_tid_or_gid)
            tid = check_tuple_id(tid)
            return self.id_dict.get(tid)
        elif isinstance(id_or_tid_or_gid, int):
            gid = id_or_tid_or_gid
            return self.id_list[gid] if gid < len(self.id_list) else None
        else:
            warnings.warn(f"id_or_tid_or_gid should be tuple(int), list(int), or int, but got {id_or_tid_or_gid}")
            return None

    def getn(self, id_or_tid_or_gid, resort=True):
        """ Type tuple and list are different here.
            If id_or_tid_or_gid is a list, it is treated as multiple ids.
            If id_or_tid_or_gid is a tuple, it is treated as a tuple id.
        """
        if not isinstance(id_or_tid_or_gid, list):
            id_or_tid_or_gid = [id_or_tid_or_gid]

        id_objs = []
        for item in id_or_tid_or_gid:
            id_obj = self.get(item)
            if id_obj is not None:
                id_objs.append(id_obj)

        if resort:
            id_objs = sorted(id_objs, key=lambda a: a.gid)

        return id_objs


class COCOData:
    """
    Main keys in COCO annotation files:
    {
        "categories": [{
            "id",
            "name",
            "supercategory",
            "keypoints",
            "skeleton"
        }],
        "images": [{
            "id",
            "file_name",
            "width",
            "height",
        }],
        "annotations": [{
            "id",
            "image_id",
            "category_id",
            "bbox",
            "area",
            "segmentation",
            "keypoints",
            "num_keypoints"
        }]
    }
    """

    def __init__(self, ann_file=None, img_dir=None, drop_prob=0., drop_imgs=None, share_cat_ids=True):
        """ ann_file (str or list(str))
            img_dir (str or list(str))
        """
        self.ann_file = ann_file
        self.img_dir = img_dir
        self.dataset = None  # {'categories': [cat_dict], 'images': [img_dict], 'annotations': [ann_dict]}
        self.cats = None   # (dict) cat_id => cat_dict
        self.imgs = None   # (dict) img_id => img_dict
        self.anns = None   # (dict) ann_id => ann_dict
        self.cat_id_reg = None
        self.img_id_reg = None
        self.ann_id_reg = None
        self.drop_prob = drop_prob
        self.drop_imgs = drop_imgs if drop_imgs is None else set(drop_imgs)
        self.share_cat_ids = share_cat_ids

        self.load_data(ann_file, img_dir)

    def load_data(self, ann_file, img_dir):
        ann_files = [ann_file] if isinstance(ann_file, str) else ann_file
        img_dirs = [img_dir] if isinstance(img_dir, str) else img_dir
        dataset = None
        imgs = None
        cats = None
        anns = None
        img_id_reg = None
        cat_id_reg = None
        ann_id_reg = None

        if ann_files and img_dirs:
            if len(ann_files) > 1 and len(img_dirs) == 1:
                img_dirs = img_dirs * len(ann_files)
            elif len(ann_files) == 1 and len(img_dirs) > 1:
                ann_files = ann_files * len(img_dirs)
            assert len(ann_files) == len(img_dirs)

            single_source = len(ann_files) == 1
            dataset = defaultdict(list)
            imgs = dict()
            cats = dict()
            anns = dict()
            img_id_reg = IDRegistry('image')
            cat_id_reg = IDRegistry('category', exist_ok=(not single_source) and self.share_cat_ids)
            ann_id_reg = IDRegistry('annotation')

            for i, ann_file in enumerate(ann_files):
                logger.info(f'Loading data from {ann_file}:')
                img_dir = img_dirs[i]

                with open(ann_file) as fin:
                    loaded_dict = json.load(fin)
                    loaded_cats = loaded_dict.get('categories', [])
                    loaded_imgs = loaded_dict.get('images', [])
                    loaded_anns = loaded_dict.get('annotations', [])

                n_imgs = len(loaded_imgs)
                n_anns = len(loaded_anns)
                if self.drop_prob > 0:
                    rands = np.random.rand(n_imgs)

                cat_list = []
                for cat in sorted_by_id(loaded_cats):
                    cat_id = (cat['id'],) if single_source or self.share_cat_ids else (i, cat['id'])
                    cat_id = cat_id_reg.gen(*cat_id)
                    if cat_id:
                        cat['orig_id'] = cat['id']
                        cat['id'] = cat_id
                        cat['image_ids'] = set()
                        cat['annotation_ids'] = set()
                        cats[cat_id] = cat
                        cat_list.append(cat)
                dataset['categories'] += cat_list

                img_list = []
                for j, img in enumerate(sorted_by_id(loaded_imgs)):
                    print_progress(j, n_imgs, interval=get_interval(n_imgs), msg_tmpl='-- scanning images {}/{}', logger_func=logger.info)

                    img_id = (img['id'],) if single_source else (i, img['id'])

                    if self.drop_prob > 0 and rands[j] < self.drop_prob:
                        continue

                    if self.drop_imgs is not None:
                        id_tuple = tuple(img_id_reg.parse(img_id))
                        if id_tuple in self.drop_imgs:
                            continue

                    img_id = img_id_reg.gen(*img_id)
                    if img_id:
                        img['orig_id'] = img['id']
                        img['id'] = img_id
                        img['annotation_ids'] = set()
                        img['file_path'] = osp.join(img_dir, img['file_name'])
                        imgs[img_id] = img
                        img_list.append(img)
                dataset['images'] += img_list
                logger.info(f'{len(img_list)} images read')

                ann_list = []
                for j, ann in enumerate(sorted_by_id(loaded_anns)):
                    print_progress(j, n_anns, interval=get_interval(n_imgs), msg_tmpl='-- scanning annotations {}/{}', logger_func=logger.info)

                    img_id = (ann['image_id'],) if single_source else (i, ann['image_id'])
                    cat_id = (ann['category_id'],) if single_source or self.share_cat_ids else (i, ann['category_id'])
                    img_id = img_id_reg.get(img_id)
                    cat_id = cat_id_reg.get(cat_id)

                    if img_id and cat_id:
                        ann_id = (ann['id'],) if single_source else (i, ann['id'])
                        ann_id = ann_id_reg.gen(*ann_id)
                        if ann_id:
                            ann['orig_id'] = ann['id']
                            ann['orig_image_id'] = ann['image_id']
                            ann['orig_category_id'] = ann['category_id']
                            ann['id'] = ann_id
                            ann['image_id'] = img_id
                            ann['category_id'] = cat_id
                            anns[ann_id] = ann
                            ann_list.append(ann)
                            cats[cat_id]['image_ids'].add(img_id)
                            cats[cat_id]['annotation_ids'].add(ann_id)
                            imgs[img_id]['annotation_ids'].add(ann_id)
                dataset['annotations'] += ann_list
                logger.info(f'{len(ann_list)} annotations read')

        self.ann_file = ann_files
        self.img_dir = img_dirs
        self.dataset = dataset
        self.imgs = imgs
        self.cats = cats
        self.anns = anns
        self.img_id_reg = img_id_reg
        self.cat_id_reg = cat_id_reg
        self.ann_id_reg = ann_id_reg

    def dump_data(self, ann_file, img_dir=None):
        if self.dataset:
            os.makedirs(osp.dirname(ann_file), exist_ok=True)
            if img_dir:
                os.makedirs(img_dir, exist_ok=True)
            dataset = copy.deepcopy(self.dataset)

            for ann in dataset['annotations']:
                ann['id'] = int(str(ann['id']))
                ann['image_id'] = int(str(ann['image_id']))
                ann['category_id'] = int(str(ann['category_id']))

            for img in dataset['images']:
                img['id'] = int(str(img['id']))
                img.pop('annotation_ids')
                img_path = img.pop('file_path')
                if img_path and img_dir and osp.isfile(img_path):
                    shutil.copy(img_path, img_dir)

            for cat in dataset['categories']:
                cat['id'] = int(str(cat['id']))
                cat.pop('image_ids')
                cat.pop('annotation_ids')

            with open(ann_file, 'w') as fout:
                json.dump(dataset, fout)

    def read_img(self, img_or_id_or_path_or_dict):
        if isinstance(img_or_id_or_path_or_dict, np.ndarray):
            img = img_or_id_or_path_or_dict
        else:
            if isinstance(img_or_id_or_path_or_dict, str):
                img_path = img_or_id_or_path_or_dict
            else:
                if not isinstance(img_or_id_or_path_or_dict, dict):
                    if not isinstance(img_or_id_or_path_or_dict, ID):
                        if isinstance(img_or_id_or_path_or_dict, int):
                            img_gid = img_or_id_or_path_or_dict
                            img_id = self.img_id_reg.get(img_gid)
                        elif isinstance(img_or_id_or_path_or_dict, (tuple, list)):
                            img_tid = tuple(img_or_id_or_path_or_dict)
                            img_id = self.img_id_reg.get(img_tid)
                        else:
                            raise TypeError(f"Invalid img_or_id_or_path_or_dict: {img_or_id_or_path_or_dict}")
                    else:
                        img_id = img_or_id_or_path_or_dict
                    img_dict = self.imgs[img_id]
                else:
                    img_dict = img_or_id_or_path_or_dict
                img_path = img_dict['file_path']
            assert osp.isfile(img_path), f"img_path: {img_path}"
            img = io.imread(img_path)  # 0 ~ 255 np.uint8 (RGB)
        if img.ndim == 2:
            img = color.gray2rgb(img)
        return img

    @property
    def categories(self):
        return [self.cats[cat_id] for cat_id in self.cat_id_reg.ids]

    @property
    def images(self):
        return [self.imgs[img_id] for img_id in self.img_id_reg.ids]

    @property
    def annotations(self):
        return [self.anns[ann_id] for ann_id in self.ann_id_reg.ids]

    def get_ann_ids(self, img_ids=[], cat_ids=[], min_area=None, max_area=None):
        img_ids = self.img_id_reg.getn(img_ids)
        cat_ids = self.cat_id_reg.getn(cat_ids)

        ann_ids = set()
        if len(img_ids) > 0:
            ids_ = set([ann_id for img_id in img_ids for ann_id in self.imgs[img_id]['annotation_ids']])
            ann_ids = ann_ids & ids_ if ann_ids else ids_
        if len(cat_ids) > 0:
            ids_ = set([ann_id for cat_id in cat_ids for ann_id in self.cats[cat_id]['annotation_ids']])
            ann_ids = ann_ids & ids_ if ann_ids else ids_

        ann_ids = self.ann_id_reg.ids if not ann_ids else sorted(ann_ids, key=lambda a: a.gid)
        ann_ids = [ann_id for ann_id in ann_ids
                   if ((min_area is None or self.anns[ann_id]['area'] >= min_area)
                       and (max_area is None or self.anns[ann_id]['area'] <= max_area))]
        return ann_ids

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        img_ids = self.img_id_reg.getn(img_ids)
        cat_ids = self.cat_id_reg.getn(cat_ids)

        img_ids = set(img_ids)
        if len(cat_ids) > 0:
            ids_ = set([img_id for cat_id in cat_ids for img_id in self.cats[cat_id]['image_ids']])
            img_ids = img_ids & ids_ if img_ids else ids_

        img_ids = self.img_id_reg.ids if not img_ids else sorted(img_ids, key=lambda a: a.gid)
        return img_ids

    def get_cat_ids(self, cat_ids=[], cat_names=[], supcat_names=[]):
        cat_ids = self.cat_id_reg.getn(cat_ids)
        cat_names = [cat_names] if not isinstance(cat_names, (list, tuple)) else cat_names
        supcat_names = [supcat_names] if not isinstance(supcat_names, (list, tuple)) else supcat_names

        cat_ids = self.cat_id_reg.ids if not cat_ids else sorted(cat_ids, key=lambda a: a.gid)
        cat_ids = [cat_id for cat_id in cat_ids
                   if (len(cat_names) == 0 or self.cats[cat_id]['name'] in cat_names)
                   and (len(supcat_names) == 0 or self.cats[cat_id]['supercategory'] in supcat_names)]
        return cat_ids

    def get_ann(self, ann_id):
        ann_id = self.ann_id_reg.get(ann_id)
        return self.anns[ann_id]

    def get_anns(self, ann_ids=[]):
        ann_ids = self.ann_id_reg.getn(ann_ids)
        return [self.anns[ann_id] for ann_id in ann_ids]

    def get_img(self, img_id):
        img_id = self.img_id_reg.get(img_id)
        return self.imgs[img_id]

    def get_imgs(self, img_ids=[]):
        img_ids = self.img_id_reg.getn(img_ids)
        return [self.imgs[img_id] for img_id in img_ids]

    def get_cat(self, cat_id):
        cat_id = self.cat_id_reg.get(cat_id)
        return self.cats[cat_id]

    def get_cats(self, cat_ids=[]):
        cat_ids = self.cat_id_reg.getn(cat_ids)
        return [self.cats[cat_id] for cat_id in cat_ids]

    def from_results(self, ann_results, keep_ann_id=False):
        if isinstance(ann_results, str) and osp.isfile(ann_results):
            ann_results = json.load(open(ann_results))
            if isinstance(ann_results, dict):
                ann_results = ann_results['annotations']

        coco_data = COCOData()

        imgs = dict()
        cats = dict()
        anns = dict()
        img_id_reg = IDRegistry('image')
        cat_id_reg = IDRegistry('category')
        ann_id_reg = IDRegistry('annotation')

        cat_list = []
        img_list = []
        ann_list = []

        for i, res_ann in enumerate(sorted_by_id(ann_results)):
            res_ann_id = tuple_id(res_ann['id']) if 'id' in res_ann else (i + 1,)
            if keep_ann_id:
                self_ann_id = self.ann_id_reg.get(res_ann_id)
                assert self_ann_id is not None, f"Unexpected ann_id {res_ann_id}"

            if 'image_id' in res_ann:
                res_img_id = tuple_id(res_ann['image_id'])
            elif keep_ann_id:
                res_img_id = tuple_id(self.anns[self_ann_id]['image_id'])
            else:
                raise ValueError(f"image_id not found for res_ann {res_ann}")
            self_img_id = self.img_id_reg.get(res_img_id)
            assert self_img_id is not None, f"Unexpected img_id {res_img_id}"

            if 'category_id' in res_ann:
                res_cat_id = tuple_id(res_ann['category_id'])
            elif keep_ann_id:
                res_cat_id = tuple_id(self.anns[self_ann_id]['category_id'])
            else:
                raise ValueError(f"category_id not found for res_ann {res_ann}")
            self_cat_id = self.cat_id_reg.get(res_cat_id)
            if self_cat_id is None:  # unrecognized category id
                continue

            cat_id = cat_id_reg.get(res_cat_id)
            if cat_id:
                cat = cats[cat_id]
            else:
                cat_id = cat_id_reg.gen(res_cat_id)
                cat = copy.deepcopy(self.cats[self_cat_id])
                cat['id'] = cat_id
                cat['image_ids'] = set()
                cat['annotation_ids'] = set()
                cats[cat_id] = cat
                cat_list.append(cat)

            img_id = img_id_reg.get(res_img_id)
            if img_id:
                img = imgs[img_id]
            else:
                img_id = img_id_reg.gen(res_img_id)
                img = copy.deepcopy(self.imgs[self_img_id])
                img['id'] = img_id
                img['annotation_ids'] = set()
                imgs[img_id] = img
                img_list.append(img)
            cat['image_ids'].add(img_id)

            ann_id = ann_id_reg.get(res_ann_id)
            if ann_id:
                raise ValueError(f"Repeated ann {ann_id}")
            else:
                ann_id = ann_id_reg.gen(res_ann_id)
                if keep_ann_id:
                    ann = self._process_res_ann(res_ann, self.anns[self_ann_id])
                else:
                    ann = self._process_res_ann(res_ann)
                ann['id'] = ann_id
                ann['image_id'] = img_id
                ann['category_id'] = cat_id
                anns[ann_id] = ann
                ann_list.append(ann)
            cat['annotation_ids'].add(ann_id)
            img['annotation_ids'].add(ann_id)

        dataset = dict(categories=sorted_by_id(cat_list),
                       images=sorted_by_id(img_list),
                       annotations=sorted_by_id(ann_list))

        coco_data.img_dir = self.img_dir
        coco_data.dataset = dataset
        coco_data.imgs = imgs
        coco_data.cats = cats
        coco_data.anns = anns
        coco_data.img_id_reg = img_id_reg
        coco_data.cat_id_reg = cat_id_reg
        coco_data.ann_id_reg = ann_id_reg

        return coco_data

    @staticmethod
    def _process_res_ann(res_ann, ref_ann=None):
        res_ann = copy.deepcopy(res_ann)

        ref_ann = copy.deepcopy(ref_ann) if ref_ann else {}
        bbox = ref_ann.get('bbox')
        seg = ref_ann.get('segmentation')
        kpts = ref_ann.get('keypoints')
        area = ref_ann.get('area')

        res_bbox = None
        bbox_seg = None
        bbox_area = None
        if 'bbox' in res_ann:
            res_bbox = res_ann.pop('bbox')
            x0, y0, width, height = res_bbox
            x1, y1 = x0 + width, y0 + height
            bbox_seg = [[x0, y0, x0, y1, x1, y1, x1, y0]]
            bbox_area = width * height

        res_seg = None
        seg_bbox = None
        seg_area = None
        if 'segmentation' in res_ann:
            res_seg = res_ann.pop('segmentation')
            seg_area = mask_utils.area(res_seg)
            seg_bbox = mask_utils.toBbox(res_seg)

        res_kpts = None
        kpts_bbox = None
        kpts_area = None
        if 'keypoints' in res_ann:
            res_kpts = res_ann.pop('keypoints')
            xs, ys = res_kpts[0::3], res_kpts[1::3]
            x0, y0, x1, y1 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
            kpts_bbox = [x0, y0, x1 - x0, y1 - y0]
            kpts_area = (x1 - x0) * (y1 - y0)

        if res_bbox is not None:
            bbox = res_bbox
            area = bbox_area
            if seg is None:
                seg = bbox_seg

        if res_seg is not None:
            seg = res_seg
            if bbox is None:
                bbox = seg_bbox
            if area is None:
                area = seg_area

        if res_kpts is not None:
            kpts = res_kpts
            if bbox is None:
                bbox = kpts_bbox
            if area is None:
                area = kpts_area

        ref_ann['bbox'] = bbox
        ref_ann['segmentation'] = seg
        ref_ann['keypoints'] = kpts
        ref_ann['area'] = area

        res_ann.pop('id', None)
        res_ann.pop('image_id', None)
        res_ann.pop('category_id', None)
        for key, val in res_ann.items():
            ref_ann[key] = val

        return ref_ann


class COCOCustomData(COCOData):
    def __init__(self, post_init_process=None, process_res_ann=None, **kwargs):
        super().__init__(**kwargs)

        if callable(post_init_process):
            post_init_process(self)

        if callable(process_res_ann):
            self._process_res_ann = process_res_ann
