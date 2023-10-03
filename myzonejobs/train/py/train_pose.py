#!/usr/bin/env python
import os.path as osp
import argparse
import json

from myzonecv.apis import train, get_config_file, Config, utils as U


# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
CONFIG_NAME = 'pose_hrnet_w48_coco_384x288'

CONF_STR = """
{"data.train.data_loader.batch_size": 4,
 "data.train.data_loader.num_workers": 0,
 "data.val.data_loader.batch_size": 4,
 "data.val.data_loader.num_workers": 0,
 "optim.max_epochs": 1,
 "optim.lr_scheduler.step": [30,40],
 "validator.val_at_start": false,
 "summarizer.interval": 10}
"""

WORK_DIR = './workspace/experiments/' + CONFIG_NAME

ANN_FILE = './workspace/data_zoo/trainval/person_keypoints_debug/annotations.json'
IMG_DIR = './workspace/data_zoo/trainval/person_keypoints_debug/images'
VAL_ANN_FILE = './workspace/data_zoo/trainval/person_keypoints_debug/annotations.json'
VAL_IMG_DIR = './workspace/data_zoo/trainval/person_keypoints_debug/images'

# ANN_FILE = './workspace/data_zoo/trainval/person_keypoints_debug3/annotations.json,' \
#            './workspace/data_zoo/trainval/person_keypoints_debug3/annotations.json'
# IMG_DIR = './workspace/data_zoo/trainval/person_keypoints_debug3/images,' \
#           './workspace/data_zoo/trainval/person_keypoints_debug3/images'
# VAL_ANN_FILE = None
# VAL_IMG_DIR = None

# PRETRAINED_PATH = './workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
PRETRAINED_PATH = './workspace/model_zoo/pose_models/hrnet_w48_coco_384x288-314c8528_20200708.pth'
PRETRAINED_DST_PREFIX = None

#PRETRAINED_PATH = './workspace/model_zoo/imagenet_pretrained/hrnet_w32-36af842e.pth'
#PRETRAINED_DST_PREFIX = 'backbone'

#FEATURE_HEAD_PRETRAINED_PATH = './workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_featurehead_128-v1.pth'
FEATURE_HEAD_PRETRAINED_PATH = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=CONFIG_NAME)
    parser.add_argument('--extra-conf-path', type=str, default=None)
    parser.add_argument('--extra-conf-str', type=str, default=CONF_STR)
    parser.add_argument('--work-dir', type=str, default=WORK_DIR)
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--ann-file', type=str, default=ANN_FILE)
    parser.add_argument('--img-dir', type=str, default=IMG_DIR)
    parser.add_argument('--val-ann-file', type=str, default=VAL_ANN_FILE)
    parser.add_argument('--val-img-dir', type=str, default=VAL_IMG_DIR)
    parser.add_argument('--split-train-val', action='store_true', default=False)
    parser.add_argument('--split-ratio', type=float, default=9)
    parser.add_argument('--pretrained', type=str, default=PRETRAINED_PATH)
    parser.add_argument('--pretrained-src-prefix', type=str, default=None)
    parser.add_argument('--pretrained-dst-prefix', type=str, default=PRETRAINED_DST_PREFIX)
    parser.add_argument('--feature-head-pretrained', type=str, default=FEATURE_HEAD_PRETRAINED_PATH)
    parser.add_argument('--use-diagnosis', action='store_true')
    args = parser.parse_args()

    config = get_config_file(args.config)
    kwargs = {'work_dir': args.work_dir, 'timestamp': args.timestamp}

    path = args.extra_conf_path
    if path and osp.isfile(path):
        extra_config = Config.from_file(path)
        kwargs.update(extra_config.to_dict())

    jsonstr = args.extra_conf_str
    if jsonstr:
        try:
            kwargs.update(json.loads(jsonstr))
        except Exception as e:
            print(e)
            pass

    ann_file = U.parse_path(args.ann_file, is_file=True)
    img_dir = U.parse_path(args.img_dir, is_dir=True)
    if args.split_train_val:
        assert args.split_ratio > 0
        kwargs['data'] = {
            'data_source': {
                'ann_file': ann_file,
                'img_dir': img_dir,
                'drop_prob': 1. / args.split_ratio
            }
        }
    else:
        assert args.val_ann_file is not None and args.val_img_dir is not None
        val_ann_file = U.parse_path(args.val_ann_file, is_file=True)
        val_img_dir = U.parse_path(args.val_img_dir, is_dir=True)
        kwargs['data'] = {
            'train.data_source': {
                'ann_file': ann_file,
                'img_dir': img_dir
            },
            'val.data_source': {
                'ann_file': val_ann_file,
                'img_dir': val_img_dir
            }
        }

    if args.pretrained:
        assert osp.isfile(args.pretrained)
        kwargs['model.init_cfg'] = {
            'type': 'pretrained',
            'path': args.pretrained,
            'src_prefix': args.pretrained_src_prefix,
            'dst_prefix': args.pretrained_dst_prefix
        }

    if args.feature_head_pretrained:
        assert osp.isfile(args.feature_head_pretrained)
        kwargs['model.feature_head'] = {
            'type': 'pose_feature',
            'build_from_init': True,
            'init_cfg': {
                'type': 'pretrained',
                'path': args.feature_head_pretrained
            }
        }

    kwargs['use_diagnosis'] = args.use_diagnosis

    train(config, **kwargs)


if __name__ == '__main__':
    main()
