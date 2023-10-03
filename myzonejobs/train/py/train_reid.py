#!/usr/bin/env python
import os.path as osp
import argparse
import json

from myzonecv.apis import train, get_config_file, Config, utils as U

CONFIG_NAME = 'reid/reid_resnet50_256x192_c16'

CONF_STR = """
{"data.train.data_loader.num_workers": 4,
 "data.train.data_params.n_samples_per_id": 3,
 "data.train.data_params.min_ids_per_batch": 3,
 "data.train.data_params.batch_size": 9,
 "data.train_with_recon.data_loader.num_workers": 4,
 "data.train_with_recon.data_params.n_samples_per_id": 3,
 "data.train_with_recon.data_params.min_ids_per_batch": 3,
 "data.train_with_recon.data_params.batch_size": 9}
"""

WORK_DIR = './workspace/experiments/' + CONFIG_NAME

ANN_FILE = './workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_clean_kpts-seg1/train/annotations.json'
IMG_DIR = './workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_clean_kpts-seg1/train/images'
VAL_ANN_FILE = './workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_clean_kpts-seg1/val/annotations.json'
VAL_IMG_DIR = './workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_clean_kpts-seg1/val/images'
# ANN_FILE = './workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_clean_kpts-seg1/train/annotations.json,' \
#            './workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_noisy_kpts-seg1/train/annotations.json'
# IMG_DIR = './workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_clean_kpts-seg1/train/images,' \
#           './workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_noisy_kpts-seg1/train/images'
# VAL_ANN_FILE = './workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_clean_kpts-seg1/val/annotations.json,' \
#                './workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_noisy_kpts-seg1/val/annotations.json'
# VAL_IMG_DIR = './workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_clean_kpts-seg1/val/images,' \
#               './workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_noisy_kpts-seg1/val/images'

TR_VAL_ANN_FILE = './workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_transfer_kpts-seg1/val/annotations.json'
TR_VAL_IMG_DIR = './workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_transfer_kpts-seg1/val/images'

BACKBONE_PRETRAINED_PATH = './workspace/model_zoo/imagenet_pretrained/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'
BACKBONE_PRETRAINED_SRC_PREFIX = 'backbone'
BACKBONE_PRETRAINED_DST_PREFIX = 'backbone'

#DISCRIMINATOR_PRETRAINED_PATH = './workspace/model_zoo/imagenet_pretrained/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'
DISCRIMINATOR_PRETRAINED_PATH = None
DISCRIMINATOR_PRETRAINED_SRC_PREFIX = 'backbone'
DISCRIMINATOR_PRETRAINED_DST_PREFIX = 'discriminator'


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
    parser.add_argument('--transfer-val-ann-file', type=str, default=TR_VAL_ANN_FILE)
    parser.add_argument('--transfer-val-img-dir', type=str, default=TR_VAL_IMG_DIR)
    parser.add_argument('--backbone-pretrained', type=str, default=BACKBONE_PRETRAINED_PATH)
    parser.add_argument('--backbone-pretrained-src-prefix', type=str, default=BACKBONE_PRETRAINED_SRC_PREFIX)
    parser.add_argument('--backbone-pretrained-dst-prefix', type=str, default=BACKBONE_PRETRAINED_DST_PREFIX)
    parser.add_argument('--discriminator-pretrained', type=str, default=DISCRIMINATOR_PRETRAINED_PATH)
    parser.add_argument('--discriminator-pretrained-src-prefix', type=str, default=DISCRIMINATOR_PRETRAINED_SRC_PREFIX)
    parser.add_argument('--discriminator-pretrained-dst-prefix', type=str, default=DISCRIMINATOR_PRETRAINED_DST_PREFIX)
    args = parser.parse_args()

    config = get_config_file(args.config)
    kwargs = {'work_dir': args.work_dir, 'timestamp': args.timestamp, 'clear_work_dir': False}

    path = args.extra_conf_path
    if path and osp.isfile(path):
        extra_config = Config.from_file(path)
        kwargs.update(extra_config.to_dict())

    jsonstr = args.extra_conf_str
    if jsonstr:
        try:
            kwargs.update(json.loads(jsonstr))
        except Exception as e:
            print('\033[1m\x1b[31m!!!JSON Parsing Error: ' + str(e) + '\x1b[0m')
            raise e

    ann_file = U.parse_path(args.ann_file, is_file=True)
    img_dir = U.parse_path(args.img_dir, is_dir=True)
    val_ann_file = U.parse_path(args.val_ann_file, is_file=True)
    val_img_dir = U.parse_path(args.val_img_dir, is_dir=True)
    transfer_val_ann_file = U.parse_path(args.transfer_val_ann_file, is_file=True)
    transfer_val_img_dir = U.parse_path(args.transfer_val_img_dir, is_dir=True)
    kwargs['data'] = {
        'train.data_source': {
            'ann_file': ann_file,
            'img_dir': img_dir
        },
        'train_with_recon.data_source': {
            'ann_file': ann_file,
            'img_dir': img_dir
        },
        'val.data_source': {
            'ann_file': val_ann_file,
            'img_dir': val_img_dir
        },
        'val_transfer.data_source': {
            'ann_file': transfer_val_ann_file,
            'img_dir': transfer_val_img_dir
        }
    }

    if args.backbone_pretrained:
        assert osp.isfile(args.backbone_pretrained)
        kwargs['model.init_cfg'] = {
            'type': 'pretrained',
            'path': args.backbone_pretrained,
            'src_prefix': args.backbone_pretrained_src_prefix,
            'dst_prefix': args.backbone_pretrained_dst_prefix
        }

    if args.discriminator_pretrained:
        assert osp.isfile(args.discriminator_pretrained)
        kwargs['model.recon_loss.gan_recon_loss.init_cfg'] = {
            'type': 'pretrained',
            'path': args.discriminator_pretrained,
            'src_prefix': args.discriminator_pretrained_src_prefix,
            'dst_prefix': args.discriminator_pretrained_dst_prefix
        }

    train(config, runner='complex_train', **kwargs)


if __name__ == '__main__':
    main()
