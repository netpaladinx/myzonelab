import os.path as osp
import json
import argparse

from myzonecv.apis import train, get_config_file, Config, utils as U


#CONFIG_NAME = 'detect/detect_yolov5v6_s_coco_640x640'
CONFIG_NAME = 'detect/detect_yolov5v6_m_coco_640x640'

CONF_STR = """
{"data.train.data_loader.batch_size": 10,
 "data.train.data_loader.num_workers": 4,
 "data.val.data_loader.batch_size": 10,
 "data.val.data_loader.num_workers": 4,
 "validator.val_at_start": false,
 "summarizer.interval": 10}
"""

WORK_DIR = './workspace/experiments/train_' + CONFIG_NAME

ANN_FILE = './workspace/data_zoo/trainval/MyZoneUFC-Detect/myzoneufc-detect_train1k-val100_10fights_v1.0/train/annotations.json'
IMG_DIR = './workspace/data_zoo/trainval/MyZoneUFC-Detect/myzoneufc-detect_train1k-val100_10fights_v1.0/train/images'
VAL_ANN_FILE = './workspace/data_zoo/trainval/MyZoneUFC-Detect/myzoneufc-detect_train1k-val100_10fights_v1.0/val/annotations.json'
VAL_IMG_DIR = './workspace/data_zoo/trainval/MyZoneUFC-Detect/myzoneufc-detect_train1k-val100_10fights_v1.0/val/images'

PRETRAINED_PATH = './workspace/model_zoo/detect_models/yolov5v6_m.pth'
# PRETRAINED_PATH = './workspace/model_zoo/detect_models/yolov5v6_s.pth'


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
    parser.add_argument('--split-train-val', action='store_true')
    parser.add_argument('--split-ratio', type=float, default=9)
    parser.add_argument('--pretrained', type=str, default=PRETRAINED_PATH)
    parser.add_argument('--pretrained-src-prefix', type=str, default=None)
    parser.add_argument('--pretrained-dst-prefix', type=str, default=None)
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
            raise e

    ann_file = U.parse_path(args.ann_file, is_file=True)
    img_dir = U.parse_path(args.img_dir, is_dir=True)
    if args.split_train_val:
        assert args.split_ratio > 0
        data_kwargs = {'data_source': {'ann_file': ann_file, 'img_dir': img_dir, 'drop_prob': 1. / args.split_ratio}}
    else:
        assert args.val_ann_file is not None and args.val_img_dir is not None
        val_ann_file = U.parse_path(args.val_ann_file, is_file=True)
        val_img_dir = U.parse_path(args.val_img_dir, is_dir=True)
        data_kwargs = {
            'train.data_source': {'ann_file': ann_file, 'img_dir': img_dir},
            'val.data_source': {'ann_file': val_ann_file, 'img_dir': val_img_dir}
        }
    kwargs['data'] = data_kwargs

    if args.pretrained:
        assert osp.isfile(args.pretrained)
        kwargs['model.init_cfg'] = {
            'type': 'pretrained',
            'path': args.pretrained,
            'src_prefix': args.pretrained_src_prefix,
            'dst_prefix': args.pretrained_dst_prefix
        }

    kwargs['use_diagnosis'] = True
    train(config, **kwargs)


if __name__ == '__main__':
    main()
