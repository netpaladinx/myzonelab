import os.path as osp
import json
import argparse

from myzonecv.apis import test, get_config_file, Config, utils as U
from myzonecv.core.utils.path import mkdir


# CONFIG_NAME = 'detect/detect_yolov5v6_s_coco_640x640'
CONFIG_NAME = 'detect/detect_yolov5v6_m_coco_640x640'

CONF_STR = """
{"data.test.data_loader.batch_size": 16,
 "data.test.data_loader.num_workers": 0}
"""

WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME

ANN_FILE = './workspace/data_zoo/trainval/fighter_bboxes_val2021-ufc202111/annotations.json'
IMG_DIR = './workspace/data_zoo/trainval/fighter_bboxes_val2021-ufc202111/images'

#CHECKPOINT_PATH = './workspace/model_zoo/detect_models/yolov5v6_s_ufc.pth'
#CHECKPOINT_PATH = './workspace/experiments/detect/detect_yolov5v6_s_coco_640x640/save_20221012_100547/best.pth'
#CHECKPOINT_PATH = './workspace/experiments/detect/detect_yolov5v6_s_coco_640x640/save_20221012_224520/best.pth'
#CHECKPOINT_PATH = './workspace/experiments/detect/detect_yolov5v6_s_coco_640x640/save_20221012_170120/best.pth'
#CHECKPOINT_PATH = './workspace/experiments/detect_yolov5v6_s_coco_640x640/save_backup/best.pth'
#CHECKPOINT_PATH = './workspace/experiments/detect_yolov5v6_s_coco_640x640/save/best.pth'
#CHECKPOINT_PATH = './workspace/experiments/train_detect/detect_yolov5v6_s_coco_640x640/save_20221122_162824/best.pth'
#CHECKPOINT_PATH = './workspace/experiments/train_detect/detect_yolov5v6_s_coco_640x640/save_20221122_175715/best.pth'

CHECKPOINT_PATH = './workspace/model_zoo/detect_models/yolov5v6_m_ufc.pth'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=CONFIG_NAME)
    parser.add_argument('--extra-conf-path', type=str, default=None)
    parser.add_argument('--extra-conf-str', type=str, default=CONF_STR)
    parser.add_argument('--work-dir', type=str, default=WORK_DIR)
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--ann-file', type=str, default=ANN_FILE)
    parser.add_argument('--img-dir', type=str, default=IMG_DIR)
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH)
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

    log_path = None
    if args.work_dir:
        mkdir(args.work_dir, exist_ok=True)
        if args.timestamp:
            log_path = osp.join(args.work_dir, f'{args.timestamp}.log')
        else:
            log_path = osp.join(args.work_dir, f'test.log')

    ann_file = U.parse_path(args.ann_file, is_file=True)
    img_dir = U.parse_path(args.img_dir, is_dir=True)

    assert osp.isfile(args.checkpoint), f'Invalid checkpoint: {args.checkpoint}'
    kwargs.update({
        'custom_dirs': {
        },
        'custom_vars': {
        },
        'log_path': log_path,
        'data.test.data_source': {'ann_file': ann_file, 'img_dir': img_dir},
        'model.init_cfg': {'type': 'checkpoint', 'path': args.checkpoint}
    })

    test(config, **kwargs)


if __name__ == '__main__':
    main()
