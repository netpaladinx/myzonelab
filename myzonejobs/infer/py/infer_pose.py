#!/usr/bin/env python
import os.path as osp
import argparse
import json

from myzonecv.apis import infer, get_config_file, Config, utils as U
from myzonecv.core.utils.path import mkdir


def main(DATA_LABEL,
         DATA_DIR,
         CONFIG_NAME,
         CHECKPOINT_PATH,
         EXTRA_CONF_STR=None,
         EXTRA_CONF_PATH=None):
    print(f'[infer_bboxseg] data_label: {DATA_LABEL}')
    print(f'[infer_bboxseg] data_dir: {DATA_DIR}')
    print(f'[infer_bboxseg] config_name: {CONFIG_NAME}')
    print(f'[infer_bboxseg] checkpoint_path: {CHECKPOINT_PATH}')

    try:
        TIMESTAMP = osp.splitext(osp.basename(CHECKPOINT_PATH))[0].split('-')[-1]
    except Exception:
        TIMESTAMP = ''

    WORK_DIR = './workspace/experiments/infer__' + CONFIG_NAME
    DUMP_DIR = f'dump_{DATA_LABEL}_{TIMESTAMP}'
    RES_ANN_FILE = f'result_annotations_{DATA_LABEL}_{TIMESTAMP}.json'

    DATA_DIR = DATA_DIR.rstrip('/')
    ANN_FILE = f'{DATA_DIR}/annotations.json'
    IMG_DIR = f'{DATA_DIR}/images'
    CHECKPOINT_TYPE = 'checkpoint'
    BATCH_SIZE = 8
    NUM_WORKERS = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=CONFIG_NAME)
    parser.add_argument('--extra-conf-path', type=str, default=EXTRA_CONF_PATH)
    parser.add_argument('--extra-conf-str', type=str, default=EXTRA_CONF_STR)
    parser.add_argument('--work-dir', type=str, default=WORK_DIR)
    parser.add_argument('--timestamp', type=str, default=TIMESTAMP)
    parser.add_argument('--dump-dir', type=str, default=DUMP_DIR)
    parser.add_argument('--res-ann-file', type=str, default=RES_ANN_FILE)
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
            pass

    dump_dir = args.dump_dir
    res_ann_file = args.res_ann_file
    if args.work_dir:
        mkdir(args.work_dir, exist_ok=True)
        if dump_dir:
            dump_dir = osp.join(args.work_dir, osp.basename(dump_dir))
        if res_ann_file:
            res_ann_file = osp.join(args.work_dir, osp.basename(res_ann_file))

    ann_file = U.parse_path(args.ann_file, is_file=True)
    img_dir = U.parse_path(args.img_dir, is_dir=True)

    assert osp.isfile(args.checkpoint), f'Invalid checkpoint: {args.checkpoint}'
    kwargs.update({
        'custom_dirs': {
            'dump_dir': dump_dir,
        },
        'custom_vars': {
            'res_ann_file': res_ann_file
        },
        'data.infer.data_source': {'ann_file': ann_file, 'img_dir': img_dir},
        'data.infer.data_loader': {'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS},
        'model.init_cfg': {'type': CHECKPOINT_TYPE, 'path': args.checkpoint}
    })

    infer(config, **kwargs)


if __name__ == '__main__':
    main(DATA_LABEL='MyZoneUFC-ReID_justin_clean_val',
         DATA_DIR='./workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_clean/val',
         CONFIG_NAME='pose_hrnet_w32_coco_256x192_v2',
         CHECKPOINT_PATH='./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-retrained_20220709T151140B.pth')
    main(DATA_LABEL='MyZoneUFC-ReID_justin_clean_train',
         DATA_DIR='./workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_clean/train',
         CONFIG_NAME='pose_hrnet_w32_coco_256x192_v2',
         CHECKPOINT_PATH='./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-retrained_20220709T151140B.pth')
    main(DATA_LABEL='MyZoneUFC-ReID_justin_noisy_val',
         DATA_DIR='./workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_noisy/val',
         CONFIG_NAME='pose_hrnet_w32_coco_256x192_v2',
         CHECKPOINT_PATH='./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-retrained_20220709T151140B.pth')
    main(DATA_LABEL='MyZoneUFC-ReID_justin_noisy_train',
         DATA_DIR='./workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_noisy/train',
         CONFIG_NAME='pose_hrnet_w32_coco_256x192_v2',
         CHECKPOINT_PATH='./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-retrained_20220709T151140B.pth')
    main(DATA_LABEL='MyZoneUFC-ReID_justin_transfer_val',
         DATA_DIR='./workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_transfer/val',
         CONFIG_NAME='pose_hrnet_w32_coco_256x192_v2',
         CHECKPOINT_PATH='./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-retrained_20220709T151140B.pth')
