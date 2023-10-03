#!/usr/bin/env python
import os.path as osp
import argparse
import json

from myzonecv.apis import test, get_config_file, Config, utils as U
from myzonecv.core.utils.path import mkdir


def main(CONFIG_NAME, CHECKPOINT_PATH, CONF_STR=None, use_tensorboard=True, show_features=False, eval_features=True):
    print(f'[test_reid] config_name: {CONFIG_NAME}, checkpoint_path: {CHECKPOINT_PATH}')
    try:
        TIMESTAMP = osp.splitext(osp.basename(CHECKPOINT_PATH))[0].split('-')[-1]
    except Exception:
        TIMESTAMP = ''

    WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
    OUTPUT_DIR = f'output_{TIMESTAMP}'
    TENSORBOARD_DIR = f'tensorboard_{TIMESTAMP}' if use_tensorboard else None

    ANN_FILE = './workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_transfer_kpts-seg1/val/annotations.json'
    IMG_DIR = './workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_transfer_kpts-seg1/val/images'

    CHECKPOINT_TYPE = 'checkpoint'

    BATCH_SIZE = 8
    NUM_WORKERS = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=CONFIG_NAME)
    parser.add_argument('--extra-conf-path', type=str, default=None)
    parser.add_argument('--extra-conf-str', type=str, default=CONF_STR)
    parser.add_argument('--work-dir', type=str, default=WORK_DIR)
    parser.add_argument('--timestamp', type=str, default=TIMESTAMP)
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--tensorboard-dir', type=str, default=TENSORBOARD_DIR)
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

    output_dir = args.output_dir
    tensorboard_dir = args.tensorboard_dir
    log_path = None
    if args.work_dir:
        mkdir(args.work_dir, exist_ok=True)
        if output_dir:
            output_dir = osp.join(args.work_dir, osp.basename(output_dir))
        if tensorboard_dir:
            tensorboard_dir = osp.join(args.work_dir, osp.basename(tensorboard_dir))
        if args.timestamp:
            log_path = osp.join(args.work_dir, f'{args.timestamp}.log')

    ann_file = U.parse_path(args.ann_file, is_file=True)
    img_dir = U.parse_path(args.img_dir, is_dir=True)

    assert osp.isfile(args.checkpoint), f'Invalid checkpoint: {args.checkpoint}'
    kwargs.update({
        'custom_dirs': {
            'output_dir': output_dir,
            'tensorboard_dir': tensorboard_dir
        },
        'custom_vars': {
            'show_features': show_features,
            'eval_features': eval_features
        },
        'log_path': log_path,
        'data.test.data_source': {'ann_file': ann_file, 'img_dir': img_dir},
        'data.test.data_loader': {'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS},
        'model.init_cfg': {'type': CHECKPOINT_TYPE, 'path': args.checkpoint}
    })

    test(config, **kwargs)


if __name__ == '__main__':
    #main('reid_resnet50_256x192_base', './workspace/model_zoo/reid_models/reid_resnet50_256x192_base-20220812_051018.pth')
    #main('reid_resnet50_256x192_base-c26', './workspace/model_zoo/reid_models/reid_resnet50_256x192_base_c26-20220812_113846.pth')
    #main('reid_resnet50_256x192_base-c26', './workspace/model_zoo/reid_models/reid_resnet50_256x192_base_c26-20220812_140703.pth')
    #main('reid_resnet50_256x192_addlocal-c26', './workspace/model_zoo/reid_models/reid_resnet50_256x192_addlocal_c26-20220812_213723.pth')
    #main('reid_resnet50_256x192_addlocal-c26', './workspace/model_zoo/reid_models/reid_resnet50_256x192_addlocal_c26-20220813_053324.pth')
    #main('reid_resnet50_256x192_base-c26', './workspace/model_zoo/reid_models/reid_resnet50_256x192_base_c26-20220813_153505.pth')
    # main('reid_resnet50_256x192_addlocal-hier-c26',
    #      './workspace/model_zoo/reid_models/reid_resnet50_256x192_addlocal8_hier35_c26_50k-20220813_233557.pth',
    #      CONF_STR='{"model.head.local_extractor.n_parts": 8}', use_tensorboard=False, show_features=False, eval_features=False)
    # main('reid_resnet50_256x192_addlocal-hier-c26',
    #      './workspace/model_zoo/reid_models/reid_resnet50_256x192_addlocal6_hier20_c26_50k-20220814_094852.pth',
    #      CONF_STR='{"model.head.local_extractor.n_parts": 6}')
    # main('reid_resnet50_256x192_addlocal-hier-c26',
    #      './workspace/model_zoo/reid_models/reid_resnet50_256x192_addlocal4_hier9_c26_50k-20220814_154424.pth',
    #      CONF_STR='{"model.head.local_extractor.n_parts": 4}')
    #main('reid_resnet50_256x192_base-c26_v2', './workspace/model_zoo/reid_models/reid_resnet50_256x192_base_c26_linear_50k-20220814_232351.pth')
    # main('reid_resnet50_256x192_base-c26_v3', './workspace/model_zoo/reid_models/reid_resnet50_256x192_base_c26_arsinh_50k-20220814_212708.pth')
    #main('reid_resnet50_256x192_base-c26_v5', './workspace/model_zoo/reid_models/reid_resnet50_256x192_base_c26_mlL1_50k-20220815_103438.pth')
    #main('reid_resnet50_256x192_base-c26_v7', './workspace/model_zoo/reid_models/reid_resnet50_256x192_base_c26_linear_mlL1_clsL1_50k-20220816_044503.pth')
    # main('reid_resnet50_256x192_base-c26_v8', './workspace/model_zoo/reid_models/reid_resnet50_256x192_base_c26_linear_mlL1_50k-20220816_044105.pth')
    #main('reid_resnet50_256x192_base-c26_v9', './workspace/model_zoo/reid_models/reid_resnet50_256x192_base_c26_mlL1_clsL1_50k-20220816_043723.pth')
    main('reid/reid_resnet50_256x192_c16', './workspace/model_zoo/reid_models/reid_resnet50_256x192_c16-20220923T054037.pth')
