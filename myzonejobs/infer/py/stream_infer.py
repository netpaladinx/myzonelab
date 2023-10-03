import os.path as osp
import argparse

from myzonecv.apis import get_config_file, unified_infer, utils as U

# VIDEO_PATH = './workspace/data_zoo/videos/20211103_Burlinson-vs-Lainesse_Camera1_Round1.mp4'
# VIDEO_PATH = './workspace/data_zoo/videos/2021-11-13_Casey-vs-Jojua_camera2_clip1.mp4'
VIDEO_PATH = './workspace/data_zoo/videos/img_test.mp4'
WORK_DIR = './workspace/experiments/stream_infer_on_test'

##### Setting detector #####
DETECT_CONFIG = 'detect/detect_yolov5v6_x_coco_640x640_stream'

# DETECT_INIT_TYPE = 'pretrained'
# DETECT_CHECKPOINT = './workspace/model_zoo/detect_models/yolov5v6_x.pth'

DETECT_INIT_TYPE = 'checkpoint'
DETECT_CHECKPOINT = './workspace/model_zoo/detect_models/yolov5v6_x_ufc.pth'

##### Setting pose estimator #####
POSE_CONFIG = 'pose/pose_hrnet_w32_coco_256x192_stream'

POSE_INIT_TYPE = 'pretrained'
POSE_CHECKPOINT = './workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-c78dce93_20200708.pth'

POSE_INIT_TYPE = 'checkpoint'
#POSE_CHECKPOINT = './workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-retrained_20220407T132620.pth'
#POSE_CHECKPOINT = './workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-retrained_20220404T143948.pth'
#POSE_CHECKPOINT = './workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-experimental_20220605T060104.pth'
#POSE_CHECKPOINT = './workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-retrained_20220707T115204.pth'
# POSE_CHECKPOINT = './workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-retrained_20220709T151140B.pth'

INFER_POSE = True
EVAL_SMOOTHNESS = False
DISPLAY = True
USE_CAMERA = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, default=VIDEO_PATH)
    parser.add_argument('--work-dir', type=str, default=WORK_DIR)
    parser.add_argument('--detect-config', type=str, default=DETECT_CONFIG)
    parser.add_argument('--detect-checkpoint', type=str, default=DETECT_CHECKPOINT)
    parser.add_argument('--pose-config', type=str, default=POSE_CONFIG)
    parser.add_argument('--pose-checkpoint', type=str, default=POSE_CHECKPOINT)
    parser.add_argument('--sink-id', type=int, default=0)
    parser.add_argument('--infer-pose', action='store_true', default=INFER_POSE)
    parser.add_argument('--eval-smoothness', action='store_true', default=EVAL_SMOOTHNESS)
    parser.add_argument('--display', action='store_true', default=DISPLAY)
    parser.add_argument('--use-camera', action='store_true', default=USE_CAMERA)
    args = parser.parse_args()

    if args.detect_config is not None:
        args.detect_config = get_config_file(args.detect_config)

    if not args.infer_pose:
        args.pose_config = None
    if args.pose_config is not None:
        args.pose_config = get_config_file(args.pose_config)

    if args.use_camera:
        source = 0
    else:
        source = U.parse_path(args.video_path, is_file=True)[0]

    assert osp.isfile(args.detect_checkpoint)
    kwargs = {
        'work_dir': args.work_dir,
        'display': args.display,
        'use_camera': args.use_camera,
        'sink_id': args.sink_id,
        'detect.data.infer.data_source': source,
        'detect.data.infer.data_eval.eval_smoothness': args.eval_smoothness,
        'detect.model.init_cfg': {'type': DETECT_INIT_TYPE, 'path': args.detect_checkpoint},
        'pose.model.init_cfg': {'type': POSE_INIT_TYPE, 'path': args.pose_checkpoint}
    }

    unified_infer(args.detect_config, args.pose_config, **kwargs)


if __name__ == '__main__':
    main()
