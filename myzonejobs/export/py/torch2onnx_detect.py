import argparse

from myzonecv.apis import torch2onnx, get_config_file


#CONFIG_NAME = 'detect/detect_yolov5v6_s_coco_640x640'
CONFIG_NAME = 'detect/detect_yolov5v6_m_coco_640x640'

# CHECKPOINT_PATH = './workspace/model_zoo/detect_models/yolov5v6_s_ufc.pth'
CHECKPOINT_PATH = './workspace/model_zoo/detect_models/yolov5v6_m_ufc.pth'

INPUT_SIZE = [1, 3, 640, 640]

# OUTPUT_FILE = './workspace/model_zoo/detect_models/yolov5v6_s_ufc.onnx'
OUTPUT_FILE = './workspace/model_zoo/detect_models/yolov5v6_m_ufc.onnx'

APPLY_PRERUN = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=CONFIG_NAME)
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH)
    parser.add_argument('--input-size', type=int, nargs='+', default=INPUT_SIZE)
    parser.add_argument('--output-file', type=str, default=OUTPUT_FILE)
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument('--verify', type=int, default=1, help='n times to verify onnx output against pytorch output')
    args = parser.parse_args()

    assert args.opset_version == 11, 'Only supports opset 11 now'

    config = get_config_file(args.config)

    torch2onnx(config,
               checkpoint=args.checkpoint,
               input_size=args.input_size,
               output_file=args.output_file,
               opset_version=args.opset_version,
               show=args.show,
               verify=args.verify,
               apply_prerun=APPLY_PRERUN)


if __name__ == '__main__':
    main()
