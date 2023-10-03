#!/bin/bash

CHECKPOINT_PATH="./workspace/model_zoo/detect_models/yolov5v6_s_ufc.pth"
ONNX_PATH="./workspace/model_zoo/detect_models/yolov5v6_s_ufc.onnx"

. ../venv/bin/activate && \
    myzone-torch2onnx --config "detect/detect_yolov5v6_s_coco_640x640" \
        --checkpoint $CHECKPOINT_PATH \
        --input-size 1 3 640 640 \
        --output-file $ONNX_PATH \
        --verify 1