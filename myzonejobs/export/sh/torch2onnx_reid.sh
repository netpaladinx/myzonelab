#!/bin/bash

CHECKPOINT_PATH="./workspace/model_zoo/reid_models/reid_resnet50_256x192_c16-20220923T054037.pth"
ONNX_PATH="./workspace/model_zoo/reid_models/reid_resnet50_256x192_c16-20220923T054037.onnx"

. ../venv_pt1.12/bin/activate && \
    myzone-torch2onnx --config "reid/reid_resnet50_256x192_c16" \
        --checkpoint $CHECKPOINT_PATH \
        --input-size 2 3 256 192 \
        --output-file $ONNX_PATH \
        --verify 1