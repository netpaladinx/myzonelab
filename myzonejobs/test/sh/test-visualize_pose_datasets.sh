#!/bin/bash

CONFIG_NAME="pose_hrnet_w32_coco_256x192_v2"
WORK_DIR="./workspace/tasks/test-visualize_pose_datasets"
TIMESTAMP=$(date +%Y%m%dT%H%M%S)
MODEL_TAG="experimental_20220605T060104"
CHECKPOINT="./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-${MODEL_TAG}.pth"

for DATA_DIR in ./workspace/data_zoo/trainval/*_FightFlow_RetrainingData_Error_0-350_Revised
do
    echo $DATA_DIR
    DATA_NAME=$(basename $DATA_DIR)
    DATA_TAG=${DATA_NAME//FightFlow_RetrainingData_Error_/}

    VISUALIZE_NAME="visualize_model-${MODEL_TAG}_data-${DATA_TAG}"
    IMG_DIR="${DATA_DIR}/images"
    ANN_FILE="${DATA_DIR}/annotations.json"
    echo $IMG_DIR
    echo $ANN_FILE
    echo $VISUALIZE_NAME

    myzone-test --config=$CONFIG_NAME \
    --work-dir=$WORK_DIR \
    --timestamp=$TIMESTAMP \
    --visualize-name=$VISUALIZE_NAME \
    --ann-file=$ANN_FILE \
    --img-dir=$IMG_DIR \
    --checkpoint=$CHECKPOINT \
    --batch-size=8 \
    --num-workers=2
done