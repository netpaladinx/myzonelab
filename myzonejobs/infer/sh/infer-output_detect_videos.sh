#!/bin/bash

VIDEO_DIR='/media/xiaoran/data/UFCCapturedDataset'

for VIDEO_PATH in $VIDEO_DIR/*; do
    if ! test -f $VIDEO_PATH; then
        continue
    fi

    WORK_DIR='./workspace/tasks/infer_detect_videos'
    TIMESTAMP=$(date +%Y%m%dT%H%M%S)
    DETECT_CONFIG='detect_yolov5v6_x_coco_640x640'
    DETECT_CHECKPOINT='./workspace/model_zoo/detect_models/yolov5v6_x_ufc.pth'
    VIDEO_FILE=$(basename $VIDEO_PATH)
    VIDEO_NAME=${VIDEO_FILE//.mp4/}
    CONF_DIR=$WORK_DIR"/confidence"
    SRC_CONF_FILE=$CONF_DIR"/0,0_confidence.txt"
    DST_CONF_FILE="${CONF_DIR}/${VIDEO_NAME}_confidence.txt"

    if test -f $DST_CONF_FILE; then
        continue
    fi

    echo "Start to process $VIDEO_PATH"

    myzone-infer --video-path=$VIDEO_PATH \
        --work-dir=$WORK_DIR \
        --timestamp=$TIMESTAMP \
        --detect-config=$DETECT_CONFIG \
        --detect-checkpoint=$DETECT_CHECKPOINT  \
        --eval-confidence

    mv $SRC_CONF_FILE $DST_CONF_FILE
done
