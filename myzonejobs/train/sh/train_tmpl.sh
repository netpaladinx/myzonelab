WORK_DIR=
mkdir -p $WORK_DIR

TRAIN_ANN_FILE=
TRAIN_IMG_DIR=
VAL_ANN_FILE=
VAL_IMG_DIR= 

PRETRAINED_MODEL_FILE=

myzone-train --config=pose_hrnet_w32_coco_256x192 \
    --work_dir=$WORK_DIR \
    --ann_file=$TRAIN_ANN_FILE \
    --img_dir=$TRAIN_IMG_DIR \
    --val_ann_file=$VAL_ANN_FILE \
    --val_img_dir=$VAL_IMG_DIR \
    --pretrained=$PRETRAINED_MODEL_FILE