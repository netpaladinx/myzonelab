CONFIG_NAME="detect/detect_yolov5v6_m_coco_640x640"

CONF_STR=$(cat <<- END
{ "data.train.data_loader.batch_size": 20,
  "data.train.data_loader.num_workers": 4,
  "data.val.data_loader.batch_size": 20,
  "data.val.data_loader.num_workers": 4,
  "validator.val_at_start": false,
  "summarizer.interval": 10 }
END
)

WORK_DIR="./workspace/experiments/docker_train_${CONFIG_NAME}"

TIMESTAMP=$(date +%Y%m%dT%H%M%S)

DATA_DIR="./workspace/data_zoo/trainval/MyZoneUFC-Detect/myzoneufc-detect_train1k-val100_10fights_v1.0"
TRAIN_ANN_FILE="$DATA_DIR/train/annotations.json"
TRAIN_IMG_DIR="$DATA_DIR/train/images"
VAL_ANN_FILE="$DATA_DIR/val/annotations.json"
VAL_IMG_DIR="$DATA_DIR/val/images"

PRETRAINED_PATH="./workspace/model_zoo/detect_models/yolov5v6_m.pth"

echo -e "CONFIG_NAME=\e[1;36m$CONFIG_NAME\e[0m"
echo -e "CONF_STR=\e[1;32m$CONF_STR\e[0m"
echo -e "WORK_DIR=\e[1;32m$WORK_DIR\e[0m"
echo -e "TRAIN_ANN_FILE=\e[1;32m$TRAIN_ANN_FILE\e[0m"
echo -e "TRAIN_IMG_DIR=\e[1;32m$TRAIN_IMG_DIR\e[0m"
echo -e "VAL_ANN_FILE=\e[1;32m$VAL_ANN_FILE\e[0m"
echo -e "VAL_IMG_DIR=\e[1;32m$VAL_IMG_DIR\e[0m"
echo -e "PRETRAINED_PATH=\e[1;32m$PRETRAINED_PATH\e[0m"

CONF_STR_DEBUG=$(cat <<- END
{ "data.train.data_loader.batch_size": 10,
  "data.train.data_loader.num_workers": 4,
  "data.val.data_loader.batch_size": 10,
  "data.val.data_loader.num_workers": 4 }
END
)

myzone-train --config=$CONFIG_NAME \
    --runner="train" \
    --work-dir=$WORK_DIR \
    --timestamp=$TIMESTAMP \
    --ann-file=$TRAIN_ANN_FILE \
    --img-dir=$TRAIN_IMG_DIR \
    --val-ann-file=$VAL_ANN_FILE \
    --val-img-dir=$VAL_IMG_DIR \
    --pretrained=$PRETRAINED_PATH \
    --extra-conf-str="$CONF_STR" \
    --extra2-conf-str="$CONF_STR_DEBUG"
