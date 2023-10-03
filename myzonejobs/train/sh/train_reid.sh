CONFIG_NAME=$1
echo -e "\e[1;36m============================================ $CONFIG_NAME ============================================\e[0m"

CONF_STR=$2
echo -e "CONF_STR=\e[1;32m$CONF_STR\e[0m"

SUFFIX_STR=$3
WORK_DIR="./workspace/experiments/docker__${CONFIG_NAME}_${SUFFIX_STR}"
echo -e "WORK_DIR=\e[1;32m$WORK_DIR\e[0m"

TIMESTAMP=$(date +%Y%m%dT%H%M%S)

DATA_DIR="./workspace/data_zoo/trainval/MyZoneUFC-ReID"
CLEAN_DATA_DIR="$DATA_DIR/MyZoneUFC-ReID_justin_clean_kpts-seg1"
NOISY_DATA_DIR="$DATA_DIR/MyZoneUFC-ReID_justin_noisy_kpts-seg1"

TRAIN_ANN_FILE="$CLEAN_DATA_DIR/train/annotations.json,$NOISY_DATA_DIR/train/annotations.json"
TRAIN_IMG_DIR="$CLEAN_DATA_DIR/train/images,$NOISY_DATA_DIR/train/images"
VAL_ANN_FILE="$CLEAN_DATA_DIR/val/annotations.json,$NOISY_DATA_DIR/val/annotations.json"
VAL_IMG_DIR="$CLEAN_DATA_DIR/val/images,$NOISY_DATA_DIR/val/images"

TRANSFER_DATA_DIR="$DATA_DIR/MyZoneUFC-ReID_justin_transfer_kpts-seg1"
TRANSFER_VAL_ANN_FILE="$TRANSFER_DATA_DIR/val/annotations.json"
TRANSFER_VAL_IMG_DIR="$TRANSFER_DATA_DIR/val/images"

echo -e "TRAIN_ANN_FILE=\e[1;32m$TRAIN_ANN_FILE\e[0m"
echo -e "TRAIN_IMG_DIR=\e[1;32m$TRAIN_IMG_DIR\e[0m"
echo -e "VAL_ANN_FILE=\e[1;32m$VAL_ANN_FILE\e[0m"
echo -e "VAL_IMG_DIR=\e[1;32m$VAL_IMG_DIR\e[0m"
echo -e "TRANSFER_VAL_ANN_FILE=\e[1;32m$TRANSFER_VAL_ANN_FILE\e[0m"
echo -e "TRANSFER_VAL_IMG_DIR=\e[1;32m$TRANSFER_VAL_IMG_DIR\e[0m"

PRETRAINED_PATH="./workspace/model_zoo/imagenet_pretrained/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth"
PRETRAINED_SRC_PREFIX="backbone"
PRETRAINED_DST_PREFIX="backbone"

echo -e "PRETRAINED_PATH=\e[1;32m$PRETRAINED_PATH\e[0m"

CONF_STR_DEBUG=$(cat <<- END
{ "complex_train.stations.0.threads.0.start_outer_round": 0,
  "complex_train.stations.0.threads.0.max_inner_steps": 4,
  "complex_train.stations.0.threads.0.max_steps": 4,
  "complex_train.stations.0.threads.0.summarize.interval": 2,
  "data.train_with_recon.data_params.n_samples_per_id": 3,
  "data.train_with_recon.data_params.min_ids_per_batch": 3,
  "data.train_with_recon.data_params.batch_size": 9}
END
)

myzone-train --config=$CONFIG_NAME \
    --runner="complex_train" \
    --work-dir=$WORK_DIR \
    --timestamp=$TIMESTAMP \
    --ann-file=$TRAIN_ANN_FILE \
    --img-dir=$TRAIN_IMG_DIR \
    --val-ann-file=$VAL_ANN_FILE \
    --val-img-dir=$VAL_IMG_DIR \
    --extra-data-key="train_with_recon" \
    --extra-ann-file=$TRAIN_ANN_FILE \
    --extra-img-dir=$TRAIN_IMG_DIR \
    --extra2-data-key="val_transfer" \
    --extra2-ann-file=$TRANSFER_VAL_ANN_FILE \
    --extra2-img-dir=$TRANSFER_VAL_IMG_DIR \
    --pretrained=$PRETRAINED_PATH \
    --pretrained-src-prefix=$PRETRAINED_SRC_PREFIX \
    --pretrained-dst-prefix=$PRETRAINED_DST_PREFIX \
    --extra-conf-str="$CONF_STR" # \
    # --extra2-conf-str="$CONF_STR_DEBUG"
