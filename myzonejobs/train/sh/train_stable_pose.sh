CONFIG_NAME="pose/pose_hrnet_stable_w32_coco_256x192"

# CONF_STR=$(cat <<- END
# {"data.train.data_loader.batch_size": 4,
#  "data.train.data_loader.num_workers": 0,
#  "data.val.data_loader.batch_size": 4,
#  "data.val.data_loader.num_workers": 0,
#  "optim.max_epochs": 50,
#  "optim.lr_scheduler.step": [30,40],
#  "validator.val_at_start": true,
#  "summarizer.interval": 10}
# END
# )

# CONF_STR=$(cat <<- END
# {"data.train.data_loader.batch_size": 32,
#  "data.train.data_loader.num_workers": 4,
#  "data.train.data_transforms.1.flip_prob": 0.5,
#  "data.val.data_loader.batch_size": 32,
#  "data.val.data_loader.num_workers": 4,
#  "optim.max_epochs": 50,
#  "optim.lr_scheduler.step": [30,40],
#  "validator.val_at_start": true,
#  "summarizer.interval": 10}
# END
# )

# no flipping
# CONF_STR=$(cat <<- END
# {"data.train.data_loader.batch_size": 32,
#  "data.train.data_loader.num_workers": 4,
#  "data.train.data_transforms.0.flip_prob": 0.0,
#  "data.val.data_loader.batch_size": 32,
#  "data.val.data_loader.num_workers": 4,
#  "optim.max_epochs": 50,
#  "optim.lr_scheduler.step": [30,40],
#  "validator.val_at_start": true,
#  "summarizer.interval": 10}
# END
# )

######################
# Best configuration
######################
# CONF_STR=$(cat <<- END
# {"data.train.data_loader.batch_size": 32,
#  "data.train.data_loader.num_workers": 4,
#  "data.train.data_transforms.0.flip_prob": 0.0,
#  "data.val.data_loader.batch_size": 32,
#  "data.val.data_loader.num_workers": 4,
#  "model.stable_loss.loss_weight": 0.1,
#  "model.stable_loss.heatmap_gen_kwargs.visibility_thr": 0.8,
#  "optim.max_epochs": 50,
#  "optim.lr_scheduler.step": [30,40],
#  "validator.val_at_start": true,
#  "summarizer.interval": 10}
# END
# )

# for debug
CONF_STR=$(cat <<- END
{"data.train.data_loader.batch_size": 4,
 "data.train.data_loader.num_workers": 0,
 "data.train.data_transforms.0.flip_prob": 0.0,
 "data.val.data_loader.batch_size": 4,
 "data.val.data_loader.num_workers": 0,
 "model.stable_loss.loss_weight": 0.1,
 "model.stable_loss.heatmap_gen_kwargs.visibility_thr": 0.8,
 "optim.max_epochs": 50,
 "optim.lr_scheduler.step": [30,40],
 "validator.val_at_start": true,
 "summarizer.interval": 10}
END
)

WORK_DIR='./workspace/experiments/docker__'$CONFIG_NAME
TIMESTAMP=$(date +%Y%m%dT%H%M%S)

# ANN_FILE='./workspace/data_zoo/trainval/person_keypoints_debug3/annotations.json'
# IMG_DIR='./workspace/data_zoo/trainval/person_keypoints_debug3/images'
# PRETRAINED_PATH='./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-c78dce93_20200708.pth'

# ANN_FILE='./workspace/data_zoo/trainval/20220426_FightFlow_RetrainingData_Error_0-350/annotations.json'
# IMG_DIR='./workspace/data_zoo/trainval/20220426_FightFlow_RetrainingData_Error_0-350/images'

TRAIN_DATA_DIR="./workspace/data_zoo/trainval/MyZoneUFC-Pose"
IGNORE_FILE=$TRAIN_DATA_DIR"/.ignore"
ANN_FILE=""
IMG_DIR=""

for dataset_dir in $TRAIN_DATA_DIR/*; do
    if ! test -d $dataset_dir; then
        continue
    fi
    if [[ "$dataset_dir" == *","* ]]; then
        echo -e "\e[1;31mPath $dataset_dir cannot contain ','\e[0m"
        exit 1
    fi

    dataset_name=$(basename $dataset_dir)

    if ( test -f $IGNORE_FILE ) && ( grep -qE "^"$dataset_name"$" $IGNORE_FILE ); then
        echo "Ignore dataset $dataset_name"
    else
        if ! test -f $dataset_dir/annotations.json; then
            echo -e "\e[1;31mInvalid dataset $dataset_name, error: missing the annotations.json file\e[0m"
            exit 1
        fi
        
        if ! test -d $dataset_dir/images; then
            echo -e "\e[1;31mInvalid dataset $dataset_name, error: missing the images directory\e[0m"
            exit 1
        fi

        if ! test -z "$ANN_FILE"; then
            ANN_FILE=$ANN_FILE","
            IMG_DIR=$IMG_DIR","
        fi
        ANN_FILE=$ANN_FILE$dataset_dir/annotations.json
        IMG_DIR=$IMG_DIR$dataset_dir/images

        echo "Add dataset $dataset_name"
    fi
done

if test -z "$ANN_FILE"; then
    echo -e "\e[1;31mANN_FILE cannot be empty\e[0m"
    exit 1
fi
if test -z "$IMG_DIR"; then
    echo -e "\e[1;31mIMG_DIR cannot be empty\e[0m"
    exit 1
fi

echo -e "ANN_FILE=\e[1;32m$ANN_FILE\e[0m"
echo -e "IMG_DIR=\e[1;32m$IMG_DIR\e[0m"

PRETRAINED_PATH="./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-c78dce93_20200708.pth"
#PRETRAINED_PATH="./workspace/model_zoo/pose_models/hrnet_w48_coco_384x288-314c8528_20200708.pth"

echo -e "PRETRAINED_PATH=\e[1;32m$PRETRAINED_PATH\e[0m"

myzone-train --config=$CONFIG_NAME \
    --work-dir=$WORK_DIR \
    --timestamp=$TIMESTAMP \
    --ann-file=$ANN_FILE \
    --img-dir=$IMG_DIR \
    --split-train-val \
    --split-ratio 49 \
    --pretrained=$PRETRAINED_PATH \
    --extra-conf-str="$CONF_STR"
