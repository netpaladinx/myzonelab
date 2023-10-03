# BASH error handling:
#   exit on command failure
set -e
#   keep track of the last executed command
trap 'LAST_COMMAND=$CURRENT_COMMAND; CURRENT_COMMAND=$BASH_COMMAND' DEBUG
#   on error: print the failed command
trap 'ERROR_CODE=$?; FAILED_COMMAND=$LAST_COMMAND; tput setaf 1; echo "ERROR: command \"$FAILED_COMMAND\" failed with exit code $ERROR_CODE"; put sgr0;' ERR INT TERM

TRAIN_ID=0
SERVER_ID=0 # 0 or 1 or 2
SERVER_NUM=1 # 3

CONFIG_NAME="reid/reid_resnet50_256x192_c16_strthr_recon_agu"

for i in {1..100}; do

    CONF_STR=$(cat <<- END
{"data.train.data_loader.num_workers": 4,
 "data.train.data_params.n_samples_per_id": 8,
 "data.train.data_params.min_ids_per_batch": 8,
 "data.train.data_params.batch_size": 64,
 "data.train_with_recon.data_loader.num_workers": 4,
 "data.train_with_recon.data_params.n_samples_per_id": 8,
 "data.train_with_recon.data_params.min_ids_per_batch": 8,
 "data.train_with_recon.data_params.batch_size": 64,
 "complex_train.stations.0.threads.0.save.max_keep_ckpts": 0}
END
)
    n=$(($TRAIN_ID % $SERVER_NUM))
    if test $n -ne $SERVER_ID; then
        TRAIN_ID=$(($TRAIN_ID + 1))
        continue
    fi
    
    echo -e "\e[1;31mTRAIN_ID=$TRAIN_ID\e[0m"
    ./myzonejobs/train/sh/train_reid.sh $CONFIG_NAME "$CONF_STR" "$i"
    TRAIN_ID=$(($TRAIN_ID + 1))

done

