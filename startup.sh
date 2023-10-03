trap "nvidia-smi && /bin/bash" INT

pip install -e .

# ./myzonejobs/train/sh/pool_train_reid.sh || nvidia-smi && /bin/bash
./myzonejobs/train/sh/train_detect.sh || nvidia-smi && /bin/bash