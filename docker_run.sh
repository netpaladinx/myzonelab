#DOCKER_REPO="xiaoran/myzonelab:latest"
DOCKER_REPO="xiaoran/myzonelab:22.07-py3"
LOCAL_DIR=$(pwd)
DOCKER_DIR="/workspace"

docker run --gpus all -it --shm-size=16g --rm --name myzonelab \
    -v $LOCAL_DIR:$DOCKER_DIR \
    --net=host \
    --add-host=host.docker.internal:host-gateway \
   $DOCKER_REPO