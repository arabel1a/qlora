set -euo pipefail

export CONTAINER_NAME=${CONTAINER_NAME:-vllm_misha}
export HOST=${HOST:-bz-ascend}
export CODE_DIR=${CODE_DIR:-/home/misha/qlora}

function create_container {
  IMAGE=${IMAGE:-"quay.nju.edu.cn/ascend/vllm-ascend:v0.23.0rc1-openeuler"}
  CMD="""
  docker run -itd --name "$CONTAINER_NAME" \
    --shm-size 50g \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /data/models:/data/models \
    -v /data/lab-shared:/data/lab-shared \
    --user root \
    --entrypoint /bin/bash \
    --ipc shareable \
    -v /home/misha:/home/misha \
    --rm \
    $IMAGE
  """
  echo $CMD $HOST
  ssh $HOST $CMD
}

function clean_copy {
  recreate=${recreate:-true}
  ssh $HOST "rm -rf $CODE_DIR/*" 
  if $recreate; then
    ssh $HOST "docker stop $CONTAINER_NAME" || echo "Creating new container"
    # ssh bz-ascend-relay "docker rm $CONTAINER_NAME"
    create_container
  fi
  rsync -azv --exclude="tmp/" --exclude="logs" --exclude=".*" ./* $HOST:$CODE_DIR
}
