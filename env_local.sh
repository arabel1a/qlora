set -euo pipefail

export CONTAINER_NAME=${CONTAINER_NAME:-vllm_misha}

function create_container {
  IMAGE=${IMAGE:-"quay.nju.edu.cn/ascend/vllm-ascend:v0.23.0rc1-openeuler"}
  CMD="""
  docker run --name "$CONTAINER_NAME" \
    --privileged \
    --network host \
    --user 1004:1005 \
    --ipc shareable \
    --security-opt label=disable \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /home/:/home/
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -it -d --rm
    $IMAGE bash
  """
  ssh bz-ascend $CMD
}

function clean_copy {
  recreate=${recreate:-true}
  ssh bz-ascend "sudo rm -rf /home/misha/qlora/*" 
  if $recreate; then
    ssh bz-ascend "docker stop $CONTAINER_NAME" || echo "Creating new container"
    # ssh bz-ascend-relay "docker rm $CONTAINER_NAME"
    create_container
  fi
  rsync -azv --exclude=".*" ./* bz-ascend:/home/misha/qlora/
}
