set -euo pipefail

function create_container {
  CONTAINER_NAME=${CONTAINER_NAME:-vllm_misha}
  IMAGE="quay.nju.edu.cn/ascend/vllm-ascend:v0.23.0rc1-openeuler"
  CMD="""
  docker run --name "$CONTAINER_NAME" \
    --privileged \
    --network host \
    --ipc shareable \
    --security-opt label=disable \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /home/russia_mmo:/home/russia_mmo \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -it -d --rm
    $IMAGE bash
  """
  ssh bz-ascend-relay $CMD
}

function clean_copy {
  recreate=${recreate:-true}
  ssh bz-ascend-relay "rm -rf /home/russia_mmo/misha/qlora"
  if $recreate; then
    ssh bz-ascend-relay "docker stop $CONTAINER_NAME"
    # ssh bz-ascend-relay "docker rm $CONTAINER_NAME"
    create_container
  fi
  rsync -azv ./ bz-ascend-relay:/home/russia_mmo/misha/qlora
}
