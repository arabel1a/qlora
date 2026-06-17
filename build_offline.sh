#!/usr/bin/env bash
# OFFLINE build: run this INSIDE the vllm-ascend container. No ssh/scp/rsync.
#
# Usage:
#   1. Copy these 4 files together to the remote (anywhere reachable in the container,
#      e.g. the bind-mounted /home/russia_mmo/misha/qlora, or `docker cp` them in):
#        build_offline.sh  torch_binding.cpp  torch_binding_meta.cpp  all.py
#   2. Run inside the container, from the dir that holds them:
#        docker exec <container> bash /path/to/build_offline.sh
#      (or `docker exec -it <container> bash`, cd to the dir, then `bash build_offline.sh`)
#
#   Revert to stock:  bash build_offline.sh revert
#
# Env overrides: VA (vllm-ascend dir), SOC_VERSION (default ascend910b1, Atlas A2/910B4),
#                MAX_JOBS (default 32).
set -euo pipefail

VA=${VA:-/vllm-workspace/vllm-ascend}
SRC="$(cd "$(dirname "$0")" && pwd)"       # dir holding the patched files (next to this script)
SO=$VA/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so
PUNICA=$VA/vllm_ascend/lora/punica_npu.py

if [ "${1:-build}" = "revert" ]; then
  echo "[revert] restoring stock csrc + .so + punica in $VA"
  cd "$VA"
  git checkout -- csrc/torch_binding.cpp csrc/torch_binding_meta.cpp 2>/dev/null || true
  git checkout -- vllm_ascend/lora/punica_npu.py 2>/dev/null || true
  [ -f /tmp/vllm_ascend_C.bak.so ] && cp /tmp/vllm_ascend_C.bak.so "$SO"
  rm -rf build
  echo "[revert] done. Clean-rebuild stock (bash build_offline.sh) if you reverted csrc; restart server."
  exit 0
fi

echo "[1/3] install patched csrc into $VA/csrc"
cp "$SRC/torch_binding.cpp"      "$VA/csrc/torch_binding.cpp"
cp "$SRC/torch_binding_meta.cpp" "$VA/csrc/torch_binding_meta.cpp"

echo "[2/3] clean-rebuild vllm_ascend_C (backup .so first)"
# rm -rf build: the AscendC kernel-merge step does NOT incremental-build cleanly.
# COMPILE_CUSTOM_KERNELS=1 (default) is required or build_extensions() is a silent no-op.
cd "$VA"
[ -f /tmp/vllm_ascend_C.bak.so ] || cp "$SO" /tmp/vllm_ascend_C.bak.so
rm -rf build
COMPILE_CUSTOM_KERNELS=1 SOC_VERSION="${SOC_VERSION:-ascend910b1}" MAX_JOBS="${MAX_JOBS:-32}" \
  python setup.py build_ext --inplace
python -c "import torch, torch_npu, vllm_ascend.vllm_ascend_C; print('op present:', hasattr(torch.ops._C_ascend, 'add_lora_shrink'))"

echo "[3/3] deploy all.py -> punica_npu.py"
cp "$SRC/all.py" "$PUNICA"

echo "DONE. Restart the vLLM server to load the new op + punica."
