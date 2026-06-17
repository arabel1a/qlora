#!/usr/bin/env bash
# End-to-end build+deploy of the fused-LoRA C++ op (gmm prefill / bgmv decode).
#
# What it does (no manual steps):
#   1. push this dir to the host
#   2. drop the 2 patched csrc files into the vllm-ascend checkout in the container
#   3. clean-rebuild the vllm_ascend_C extension (backs up the working .so first)
#   4. deploy all.py as punica_npu.py
# Then restart the vLLM server to pick it up.
#
# Patched files live HERE in the work dir (replace-the-file model, like all.py):
#   torch_binding.cpp        -> csrc/torch_binding.cpp        (the add_lora_* ops + registration)
#   torch_binding_meta.cpp   -> csrc/torch_binding_meta.cpp   (their Meta impls)
#   all.py                   -> vllm_ascend/lora/punica_npu.py (calls torch.ops._C_ascend.add_lora_*)
#
# Revert: ./build_op.sh revert   (restores stock csrc via git + the backed-up .so, redeploys stock punica)
set -euo pipefail

HOST=${HOST:-bz-ascend-relay}          # bz-ascend is down; relay is the working route
CONT=${CONT:-va18_misha}
VA=/vllm-workspace/vllm-ascend         # vllm-ascend checkout inside the container
MNT=/home/russia_mmo/misha/qlora       # this dir, bind-mounted into the container
SO=$VA/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so
PUNICA=$VA/vllm_ascend/lora/punica_npu.py

if [ "${1:-build}" = "revert" ]; then
  echo "[revert] restoring stock csrc + .so + punica in $CONT"
  ssh "$HOST" "docker exec $CONT bash -lc '
    cd $VA && git checkout -- csrc/torch_binding.cpp csrc/torch_binding_meta.cpp
    [ -f /tmp/vllm_ascend_C.bak.so ] && cp /tmp/vllm_ascend_C.bak.so $SO
    git checkout -- vllm_ascend/lora/punica_npu.py 2>/dev/null || true
    echo reverted; rm -rf build'"
  echo "[revert] done. clean-rebuild stock if you reverted csrc: HOST=$HOST ./build_op.sh  (or restart server for .so-only revert)"
  exit 0
fi

echo "[1/4] push patched files -> $HOST:$MNT"
rsync -az torch_binding.cpp torch_binding_meta.cpp all.py "$HOST":"$MNT"/

echo "[2/4] drop patched csrc into $CONT:$VA/csrc"
ssh "$HOST" "docker cp $MNT/torch_binding.cpp      $CONT:$VA/csrc/torch_binding.cpp
             docker cp $MNT/torch_binding_meta.cpp $CONT:$VA/csrc/torch_binding_meta.cpp"

echo "[3/4] clean-rebuild vllm_ascend_C (backup .so first)"
# rm -rf build: the AscendC kernel-merge step does NOT incremental-build cleanly.
# COMPILE_CUSTOM_KERNELS=1 (default) is required or build_extensions() is a no-op.
# SOC_VERSION=ascend910b1 for Atlas A2 (910B4).
ssh "$HOST" "docker exec $CONT bash -lc '
  set -e
  cd $VA
  [ -f /tmp/vllm_ascend_C.bak.so ] || cp $SO /tmp/vllm_ascend_C.bak.so
  rm -rf build
  COMPILE_CUSTOM_KERNELS=1 SOC_VERSION=ascend910b1 MAX_JOBS=32 python setup.py build_ext --inplace
  python -c \"import torch, torch_npu, vllm_ascend.vllm_ascend_C; print(\\\"op present:\\\", hasattr(torch.ops._C_ascend, \\\"add_lora_shrink\\\"))\"
'"

echo "[4/4] deploy all.py -> punica_npu.py"
ssh "$HOST" "docker cp $MNT/all.py $CONT:$PUNICA"

echo "DONE. Restart the vLLM server to load the new op + punica."
