#!/usr/bin/env bash
# applies all patches; pass --no-rebuild to deploy python only (skip operator compile)

set -euo pipefail
VA=${VA:-/vllm-workspace/vllm-ascend}
SRC="$(cd "$(dirname "$0")" && pwd)"
export MAX_JOBS=128
export COMPILE_CUSTOM_KERNELS=1
export SOC_VERSION=ascend910b1

REBUILD=1
for arg in "$@"; do
  case "$arg" in
    --no-rebuild) REBUILD=0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [ "$REBUILD" -eq 1 ]; then
  echo "[1/3] copy patched csrc into $VA/csrc"
  cp "$SRC/torch_binding_7d45286c9.cpp"      "$VA/csrc/torch_binding.cpp"
  cp "$SRC/torch_binding_meta_7d45286c9.cpp" "$VA/csrc/torch_binding_meta.cpp"

  echo "[2/3] clean-rebuild vllm_ascend_C"
  cd "$VA"
  rm -rf build
  python setup.py build_ext --inplace
  python -c "import torch, torch_npu, vllm_ascend.vllm_ascend_C; print('op present:', hasattr(torch.ops._C_ascend, 'add_lora_shrink'))"
else
  echo "[1-2/3] --no-rebuild: skipping operator compilation"
fi

echo "[3/3] deploy patches"
cp "$SRC/dsv4_7d45286c9.py"      "$VA/vllm_ascend/models/deepseek_v4.py"
cp "$SRC/punica_7d45286c9.py"    "$VA/vllm_ascend/lora/punica_npu.py"
cp "$SRC/fused_moe_7d45286c9.py" "$VA/vllm_ascend/lora/fused_moe.py"
cp "$SRC/quant_moe_7d45286c9.py" "$VA/vllm_ascend/lora/quant_moe.py"
cp "$SRC/moe_mlp_7d45286c9.py"   "$VA/vllm_ascend/ops/fused_moe/moe_mlp.py"

echo "DONE. Restart the vLLM server to load the new op + punica."
