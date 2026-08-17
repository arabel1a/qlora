#!/usr/bin/env bash
# applies all patches

set -euo pipefail
VA=${VA:-/vllm-workspace/vllm-ascend}
SRC="$(cd "$(dirname "$0")" && pwd)"
PUNICA=$
export MAX_JOBS=128
export COMPILE_CUSTOM_KERNELS=1 
export SOC_VERSION=ascend910b1

echo "[1/3] copy patched csrc into $VA/csrc"
cp "$SRC/torch_binding.cpp"      "$VA/csrc/torch_binding.cpp"
cp "$SRC/torch_binding_meta.cpp" "$VA/csrc/torch_binding_meta.cpp"

echo "[2/3] clean-rebuild vllm_ascend_C (backup .so first)"
cd "$VA"
rm -rf build
python setup.py build_ext --inplace

python -c "import torch, torch_npu, vllm_ascend.vllm_ascend_C; print('op present:', hasattr(torch.ops._C_ascend, 'add_lora_shrink'))"

echo "[3/3] deploy punica_npu.py"
cp "$SRC/punica_npu.py" "$VA/vllm_ascend/lora/punica_npu.py"

echo "DONE. Restart the vLLM server to load the new op + punica."
