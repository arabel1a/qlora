#!/usr/bin/env bash
# Deploy the MoE-LoRA backport (vllm-ascend PR #10977 -> v0.18.0). PURE PYTHON:
# no .so rebuild -- the fused-MoE LoRA delta reuses the bgmv AscendC kernels the
# existing build already ships. Run INSIDE the container (offline), from the dir
# holding these files (e.g. the bind-mounted /home/russia_mmo/misha/qlora).
#
#   Deploy:  bash deploy_moe.sh
#   Revert:  bash deploy_moe.sh revert
#
# What it deploys (replace-file model, like all.py):
#   moe/lora/fused_moe.py        -> vllm_ascend/lora/fused_moe.py        (NEW)
#   moe/lora/utils.py            -> vllm_ascend/lora/utils.py            (registration)
#   moe/ops/moe_mlp.py           -> vllm_ascend/ops/fused_moe/moe_mlp.py (w13/w2 hooks)
#   moe/ops/moe_runtime_args.py  -> .../moe_runtime_args.py             (thread lora_context)
#   moe/ops/moe_stage_contracts.py -> .../moe_stage_contracts.py        (dataclass fields)
#   moe/ops/fused_moe.py         -> .../fused_moe.py                     (publish context @ apply)
#   all.py                       -> vllm_ascend/lora/punica_npu.py       (add_lora_fused_moe)
set -euo pipefail
VA=${VA:-/vllm-workspace/vllm-ascend}
SRC="$(cd "$(dirname "$0")" && pwd)"
NEWFILE=$VA/vllm_ascend/lora/fused_moe.py

# tracked file <- source file  (paths relative to $VA and $SRC)
declare -a MAP=(
  "vllm_ascend/lora/utils.py|moe/lora/utils.py"
  "vllm_ascend/ops/fused_moe/moe_mlp.py|moe/ops/moe_mlp.py"
  "vllm_ascend/ops/fused_moe/moe_runtime_args.py|moe/ops/moe_runtime_args.py"
  "vllm_ascend/ops/fused_moe/moe_stage_contracts.py|moe/ops/moe_stage_contracts.py"
  "vllm_ascend/ops/fused_moe/fused_moe.py|moe/ops/fused_moe.py"
  "vllm_ascend/lora/punica_npu.py|all.py"
)

if [ "${1:-deploy}" = "revert" ]; then
  echo "[revert] restoring stock MoE + punica files in $VA"
  cd "$VA"
  for pair in "${MAP[@]}"; do
    dst="${pair%%|*}"; git checkout -- "$dst" 2>/dev/null || true
  done
  rm -f "$NEWFILE"
  echo "[revert] done (removed new lora/fused_moe.py, git-restored the rest). Restart the server."
  exit 0
fi

echo "[1/2] install MoE-LoRA files into $VA"
cp "$SRC/moe/lora/fused_moe.py" "$NEWFILE"
echo "  + vllm_ascend/lora/fused_moe.py (NEW)"
for pair in "${MAP[@]}"; do
  dst="${pair%%|*}"; src="${pair##*|}"
  cp "$SRC/$src" "$VA/$dst"
  echo "  ~ $dst"
done

echo "[2/2] byte-compile check"
python3 -m py_compile \
  "$NEWFILE" \
  "$VA/vllm_ascend/lora/utils.py" \
  "$VA/vllm_ascend/lora/punica_npu.py" \
  "$VA/vllm_ascend/ops/fused_moe/moe_mlp.py" \
  "$VA/vllm_ascend/ops/fused_moe/moe_runtime_args.py" \
  "$VA/vllm_ascend/ops/fused_moe/moe_stage_contracts.py" \
  "$VA/vllm_ascend/ops/fused_moe/fused_moe.py"
echo "[ok] deployed. Restart the vLLM server to pick it up."
echo "     Test model: Qwen3-30B-A3B-Thinking-2507, TP=4, --enable-expert-parallel=false,"
echo "     VLLM_ASCEND_ENABLE_FUSED_MC2=0, --enable-lora --lora-modules <moe-lora>."
