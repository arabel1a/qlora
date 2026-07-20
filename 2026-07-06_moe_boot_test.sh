#!/usr/bin/env bash
# E2E boot test for the MoE-LoRA backport: Qwen3-30B-A3B + random MoE LoRA adapter.
# Verifies: adapter loads onto FusedMoE (AscendFusedMoEWithLoRA), aclgraph capture
# passes (static-shape bgmv path), and a generation request returns with the adapter.
set -u
Q=/home/russia_mmo/misha/qlora
MODEL=/home/russia_mmo/models/Qwen3-30B-A3B-Thinking-2507
ADAPTER=/home/russia_mmo/models/Qwen3-30B-A3B-lora-moe
LOG=$Q/logs/moe_boot; mkdir -p $LOG
PORT=1606; URL=http://0.0.0.0:${PORT}/v1/chat/completions
export VLLM_USE_V1=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True VLLM_RPC_TIMEOUT=100000
export HCCL_IF_BASE_PORT=48000 OMP_NUM_THREADS=100
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NPU_VISIBLE_DEVICES=0,1,2,3
export VLLM_ASCEND_ENABLE_FUSED_MC2=0   # required by the MoE-LoRA v1 guard

pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; sleep 8
echo "[*] boot Qwen3-30B-A3B TP4 + MoE LoRA  $(date)"
vllm serve $MODEL \
  --dtype bfloat16 --data-parallel-size 1 --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 --trust-remote-code --served-model-name qwen3moe \
  --block-size 128 --host 0.0.0.0 --port $PORT \
  --max-num-seqs 16 --max-model-len 8192 --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching --no-enable-expert-parallel \
  --enable-lora --lora-modules moe-lora=$ADAPTER --max_lora_rank 16 \
  > $LOG/server.log 2>&1 &

for i in $(seq 1 120); do
  grep -q "Application startup complete" $LOG/server.log 2>/dev/null && { echo "RESULT: SERVER_UP"; break; }
  grep -qiE "MoE LoRA|Ascend MoE LoRA v1|aclnnUnique2|too many values to unpack" $LOG/server.log 2>/dev/null && { echo "RESULT: MOE_LORA_ERROR"; break; }
  grep -qiE "EngineCore.*failed|Engine core init.*failed|raise |Error:" $LOG/server.log 2>/dev/null && { echo "checking..."; }
  sleep 15
done

echo "===== capture / wrap signals ====="
grep -E "Capturing CUDA graphs.*100%" $LOG/server.log | tail -1
echo "capture ok: $(grep -c "Application startup complete" $LOG/server.log)"
echo "bgmv/unique errors: $(grep -icE "aclnnUnique2|first dimension of x" $LOG/server.log)"
grep -iE "AscendFusedMoEWithLoRA|MoE LoRA|Ascend MoE LoRA" $LOG/server.log | tail -3

if grep -q "Application startup complete" $LOG/server.log; then
  echo "===== generation test ====="
  curl -sf -X POST "$URL" -H "Content-Type: application/json" -d '{
    "model": "moe-lora",
    "messages": [{"role":"user","content":"In one sentence, what is a mixture-of-experts model?"}],
    "max_tokens": 40, "temperature": 0
  }' 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('GEN_OK:', d['choices'][0]['message']['content'][:200])" 2>&1 | tail -3
fi
echo "===== MOE BOOT TEST DONE $(date) ====="
