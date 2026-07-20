#!/usr/bin/env bash
# Throughput A/B: Qwen3-30B-A3B base (NO lora) vs +MoE-LoRA. Two clean boots so the
# delta is the true cost of LoRA (base boots without --enable-lora at all).
# evalscope: parallel 8 & 16, mnbt=32768, random 2-7k prefill, 128 gen.
set -u
Q=/home/russia_mmo/misha/qlora
MODEL=/home/russia_mmo/models/Qwen3-30B-A3B-Thinking-2507
ADAPTER=/home/russia_mmo/models/Qwen3-30B-A3B-lora-moe
D=$Q/logs/moe_tput; mkdir -p $D
PORT=1606; URL=http://0.0.0.0:${PORT}/v1/chat/completions
export VLLM_USE_V1=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True VLLM_RPC_TIMEOUT=100000
export HCCL_IF_BASE_PORT=48000 OMP_NUM_THREADS=100
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NPU_VISIBLE_DEVICES=0,1,2,3
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
cd $Q

boot() {  # $1 tag  $2 = "lora" | "base"
  pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; sleep 8
  local EXTRA=""
  if [ "$2" = "lora" ]; then
    EXTRA="--enable-lora --lora-modules moe-lora=$ADAPTER --max_lora_rank 16"
  fi
  echo "[*] boot $1 ($2) $(date)"
  vllm serve $MODEL \
    --dtype bfloat16 --data-parallel-size 1 --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 --trust-remote-code --served-model-name qwen3moe \
    --block-size 128 --host 0.0.0.0 --port $PORT \
    --max-num-seqs 16 --max-model-len 8192 --max-num-batched-tokens 32768 \
    --no-enable-prefix-caching --no-enable-expert-parallel \
    $EXTRA > $D/server_$1.log 2>&1 &
  for i in $(seq 1 100); do
    curl -sf http://0.0.0.0:${PORT}/health >/dev/null 2>&1 && { echo "[health ok] $1"; return 0; }
    sleep 10
  done
  echo "[!] BOOT FAILED $1"; tail -15 $D/server_$1.log; return 1
}

run_eval() {  # $1 tag  $2 = model-name to request (qwen3moe | moe-lora)
  echo "======== EVAL $1 model=$2 parallel=8,16 gen=128 $(date) ========"
  python run_evalscope.py \
    --number 64 128 --warmup-num 8 --parallel 8 16 \
    --model "$2" --api openai --url "$URL" \
    --dataset random --seed 42 \
    --tokenizer-path $MODEL \
    --min-prompt-length 2000 --max-prompt-length 7000 \
    --min-tokens 128 --max-tokens 128 --prefix-length 0 \
    --extra-args '{"ignore_eos": true}' \
    --outputs-dir $D/out_$1 --rate -1 2>&1 | tee $D/eval_$1.log \
    | grep -E "TTFT \(ms\)|Output Throughput|Total Throughput|Avg Latency|Req Throughput" | head -12
}

# ---- A: BASE (no lora) ----
boot base base && run_eval base qwen3moe
# ---- B: +MoE-LoRA ----
boot lora lora && run_eval lora moe-lora

pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null
echo "===== MOE TPUT AB DONE $(date) ====="
