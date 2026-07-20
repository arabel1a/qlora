#!/usr/bin/env bash
# Faithful reproduction of the user's eval_perf.sh + run_vllm_qlora_service.sh on
# va18_misha (bz-ascend), devices 0-3, with OUR gmm-kernel build deployed.
# Serve args + evalscope args are IDENTICAL to the user's scripts; only devices
# (4-7 -> 0-3) and the adapter (fsdp-rank32 absent -> Qwen3-32B-lora-fsdplike,
# same targets: blocks+embed+lm_head r=32) are adjusted. Runs BASE (no lora) then
# BF16+LoRA so the "performance drop" is quantified. w8a8 skipped (assets absent).
set -u
EXPERIMENT=2026_07_07_va18_perf_qwen3_32b_group_gemm
Q=/home/russia_mmo/misha/qlora
D=$Q/logs/$EXPERIMENT; mkdir -p $D
MODEL=/home/russia_mmo/models/Qwen3-32B
ADAPTER=/home/russia_mmo/models/Qwen3-32B-lora-fsdplike
PORT=1995; URL=http://0.0.0.0:${PORT}/v1/chat/completions
# --- env exactly as run_vllm_qlora_service.sh (devices adjusted) ---
export DEVICES=0,1,2,3
export OMP_PROC_BIND=false OMP_NUM_THREADS=100
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=$DEVICES NPU_VISIBLE_DEVICES=$DEVICES
export VLLM_RPC_TIMEOUT=100000 HCCL_IF_BASE_PORT=48000
cd $Q

boot() {   # $1 = tag  $2 = "lora"|"base"
  pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; sleep 8
  local LORA=""
  [ "$2" = "lora" ] && LORA="--enable-lora --lora-modules lora-adapter=$ADAPTER --max_lora_rank 32"
  echo "[*] boot $1 ($2) $(date)"
  vllm serve $MODEL \
      --dtype bfloat16 --data-parallel-size 1 --tensor-parallel-size 4 \
      --gpu-memory-utilization 0.9 --trust-remote-code --served-model-name qwen3 \
      --block-size 128 --host 0.0.0.0 --port $PORT \
      --max-num-seqs 16 --max-model-len 16384 --max-num-batched-tokens 32768 \
      --no-enable-prefix-caching $LORA > $D/log_serve_$1.txt 2>&1 &
  for i in $(seq 1 75); do
    curl -sf http://0.0.0.0:${PORT}/health >/dev/null 2>&1 && { echo "[health ok @ ${i}0s] $1"; return 0; }
    sleep 10
  done
  echo "[!] BOOT FAILED $1"; tail -15 $D/log_serve_$1.txt; return 1
}

run_eval() {  # $1 tag  $2 model-name (qwen3|lora-adapter)
  echo "======== EVAL $1 model=$2 parallel=8,16 number=64,128 gen=2048 $(date) ========"
  python run_evalscope.py \
      --number 64 128 --warmup-num 16 --parallel 8 16 \
      --model "$2" --api openai --url "$URL" \
      --dataset random --seed 42 \
      --tokenizer-path $MODEL \
      --min-prompt-length 2000 --max-prompt-length 7000 \
      --min-tokens 2048 --max-tokens 2048 --prefix-length 0 \
      --extra-args '{"ignore_eos": true}' \
      --outputs-dir $D/out_$1 --rate -1 2>&1 | tee $D/eval_$1.log \
      | grep -E "Total Throughput|Output Throughput|TTFT \(ms\)|TPOT \(ms\)|Avg Latency|Req Throughput" | head -14
}

echo "##### BASE (bf16, no lora) #####"
boot base base   && run_eval base qwen3
echo "##### BF16 + LoRA (gmm kernel) #####"
boot lora lora   && run_eval lora lora-adapter

pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null
echo "===== EVAL_PERF_VA18 DONE $(date) ====="
