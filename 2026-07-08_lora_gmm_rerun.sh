#!/usr/bin/env bash
# Re-run +LoRA with the gmm kernel ACTUALLY ON (LORA_GMM=threshold). The prior run
# used the default LORA_GMM=off -> bgmv fallback (not gmm). Base is unchanged.
set -u
EXPERIMENT=2026_07_07_va18_perf_qwen3_32b_group_gemm
Q=/home/russia_mmo/misha/qlora
D=$Q/logs/$EXPERIMENT
MODEL=/home/russia_mmo/models/Qwen3-32B
ADAPTER=/home/russia_mmo/models/Qwen3-32B-lora-fsdplike
PORT=1995; URL=http://0.0.0.0:${PORT}/v1/chat/completions
export DEVICES=0,1,2,3
export OMP_PROC_BIND=false OMP_NUM_THREADS=100
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=$DEVICES NPU_VISIBLE_DEVICES=$DEVICES
export VLLM_RPC_TIMEOUT=100000 HCCL_IF_BASE_PORT=48000
export LORA_GMM=threshold          # <<< enable YOUR gmm-prefill kernel
cd $Q

pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; sleep 8
echo "[*] boot lora_GMM ($(date)) LORA_GMM=$LORA_GMM"
vllm serve $MODEL \
    --dtype bfloat16 --data-parallel-size 1 --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 --trust-remote-code --served-model-name qwen3 \
    --block-size 128 --host 0.0.0.0 --port $PORT \
    --max-num-seqs 16 --max-model-len 16384 --max-num-batched-tokens 32768 \
    --no-enable-prefix-caching \
    --enable-lora --lora-modules lora-adapter=$ADAPTER --max_lora_rank 32 \
    > $D/log_serve_lora_gmm.txt 2>&1 &
for i in $(seq 1 75); do curl -sf http://0.0.0.0:${PORT}/health >/dev/null 2>&1 && { echo "[health ok @ ${i}0s]"; break; }; sleep 10; done
echo "=== [LORA_GMM] mode line ==="; grep "LORA_GMM] mode=" $D/log_serve_lora_gmm.txt | head -1

echo "======== EVAL lora_gmm model=lora-adapter parallel=8,16 gen=2048 $(date) ========"
python run_evalscope.py \
    --number 64 128 --warmup-num 16 --parallel 8 16 \
    --model lora-adapter --api openai --url "$URL" \
    --dataset random --seed 42 --tokenizer-path $MODEL \
    --min-prompt-length 2000 --max-prompt-length 7000 \
    --min-tokens 2048 --max-tokens 2048 --prefix-length 0 \
    --extra-args '{"ignore_eos": true}' \
    --outputs-dir $D/out_lora_gmm --rate -1 2>&1 | tee $D/eval_lora_gmm.log \
    | grep -E "Total Throughput|Output Throughput|TTFT \(ms\)|TPOT \(ms\)" | head -10

echo "=== PROOF: gmm FIRED lines from server log ==="; grep "gmm prefill path FIRED" $D/log_serve_lora_gmm.txt | head -2
pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null
echo "===== LORA GMM RERUN DONE $(date) ====="
