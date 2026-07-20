#!/usr/bin/env bash
# A/B PREFILL repro: stock sgmv-always vs ours gmm+bgmv, 16 parallel, gen=10.
# Goal: reproduce the ~2x TTFT (prefill) regression at 16/128 and capture WHERE it
# goes (preemption / recompute / eager / capture) since the kernels (microbench)
# say gmm is FASTER. Only the punica .py is swapped between runs (same .so), so
# the ONLY difference is sgmv-always vs gmm-prefill+bgmv-decode.
set -u
VA=/vllm-workspace/vllm-ascend
Q=/home/russia_mmo/misha/qlora
PUNICA=$VA/vllm_ascend/lora/punica_npu.py
PORT=${PORT:-1606}; URL=http://0.0.0.0:${PORT}/v1/chat/completions
ADAPTER=${ADAPTER:-/home/russia_mmo/models/Qwen3-32B-lora}
MODEL=/home/russia_mmo/models/Qwen3-32B
export VLLM_USE_V1=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True VLLM_RPC_TIMEOUT=100000
export HCCL_IF_BASE_PORT=48000 OMP_NUM_THREADS=100
export ASCEND_RT_VISIBLE_DEVICES=${DEVICES:-0,1,2,3} NPU_VISIBLE_DEVICES=${DEVICES:-0,1,2,3}
mkdir -p $Q/logs/${ABDIR:-ab_prefill}
cd $Q

# back up ours so we can restore it after the stock pass
cp $PUNICA /tmp/punica_ours.py

boot() {  # $1 = tag, server log path implied
  local TAG=$1
  pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; sleep 8
  echo "[*] boot $TAG $(date)"
  vllm serve $MODEL \
    --dtype bfloat16 --data-parallel-size 1 --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 --trust-remote-code --served-model-name qwen3 \
    --block-size 128 --host 0.0.0.0 --port $PORT \
    --max-num-seqs 16 --max-model-len 16384 --max-num-batched-tokens 32768 \
    --no-enable-prefix-caching --enable-lora \
    --lora-modules lora-adapter=$ADAPTER --max_lora_rank 32 \
    > $Q/logs/${ABDIR:-ab_prefill}/server_${TAG}.log 2>&1 &
  for i in $(seq 1 90); do
    curl -sf http://0.0.0.0:${PORT}/health >/dev/null 2>&1 && { echo "[health ok @ ${i}0s] $TAG"; return 0; }
    sleep 10
  done
  echo "[!] BOOT FAILED $TAG"; tail -20 $Q/logs/${ABDIR:-ab_prefill}/server_${TAG}.log; return 1
}

run_eval() {  # $1 = tag
  local TAG=$1
  echo "======== EVAL $TAG gen=10 parallel=16 $(date) ========"
  python run_evalscope.py \
    --number ${NUMBER:-64} --warmup-num ${WARMUP:-8} --parallel ${PARALLEL:-16} \
    --model lora-adapter --api openai --url "$URL" \
    --dataset random --seed 42 \
    --tokenizer-path $MODEL \
    --min-prompt-length ${MINP:-2000} --max-prompt-length ${MAXP:-7000} \
    --min-tokens ${GEN:-10} --max-tokens ${GEN:-10} --prefix-length 0 \
    --extra-args '{"ignore_eos": true}' \
    --outputs-dir $Q/logs/${ABDIR:-ab_prefill}/out_${TAG} \
    --rate -1 2>&1 | tee $Q/logs/${ABDIR:-ab_prefill}/eval_${TAG}.log
}

signals() {  # $1 = tag -- pull systemic-cause signals from server log
  local L=$Q/logs/${ABDIR:-ab_prefill}/server_$1.log
  echo "---- SIGNALS $1 ----"
  grep -iE "Maximum concurrency|GPU KV cache size|KV cache" $L | tail -2
  echo "preempt/recompute count: $(grep -icE 'preempt|recompute|recomputed' $L)"
  echo "eager/fallback count:    $(grep -icE 'eager|fallback|will not be captured|graph capture.*skip' $L)"
  grep -iE "Capturing CUDA graphs.*100%|capturing.*100" $L | tail -1
  echo "compile/recapture count: $(grep -icE 'Dynamo|recompil|guard fail|Recapturing' $L)"
}

# ===== PASS A: OURS (gmm prefill + bgmv decode) =====
cp /tmp/punica_ours.py $PUNICA
echo "[A] deployed OURS (gmm). use_gmm hits: $(grep -c use_gmm $PUNICA)"
boot ours && { run_eval ours; signals ours; }

# ===== PASS B: STOCK (sgmv always) =====
cd $VA && git checkout -- vllm_ascend/lora/punica_npu.py && cd $Q
echo "[B] reverted to STOCK (sgmv). use_gmm hits: $(grep -c use_gmm $PUNICA)"
boot stock && { run_eval stock; signals stock; }

# restore ours
cp /tmp/punica_ours.py $PUNICA
pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null
echo "===== AB PREFILL DONE $(date) (ours restored) ====="
