#!/usr/bin/env bash
# parallel=1 prefill-latency A/B: ours(gmm) vs stock(sgmv), embed+lm_head adapter.
# At parallel=1 TTFT == pure prefill latency (NO queueing), so the LoRA-kernel
# difference is exposed. 3 prompt buckets show whether the gap grows with tokens.
set -u
VA=/vllm-workspace/vllm-ascend
Q=/home/russia_mmo/misha/qlora
PUNICA=$VA/vllm_ascend/lora/punica_npu.py
PORT=1606; URL=http://0.0.0.0:${PORT}/v1/chat/completions
ADAPTER=/home/russia_mmo/models/Qwen3-32B-lora-fsdplike
MODEL=/home/russia_mmo/models/Qwen3-32B
export VLLM_USE_V1=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True VLLM_RPC_TIMEOUT=100000
export HCCL_IF_BASE_PORT=48000 OMP_NUM_THREADS=100
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NPU_VISIBLE_DEVICES=0,1,2,3
D=$Q/logs/p1_scaling; mkdir -p $D; cd $Q
cp $PUNICA /tmp/punica_ours.py

boot() {
  pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; sleep 8
  echo "[*] boot $1 $(date)"
  vllm serve $MODEL --dtype bfloat16 --data-parallel-size 1 --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 --trust-remote-code --served-model-name qwen3 \
    --block-size 128 --host 0.0.0.0 --port $PORT \
    --max-num-seqs 16 --max-model-len 16384 --max-num-batched-tokens 32768 \
    --no-enable-prefix-caching --enable-lora \
    --lora-modules lora-adapter=$ADAPTER --max_lora_rank 32 > $D/server_$1.log 2>&1 &
  for i in $(seq 1 90); do
    curl -sf http://0.0.0.0:${PORT}/health >/dev/null 2>&1 && { echo "[health ok] $1"; return 0; }
    sleep 10
  done
  echo "BOOT FAIL $1"; return 1
}

eval_bucket() {  # $1 tag  $2 minp  $3 maxp
  echo "==== EVAL $1 prompt=$2-$3 parallel=1 gen=10 $(date) ===="
  python run_evalscope.py --number 10 --warmup-num 2 --parallel 1 \
    --model lora-adapter --api openai --url "$URL" --dataset random --seed 42 \
    --tokenizer-path $MODEL --min-prompt-length $2 --max-prompt-length $3 \
    --min-tokens 10 --max-tokens 10 --prefix-length 0 \
    --extra-args '{"ignore_eos": true}' --outputs-dir $D/out_$1_$2 --rate -1 \
    2>&1 | tee $D/eval_$1_$2.log | grep -E "TTFT \(ms\)" | head -1
}

run_cfg() {  # $1 tag
  boot $1 || return
  eval_bucket $1 2000 2500
  eval_bucket $1 4000 4500
  eval_bucket $1 6500 7000
}

cp /tmp/punica_ours.py $PUNICA
echo "[A] OURS (gmm) use_gmm=$(grep -c use_gmm $PUNICA)"; run_cfg ours

cd $VA && git checkout -- vllm_ascend/lora/punica_npu.py && cd $Q
echo "[B] STOCK (sgmv) use_gmm=$(grep -c use_gmm $PUNICA)"; run_cfg stock

cp /tmp/punica_ours.py $PUNICA
pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null
echo "===== P1 SCALING DONE $(date) (ours restored) ====="
