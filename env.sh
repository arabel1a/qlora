# set -e
# vllm
: "${PORT:=1606}"
: "${DEVICES:=0,1,2,3}"
export HOST=127.0.0.1

# vllm magic
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$DEVICES
export NPU_VISIBLE_DEVICES=$DEVICES
export VLLM_RPC_TIMEOUT=100000
export HCCL_IF_BASE_PORT=48000
export VLLM_ENGINE_READY_TIMEOUT_S=1800

export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=512

sudo sysctl -w vm.swappiness=0 2>/dev/null || true
sudo sysctl -w kernel.numa_balancing=0 2>/dev/null || true
sudo sysctl -w kernel.sched_migration_cost_ns=50000 2>/dev/null || true

export VLLM_DISABLE_COMPILE_CACHE=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1   # also kills FX graph + autotune caches
rm -rf ~/.cache/vllm/torch_compile_cache
#model

export LORA_ADAPTER1=${LORA_ADAPTER1:-"/home/russia_mmo/models/Qwen3-32B-lora-r8"}
export LORA_ADAPTER2=${LORA_ADAPTER1:-"/home/russia_mmo/models/Qwen3-32B-lora-r8"}
export MODEL=${MODEL:-"/home/russia_mmo/models/Qwen3-4B-Instruct-2507"}
export MAX_LORAS=2 
export MAX_LORA_RANK=16

# benchmarks
: "${NUM_PROMPTS:=80}"
: "${PARALLEL:=8}"

WARMUP_PROMPTS_NUM=8
MIN_PROMPT_LENGTH=${MIN_PROMPT_LENGTH:-2048}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MIN_GEN_TOKENS=128
MAX_GEN_TOKENS=128

: "${DATASET:=random_2056.jsonl}"
export START_TIMEOUT=300
export STOP_TIMEOUT=30
: "${PROCESS_KILL_TIMEOUT_S:=10}"   # was unbound -> broke cleanup_vllm under set -u

# vllm
export TENSOR_PARALLEL_SIZE=${TP:-4}
export DATA_PARALLEL_SIZE=1
export MAX_NUM_SEQ=8
: "${MAX_MODEL_LEN:=32768}"
: "${MAX_NUM_BATCHED_TOKENS:=16384}"
export MEMORY_UTILIZATION=${MEMORY_UTILIZATION:-0.9}
export DTYPE="bfloat16"
export BLOCK_SIZE=128

COMMON_VLLM_ARGS=(
    "$MODEL"
    --data-parallel-size "$DATA_PARALLEL_SIZE"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --gpu-memory-utilization "$MEMORY_UTILIZATION"
    --block-size "$BLOCK_SIZE"
    --max-num-seqs "$MAX_NUM_SEQ"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --no-enable-prefix-caching
    --enable-chunked-prefill
    --trust-remote-code
    --async-scheduling
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"./logs/profile"}'
    --port $PORT
    --host $HOST
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":false,"multistream_dsa_preprocess":false}'
    --safetensors-load-strategy prefetch
)

QWEN_ARGS=(
    --dtype "$DTYPE"
)

DS_ARGS=(
    --quantization ascend
    # --enable-expert-parallel
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4
    --enable-auto-tool-choice
    --reasoning-parser deepseek_v4
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    # --enforce-eager
)

LORA_ARGS=(
  --enable-lora
  --max-loras $MAX_LORAS
  --max-lora-rank $MAX_LORA_RANK
  --lora-modules lora-adapter1=${LORA_ADAPTER1} lora-adapter2=${LORA_ADAPTER2}
)

run_qwen_lora() {
    vllm serve ${COMMON_VLLM_ARGS[@]} ${QWEN_ARGS[@]} ${LORA_ARGS[@]} $@
}

run_ds_lora() {
    vllm serve ${COMMON_VLLM_ARGS[@]} ${DS_ARGS[@]} ${LORA_ARGS[@]} $@
}

run_qwen() {
    vllm serve ${COMMON_VLLM_ARGS[@]} ${QWEN_ARGS[@]} $@
}

run_ds() {
    vllm serve ${COMMON_VLLM_ARGS[@]} ${DS_ARGS[@]} $@
}

start_profile() {
	curl -X POST http://$HOST:$PORT/start_profile
}
stop_profile() {
	curl -X POST http://$HOST:$PORT/stop_profile
}

send_request() {
    local model_name=$1
    local STRING=$2
    [ -n "$label" ] && echo "--> Testing: $label"
    
    time curl -s -X POST "http://localhost:${PORT}/v1/completions" -H "Content-Type: application/json" -d "{\"prompt\": \"${STRING}\",\"model\": \"${model_name}\",\"max_tokens\": 10,\"temperature\": 0.0,\"top_p\": 1.0,\"seed\": 42}"
}

profile(){
    local model_name=${1:-$MODEL}
    local label=$2
    rm -r logs/profile
    NUM_PROMPTS=8 PARALLEL=8 MIN_PROMPT_LENGTH=2048 MAX_PROMPT_LENGTH=2048 MIN_GEN_TOKENS=16 MAX_GEN_TOKENS=16 WARMUP=0 run_evalscope $1 $2_es
    rm -rf logs/$2_es
    start_profile
    NUM_PROMPTS=8 PARALLEL=8 MIN_PROMPT_LENGTH=2048 MAX_PROMPT_LENGTH=2048 MIN_GEN_TOKENS=16 MAX_GEN_TOKENS=16 WARMUP=0 run_evalscope $1 $2_es
    stop_profile
    if [ "${TENSOR_PARALLEL_SIZE:-1}" -gt 1 ]; then
    	python -c "from torch_npu.profiler.profiler import analyse; analyse('./logs/profile')"
    fi
}

run_evalscope() {
    local model_name=$1
    local label=$2
    warmup=${WARMUP:-$NUM_PROMPTS}
    
    echo "Running Eval: [$label] with model: $model_name"
    export URL=http://$HOST:${PORT}/v1/chat/completions
    python3 run_evalscope.py \
        --number "$NUM_PROMPTS" \
        --warmup-num $warmup \
        --parallel "$PARALLEL" \
        --model "$model_name" \
        --api openai \
        --dataset random \
        --seed 42 \
        --tokenizer-path $MODEL\
        --min-prompt-length $MIN_PROMPT_LENGTH \
        --max-prompt-length $MAX_PROMPT_LENGTH \
        --min-tokens $MIN_GEN_TOKENS \
        --max-tokens $MAX_GEN_TOKENS \
        --url "$URL" \
        --prefix-length 0 \
        --extra-args '{"ignore_eos": true}' \
        --outputs-dir "logs/${label}" \
        --rate -1 \
        --tokenizer-path $MODEL \
	#         --dataset custom \
	#         --dataset-path "$DATASET" \
	#         --max-tokens $RANDOM_OUTPUT_LEN \
	#        --min-tokens $RANDOM_OUTPUT_LEN \
}

run_mae() {
    local model_name=$1
    local label=$2

    local par=${MAE_PARALLEL:-8}
    local num=${MAE_NUM:-$((par * 10))}
    local in_min=${MAE_IN_MIN:-7946}    # 8192 - 3%
    local in_max=${MAE_IN_MAX:-8438}    # 8192 + 3%
    local out_min=${MAE_OUT_MIN:-120}
    local out_max=${MAE_OUT_MAX:-130}

    echo "Warmup: 8 requests via run_evalscope"
    NUM_PROMPTS=8 PARALLEL=8 WARMUP=8 \
        MIN_PROMPT_LENGTH=$in_min MAX_PROMPT_LENGTH=$in_max \
        MIN_GEN_TOKENS=$out_min MAX_GEN_TOKENS=$out_max \
        run_evalscope "$model_name" "${label}_warmup"

    echo "Running MAE perf test: [$label] with model: $model_name (par=$par, num=$num, in=[$in_min,$in_max], out=[$out_min,$out_max])"
    export URL=http://$HOST:${PORT}/v1/chat/completions
    python3 run_mae_perf_tests.py \
        --number "$num" \
        --parallel "$par" \
        --model "$model_name" \
        --api openai \
        --dataset random \
        --seed 42 \
        --tokenizer-path "$MODEL" \
        --min-prompt-length "$in_min" \
        --max-prompt-length "$in_max" \
        --min-tokens "$out_min" \
        --max-tokens "$out_max" \
        --url "$URL" \
        --prefix-length 0 \
        --extra-args '{"ignore_eos": true}' \
        --outputs-dir "logs/${label}" \
        --temperature 0.0 \
        --rate -1
}

cleanup_vllm() {
    PID=$(ps -ef | grep "vllm serve ${MODEL}" | grep -v grep | awk '{print $2}')
    if [ -n "$PID" ]; then
        kill "$PID"
        echo "Killed vLLM process: $PID"
        sleep "$PROCESS_KILL_TIMEOUT_S"
    else
        echo "No vLLM process found for ${MODEL}"
    fi
}

