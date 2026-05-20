# set -e
# vllm
: "${PORT:=1606}"
: "${DEVICES:=0,1,2,3}"
export HOST=0.0.0.0

# vllm magic
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=$DEVICES
export NPU_VISIBLE_DEVICES=$DEVICES
export VLLM_RPC_TIMEOUT=100000
export HCCL_IF_BASE_PORT=48000

# model
# export SERVED_MODEL_NAME="qwen3"
# export SERVED_MODEL_NAME_LORA="lora-adapter"
# export MAX_NUM_SEQ=64
#export MODEL_TAG="Qwen3-4B-Instruct-2507"
export MODEL_TAG="Qwen3-32B"
#export LORA_ADAPTER="/home/russia_mmo/models/Qwen3-4b-nsfw"
export LORA_ADAPTER="/home/russia_mmo/models/Qwen3-32B-lora"
export MODEL="/home/russia_mmo/models/${MODEL_TAG}"
export MAX_LORAS=20 # IMPORTANT! number of loras per batch can not exceed the number of AI cubes
export MAX_LORA_RANK=32

# benchmarks
: "${NUM_PROMPTS:=256}"
: "${RANDOM_INPUT_LEN:=2048}"
: "${RANDOM_OUTPUT_LEN:=512}"
# export DATASET="random"
# export DATASET=./custom_dataset_qwen3_2000.jsonl
: "${DATASET:=random}"
# export DATASET=/home/russia_mmo/vllm_ascend_hub/vllm_repos/scripts/generated/custom_dataset_qwen3_2000.jsonl
export START_TIMEOUT=300
export STOP_TIMEOUT=30


# vllm
export TENSOR_PARALLEL_SIZE=4
export DATA_PARALLEL_SIZE=1
export MAX_NUM_SEQ=32
export MAX_MODEL_LEN=32768
export MAX_NUM_BATCHED_TOKENS=32768
export MEMORY_UTILIZATION=0.95
export DTYPE="bfloat16"
export BLOCK_SIZE=128

COMMON_VLLM_ARGS=(
    "$MODEL"
    --dtype "$DTYPE"
    --data-parallel-size "$DATA_PARALLEL_SIZE"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --gpu-memory-utilization "$MEMORY_UTILIZATION"
    --block-size "$BLOCK_SIZE"
    --max-num-seqs "$MAX_NUM_SEQ"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --no-enable-prefix-caching
    --no-enable-chunked-prefill
    --trust-remote-code
)

LORA_ARGS=(
  --enable-lora
  --max-loras $MAX_LORAS
  --max-lora-rank $MAX_LORA_RANK
  --lora-modules lora-adapter=${LORA_ADAPTER}
)

throughput_bench() {
    vllm bench throughput \
	--backend vllm \
	--model "${COMMON_VLLM_ARGS[@]}" \
        --num-prompts "$NUM_PROMPTS" \
        --dataset-name "$DATASET" \
        --random-input-len "$RANDOM_INPUT_LEN" \
        --random-output-len "$RANDOM_OUTPUT_LEN" \
        --seed 0 \
	--disable-detokenize \
	--disable-frontend-multiprocessing \
        --enable-lora \
        --max-loras $MAX_LORAS \
        --max-lora-rank $MAX_LORA_RANK \
        --lora-path ${LORA_ADAPTER} \
	"$@"
}

run_lora_server() {
    # shitty pydantic does not digest spaces in json...
    SERVER_ARGS=(
    # --additional_config '{"ascend_compilation_config":{"enable_npugraph_ex":false}}'
    --compilation-config '{"max_cudagraph_capture_size":176}'
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"./logs/qlora_profile"}'
    --port $PORT
    --host=$HOST
    )

    vllm serve ${COMMON_VLLM_ARGS[@]} ${SERVER_ARGS[@]} ${LORA_ARGS[@]} $@
}

run_server() {
    # shitty pydantic does not digest spaces in json...
    SERVER_ARGS=(
    # --additional_config '{"torchair_graph_config":{"enable":false},"ascend_scheduler_config":{"enabled":false,"enable_chunked_prefill":false,"chunked_prefill_enabled":false}}'
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"./logs/qlora_profile"}'
    --port $PORT
    --host=$HOST
    )

    vllm serve ${COMMON_VLLM_ARGS[@]} ${SERVER_ARGS[@]} $@
}

send_request() {
    local model_name=$1
    local STRING=$2
    [ -n "$label" ] && echo "--> Testing: $label"
    
    time curl -s -X POST "http://localhost:${PORT}/v1/completions" -H "Content-Type: application/json" -d "{\"prompt\": \"${STRING}\",\"model\": \"${model_name}\",\"max_tokens\": 100,\"temperature\": 0.0,\"top_p\": 1.0,\"seed\": 42}"
}

run_evalscope() {
    local model_name=$1
    local label=$2
    echo "Running Eval: [$label] with model: $model_name"
    export DATASET=random.jsonl
    export URL=http://0.0.0.0:${PORT}/v1/chat/completions
    python run_evalscope.py \
        --number "$NUM_PROMPTS" \
        --parallel 32 \
        --model "$model_name" \
        --api openai \
        --url "$URL" \
        --dataset custom \
        --dataset-path "$DATASET" \
        --max-tokens 512 \
        --min-tokens 512 \
        --prefix-length 0 \
        --extra-args '{"ignore_eos": true}' \
        --outputs-dir "logs/${label}" \
        --rate -1
}

benchmark() {
	run_name=$1
	bash -c "run_lora_server" &
	sleep $START_TIMEOUT
	bash -c "run_evalscope $lora-adapter $run_name" | tee > $run_name
 	cleanup_vllm	
	sleep $STOP_TIMEOUT
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
