source env.sh 
# 
# echo $1
# export filename=log_$1.txt
# [ -e $filename ] && { echo "log exist"; exit 1; }
# export nruns=10
# 
# # prefill
# export RANDOM_INPUT_LEN=1024
# export NUM_PROMPTS=40
# export RANDOM_OUTPUT_LEN=1 
# export MAX_MODEL_LEN=1025
# echo "Prefill: $RANDOM_INPUT_LEN $RANDOM_OUTPUT_LEN $NUM_PROMPTS" >> $filename
# for run in $(seq 0 $nruns); do
# 	throughput_bench | grep Throughput | tee >> $filename 
# done
# 
# # decode 
# export RANDOM_INPUT_LEN=1
# export NUM_PROMPTS=1000
# export RANDOM_OUTPUT_LEN=512
# export MAX_MODEL_LEN=513
# 
# echo "decode: $RANDOM_INPUT_LEN $RANDOM_OUTPUT_LEN $NUM_PROMPTS" >> $filename
# echo $(printenv) >> ${1}_env
# for run in $(seq 0 $nruns); do
#     throughput_bench --output-json ${1}_${run}.json | grep Throughput | tee >> $filename
# done
#

filename=res.txt
echo "nolora" >> $filename
throughput_bench --output-json nolora.json | grep Throughput | tee >> $filename
echo "stub" >> $filename
VLLM_LORA_SPLIT_PREFILL_DECODE=0 VLLM_LORA_USE_STUB_KERNEL=1 throughput_bench "${LORA_ARGS[@]}" --output-json stub.json | grep Throughput | tee >> $filename
