rsync -azv ./ bz-ascend:/home/russia_mmo/misha/qlora/
ssh bz-ascend "docker cp /home/russia_mmo/misha/qlora/all.py va18_misha:/vllm-workspace/vllm-ascend/vllm_ascend/lora/punica_npu.py"
# ssh bz-ascend "docker cp /home/russia_mmo/misha/qlora/test_e2e.py va18_misha:/home/russia_mmo/misha/qlora/test_e2e.py"
# ssh bz-ascend "docker cp /home/russia_mmo/misha/qlora/test_kernels.py va18_misha:/home/russia_mmo/misha/qlora/test_kernels.py"
# echo "--- Running e2e correctness test ---"
# ssh bz-ascend "docker exec va18_misha python /home/russia_mmo/misha/qlora/test_e2e.py --tokens 4 128 2048 --num-loras 2 4 --ranks 16 32 --num-requests 4 16"
