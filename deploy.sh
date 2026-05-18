rsync -azv ./ bz-ascend:/home/russia_mmo/misha/qlora/
ssh bz-ascend "docker cp /home/russia_mmo/misha/qlora/all.py va18_misha:/vllm-workspace/vllm-ascend/vllm_ascend/lora/punica_npu.py"
