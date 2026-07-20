"""Generate an adapter matching the OTHER server's Qwen3-32B-lora-fsdp-rank32:
targets transformer blocks AND embed_tokens AND lm_head. This is the adapter that
reproduces the ours-gmm regression there but not here (blocks-only adapter is fine).
r=32 to match fsdp-rank32.
"""
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/russia_mmo/models/Qwen3-32B"
OUT = "/home/russia_mmo/models/Qwen3-32B-lora-fsdplike"

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cpu", torch_dtype=torch.bfloat16)
cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
    modules_to_save=None,
    init_lora_weights=True, inference_mode=True,
)
m = get_peft_model(m, cfg)
# PEFT puts lm_head LoRA via target; add lm_head explicitly if not matched
for n, p in m.named_parameters():
    if "lora_" in n:
        p.data = torch.randn_like(p) * 0.01
m.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("SAVED", OUT)
