"""Generate a small random MoE LoRA adapter for Qwen3-30B-A3B-Thinking-2507 to
exercise the Ascend MoE-LoRA path (AscendFusedMoEWithLoRA). Targets the per-expert
MLP projections (gate_proj/up_proj/down_proj) AND attention (q/k/v/o). r=16 to
keep the adapter modest across 128 experts x 48 layers.
"""
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/russia_mmo/models/Qwen3-30B-A3B-Thinking-2507"
OUT = "/home/russia_mmo/models/Qwen3-30B-A3B-lora-moe"

print("loading base (CPU, ~min)...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cpu", torch_dtype=torch.bfloat16)
cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0,
    # gate_proj/up_proj/down_proj match the per-expert MLP linears (mlp.experts.{j}.*);
    # q/k/v/o match attention. The router (mlp.gate) is NOT matched.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    init_lora_weights=True, inference_mode=True,
)
print("injecting LoRA (per-expert, may take a few min)...", flush=True)
m = get_peft_model(m, cfg)
n = 0
for name, p in m.named_parameters():
    if "lora_" in name:
        p.data = torch.randn_like(p) * 0.01
        n += 1
print(f"initialized {n} lora tensors; saving...", flush=True)
m.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("SAVED", OUT, flush=True)
