"""
Minimal HuggingFace-compatible LoRA adapter initialization.
Requirements: pip install peft transformers torch
"""

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/russia_mmo/models/Qwen3-32B"

# ── Load base model ──────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cpu")

# ── LoRA config ───────────────────────────────────────────────────────────────
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    target_modules=["v_proj", "o_proj", "k_proj", "q_proj", "gate_proj", "down_proj", "up_proj"],
    fan_in_fan_out=False,
    init_lora_weights=True,
    use_dora=False,
    use_rslora=False,
    lora_bias=False,
    inference_mode=True,  
)

# ── Wrap model with LoRA ──────────────────────────────────────────────────────
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ── Save adapter (HF-compatible) ─────────────────────────────────────────────
ADAPTER_PATH = "/home/russia_mmo/models/Qwen3-32B-lora-r8"
import torch
for name, param in model.named_parameters():
    if "lora_" in name:
        param.data = torch.randn_like(param, device=param.device)
model.save_pretrained(ADAPTER_PATH)       # writes adapter_config.json + adapter_model.safetensors
tokenizer.save_pretrained(ADAPTER_PATH)

print(f"Adapter saved to {ADAPTER_PATH}")

# ── Reload later ─────────────────────────────────────────────────────────────
# from peft import PeftModel
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
# model = PeftModel.from_pretrained(model, ADAPTER_PATH)
