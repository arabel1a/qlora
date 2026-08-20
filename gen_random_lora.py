import json
import os
import re
import sys
import glob
import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/russia_mmo/models/dsv4-w8a8-mtp"
model_name = MODEL_PATH.strip("/").split("/")[-1]
LORA_RANK = int(sys.argv[3]) if len(sys.argv) > 3 else 8
LORA_ALPHA = int(sys.argv[4]) if len(sys.argv) > 4 else 16
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else f"/home/russia_mmo/models/lora-{model_name}-r{LORA_RANK}"
W_TO_PROJ = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
TARGET_LEAVES = set(W_TO_PROJ) | set(W_TO_PROJ.values())

def leaf(module_path):
    return module_path.rsplit(".", 1)[-1]

def canonicalize(module_path):
    parts = module_path.split(".")
    if parts[-1] in W_TO_PROJ:
        parts[-1] = W_TO_PROJ[parts[-1]]
    return ".".join(parts)

discovered = {}  # module_path -> (out_features, in_features)
shards = sorted(glob.glob(os.path.join(MODEL_PATH, "*.safetensors")))
if not shards:
    sys.exit(f"No *.safetensors found under {MODEL_PATH}")

all_module_types = set()
for shard in shards:
    with safe_open(shard, framework="pt") as f:
        for key in f.keys():
            if not key.endswith(".weight"): continue
            module = key[: -len(".weight")]
            all_module_types.add(re.sub(r"\d+", "*", module))
            if leaf(module) not in TARGET_LEAVES:
                continue
            shape = f.get_slice(key).get_shape()  # header read only, no data load
            if len(shape) != 2:
                continue
            discovered[module] = (shape[0], shape[1])
print("All module types:", all_module_types)

if not discovered:
    sys.exit(f"No LoRA-target linear weights found under {MODEL_PATH}")

state_dict = {}
leaves_used = set()
for module, (out_dim, in_dim) in sorted(discovered.items()):
    canon = canonicalize(module)
    state_dict[f"{canon}.lora_A.weight"] = torch.ones(LORA_RANK, in_dim, dtype=torch.bfloat16) * 0.1337
    state_dict[f"{canon}.lora_B.weight"] = torch.ones(out_dim, LORA_RANK, dtype=torch.bfloat16) * 0.1337
    leaves_used.add(leaf(canon))

    os.makedirs(OUTPUT_PATH, exist_ok=True)
save_file(state_dict, os.path.join(OUTPUT_PATH, "adapter_model.safetensors"))

adapter_config = {
    "base_model_name_or_path": MODEL_PATH,
    "bias": "none",
    "fan_in_fan_out": False,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": 0.0,
    "modules_to_save": None,
    "peft_type": "LORA",
    "r": LORA_RANK,
    "target_modules": sorted(leaves_used),
    "task_type": "CAUSAL_LM",
    "use_rslora": False,
}

with open(os.path.join(OUTPUT_PATH, "adapter_config.json"), "w") as f:
    json.dump(adapter_config, f, indent=2)

n_experts = len({m for m in discovered if ".experts." in m})
print(f"LoRA saved to {OUTPUT_PATH}")
print(f"  {len(discovered)} target modules ({len(state_dict)} tensors), rank={LORA_RANK}, alpha={LORA_ALPHA}")
print(f"  leaf types: {sorted(leaves_used)}")
print(f"  expert projection tensors: {n_experts}")
