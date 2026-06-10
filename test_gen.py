"""E2E generation comparison: base vs LoRA, under a chosen LoRA kernel path.

The LoRA kernel (sgmv vs gmm) is selected at IMPORT time by the LORA_GMM env var
read inside punica_npu (off|threshold|force). eager vs compiled is --eager.

For each prompt this prints the greedy output token ids for the BASE model
(no lora) and the LoRA model, so a driver can diff:
  - base vs lora        -> is the lora doing anything at all?
  - run-to-run (sgmv vs gmm, eager vs compiled) -> does the kernel/path matter?

Run (one process per config):
  ASCEND_RT_VISIBLE_DEVICES=7 LORA_GMM=off   python test_gen.py --eager
  ASCEND_RT_VISIBLE_DEVICES=7 LORA_GMM=force python test_gen.py --eager
  ASCEND_RT_VISIBLE_DEVICES=7 LORA_GMM=off   python test_gen.py
  ASCEND_RT_VISIBLE_DEVICES=7 LORA_GMM=force python test_gen.py
"""

import argparse
import json
import os

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL = "/home/russia_mmo/models/Qwen3-4B-Instruct-2507"
LORA = "/home/russia_mmo/models/Qwen3-4b-nsfw"

PROMPTS = [
    "Write a short story about a dragon who loves gardening.",
    "Explain why the sky is blue in two sentences.",
    "Describe your perfect weekend.",
    "Give me three tips for staying focused while working.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--long", action="store_true",
                    help="use one long (>1024 tok) prompt so threshold-mode gmm fires in prefill")
    args = ap.parse_args()

    global PROMPTS
    if args.long:
        filler = ("The history of gardening spans many centuries and cultures. " * 200)
        PROMPTS = [filler + "\n\nNow, summarize the passage above in one sentence."]

    gmm_mode = os.environ.get("LORA_GMM", "off")
    mode = "compiled" if not args.eager else "eager"
    tag = f"gmm={gmm_mode}/{mode}"

    llm = LLM(
        model=MODEL,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=32,
        max_model_len=2048,
        enforce_eager=args.eager,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    msgs = [[{"role": "user", "content": p}] for p in PROMPTS]

    base = llm.chat(msgs, sp)
    lora_req = LoRARequest("nsfw", 1, LORA)
    lora = llm.chat(msgs, sp, lora_request=lora_req)

    results = []
    for i, p in enumerate(PROMPTS):
        b_ids = list(base[i].outputs[0].token_ids)
        l_ids = list(lora[i].outputs[0].token_ids)
        n = min(len(b_ids), len(l_ids))
        same = sum(1 for k in range(n) if b_ids[k] == l_ids[k])
        results.append({
            "prompt": p,
            "base_ids": b_ids,
            "lora_ids": l_ids,
            "base_text": base[i].outputs[0].text,
            "lora_text": lora[i].outputs[0].text,
            "base_lora_prefix_match": same,
            "base_lora_identical": b_ids == l_ids,
        })

    print("===RESULT_JSON_BEGIN===")
    print(json.dumps({"tag": tag, "gmm_mode": gmm_mode, "mode": mode,
                      "results": results}))
    print("===RESULT_JSON_END===")


if __name__ == "__main__":
    main()
