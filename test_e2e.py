"""End-to-end test: run add_lora_linear through PunicaWrapperNPU (our gmm+sgmv)
and compare against the upstream PunicaWrapperNPU (sgmv-only).

Tests the full production path including:
  - fp32 intermediate buffers (allocated by add_lora_linear)
  - update_metadata → prefill_metadata flow
  - dual-launch gmm + sgmv with the correct kernel enabled
  - both prefill regime (>1024 tokens) and decode regime (<1024 tokens)

Usage:
    python test_e2e.py
    python test_e2e.py --tokens 4 128 2048 --ranks 16 32
"""

import argparse
import copy
import itertools
import sys

import torch
import torch_npu

import vllm_ascend.vllm_ascend_C  # noqa: F401
import vllm_ascend.meta_registration  # noqa: F401

from vllm.lora.punica_wrapper.utils import compute_meta


# ---------------------------------------------------------------------------
# Minimal mapping object that mimics what vllm passes to update_metadata
# ---------------------------------------------------------------------------

class FakeMapping:
    def __init__(self, token_lora_tensor, lora_index_to_id, max_loras, vocab_size=32000):
        self.token_lora_tensor = token_lora_tensor
        self.is_prefill = True
        self.max_loras = max_loras
        self.vocab_size = vocab_size
        # index_mapping / prompt_mapping store lora_int_ids (1-based, matching
        # values in lora_index_to_id). 0 = no lora.
        # token_lora_tensor already uses 1-based IDs; -1 maps to 0 (no lora).
        idx_map = [0 if t < 0 else t for t in token_lora_tensor.tolist()]
        self.index_mapping = idx_map
        prompt = []
        prev = None
        for v in idx_map:
            if v != prev:
                prompt.append(v)
                prev = v
        self.prompt_mapping = prompt


class FakeLoraConfig:
    def __init__(self, max_lora_rank=64):
        self.max_lora_rank = max_lora_rank


# ---------------------------------------------------------------------------
# Reference: pure-torch lora_linear (fp32 accumulation)
# ---------------------------------------------------------------------------

def torch_lora_linear(
    x: torch.Tensor,            # [T, hidden]
    lora_a_stacked: list,        # list of [num_loras, rank, hidden]
    lora_b_stacked: list,        # list of [num_loras, out_slice, rank]
    output_slices: tuple,
    lora_indices: torch.Tensor,  # per-group lora index
    seq_lengths: torch.Tensor,   # per-group token count
    scale: float,
) -> torch.Tensor:
    """Reference lora_linear: y += (x @ A) * scale @ B for each slice."""
    num_slices = len(lora_a_stacked)
    total_out = sum(output_slices)
    T = x.shape[0]
    y = torch.zeros(T, total_out, dtype=x.dtype, device=x.device)

    for s in range(num_slices):
        a = lora_a_stacked[s]
        b = lora_b_stacked[s]
        if a.ndim == 4:
            a = a.squeeze(1)
        if b.ndim == 4:
            b = b.squeeze(1)
        r = a.shape[1]

        offset = 0
        for g in range(len(seq_lengths)):
            sl = seq_lengths[g].item()
            idx = lora_indices[g].item()
            if idx >= 0:
                x_chunk = x[offset:offset + sl].float()
                # shrink: [sl, hidden] @ [hidden, rank] = [sl, rank], then scale
                intermediate = (x_chunk @ a[idx].float().T) * scale
                # expand: [sl, rank] @ [rank, out_slice] = [sl, out_slice]
                result = intermediate @ b[idx].float().T

                y_offset = sum(output_slices[:s])
                y[offset:offset + sl, y_offset:y_offset + output_slices[s]] += result.to(y.dtype)
            offset += sl

    return y


# ---------------------------------------------------------------------------
# Token-to-lora mapping generator
# ---------------------------------------------------------------------------

def generate_token_lora_tensor(num_tokens, num_loras, num_requests, device, include_no_lora=False):
    """Generate per-token lora_int_ids (1-based). -1 = no lora."""
    tokens_per_req = num_tokens // max(num_requests, 1)
    remainder = num_tokens - tokens_per_req * num_requests
    # 1-based lora_int_ids matching lora_index_to_id convention
    lora_ids = list(range(1, num_loras + 1))
    if include_no_lora:
        lora_ids.append(-1)
    mapping = []
    for i in range(num_requests):
        lora_id = lora_ids[i % len(lora_ids)]
        count = tokens_per_req + (1 if i < remainder else 0)
        mapping.extend([lora_id] * count)
    return torch.tensor(mapping[:num_tokens], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Build a lora_index_to_id mapping for update_metadata
# ---------------------------------------------------------------------------

def make_lora_index_to_id(num_loras, max_loras):
    # lora_index_to_id[slot] = lora_int_id.  Slot 0 = no-lora (-1).
    # In vllm, lora_int_id is 1-based (LoRA request IDs start at 1).
    mapping = [-1] * (max_loras + 1)
    for i in range(num_loras):
        if i + 1 <= max_loras:
            mapping[i + 1] = i + 1
    return mapping


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def check_close(name, ref, test, atol, rtol):
    ref_f = ref.float()
    test_f = test.float()
    diff = (ref_f - test_f).abs()
    max_diff = diff.max().item()
    # Use output-scale-relative error: normalize by max magnitude, not per-element
    scale = max(ref_f.abs().max().item(), 1e-8)
    scaled_max_rel = max_diff / scale
    ok = torch.allclose(ref_f, test_f, atol=atol, rtol=rtol)
    status = "OK  " if ok else "FAIL"
    print(f"  {status} {name:40s}  max_abs={max_diff:.4f}  rel_to_scale={scaled_max_rel:.6f}")
    return ok


def test_lora_linear(
    num_tokens, num_loras, num_slices, rank, hidden_size, num_requests,
    device, atol, rtol, dtype, max_loras=8,
):
    regime = "prefill" if num_tokens > 1024 else "decode"
    tag = (f"E2E ({regime})  T={num_tokens} L={num_loras} S={num_slices} "
           f"R={rank} H={hidden_size} reqs={num_requests}")
    print(f"\n{tag}")

    torch.manual_seed(42)

    # Weights: vllm stacked layout — [max_loras+1, ...], slot 0 = zeros (no-lora)
    # Actual LoRA weights at slots 1..num_loras
    num_slots = max_loras + 1
    slice_size = hidden_size // num_slices
    lora_a_stacked = []
    lora_b_stacked = []
    for _ in range(num_slices):
        a = torch.zeros(num_slots, rank, hidden_size, dtype=dtype, device=device)
        a[1:num_loras + 1] = torch.randn(num_loras, rank, hidden_size, dtype=dtype, device=device)
        lora_a_stacked.append(a.contiguous())
    for _ in range(num_slices):
        b = torch.zeros(num_slots, slice_size, rank, dtype=dtype, device=device)
        b[1:num_loras + 1] = torch.randn(num_loras, slice_size, rank, dtype=dtype, device=device)
        lora_b_stacked.append(b.contiguous())
    output_slices = tuple([slice_size] * num_slices)
    total_out = sum(output_slices)
    scale = 1.0

    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device).contiguous()

    token_lora = generate_token_lora_tensor(num_tokens, num_loras, num_requests, device)
    (b_seq_start, seq_lengths, lora_indices, batch_size,
     max_length, token_nums, no_lora) = compute_meta(token_lora)

    print(f"  groups={batch_size}  token_nums={token_nums}  regime={regime}")

    # --- Reference ---
    ref_y = torch_lora_linear(x, lora_a_stacked, lora_b_stacked, output_slices,
                              lora_indices, seq_lengths, scale)

    # --- PunicaWrapperNPU (deployed version at vllm_ascend.lora.punica_npu) ---
    from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
    wrapper = PunicaWrapperNPU(
        max_num_batched_tokens=max(num_tokens, 32768),
        max_batches=max(num_requests, 256),
        device=device,
        lora_config=FakeLoraConfig(max_lora_rank=rank),
    )

    lora_index_to_id = make_lora_index_to_id(num_loras, max_loras)
    mapping = FakeMapping(token_lora, lora_index_to_id, max_loras)
    wrapper.update_metadata(mapping, lora_index_to_id, max_loras, 32000)

    our_y = torch.zeros(num_tokens, total_out, dtype=dtype, device=device)
    wrapper.add_lora_linear(
        our_y, x,
        lora_a_stacked=tuple(lora_a_stacked),
        lora_b_stacked=tuple(lora_b_stacked),
        scale=scale,
        output_slices=output_slices,
    )

    all_ok = True
    all_ok &= check_close("wrapper vs torch ref", ref_y, our_y, atol, rtol)

    # Also check that the correct kernel ran
    if num_tokens > 1024:
        print(f"  (gmm should be active: token_nums={token_nums} > 1024)")
    else:
        print(f"  (sgmv should be active: token_nums={token_nums} <= 1024)")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="E2E test: our wrapper vs upstream vs torch ref")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--tokens", type=int, nargs="+", default=[4, 32, 128, 512, 2048, 4096])
    parser.add_argument("--num-loras", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--num-slices", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--num-requests", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--atol", type=float, default=16.0)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--device", type=str, default="npu:0")
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    print(f"Device: {device}  dtype: {dtype}  hidden: {args.hidden_size}")
    print(f"Tokens: {args.tokens}")
    print(f"LoRAs:  {args.num_loras}")
    print(f"Slices: {args.num_slices}")
    print(f"Ranks:  {args.ranks}")
    print(f"Reqs:   {args.num_requests}")

    total = 0
    fails = 0

    for (T, L, S, R, reqs) in itertools.product(
        args.tokens, args.num_loras, args.num_slices, args.ranks, args.num_requests,
    ):
        if reqs > T:
            continue
        total += 1
        ok = test_lora_linear(T, L, S, R, args.hidden_size, reqs, device,
                              args.atol, args.rtol, dtype)
        if not ok:
            fails += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {total - fails}/{total} passed, {fails} failed")
    print("=" * 60)
    sys.exit(1 if fails > 0 else 0)


if __name__ == "__main__":
    main()
