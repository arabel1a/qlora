"""Correctness tests for gmm lora kernels vs sgmv and torch reference.

Tests shrink (lora_a) and expand (lora_b) paths across different regimes:
  - token counts (small decode-like, large prefill-like)
  - number of LoRA adapters
  - number of slices (1 for q/k/v individual, 2 for qkv merged)
  - ranks

Runs standalone on NPU, no vllm server needed.

Usage:
    uv run test_kernels.py
    uv run test_kernels.py --hidden-size 2048 --ranks 8 16 32
"""

import argparse
import itertools
import sys

import torch
import torch_npu

import vllm_ascend.vllm_ascend_C  # noqa: F401  — registers _C_ascend ops (sgmv_shrink etc.)
import vllm_ascend.meta_registration  # noqa: F401

from vllm.lora.punica_wrapper.utils import compute_meta


# ---------------------------------------------------------------------------
# Reference: pure-torch shrink & expand (fp32 accumulation, known correct)
# ---------------------------------------------------------------------------

def torch_shrink(
    x: torch.Tensor,           # [T, K]
    w: torch.Tensor,           # [num_loras, (1,) N_rank, K]
    lora_indices: torch.Tensor, # per-group lora index
    seq_lengths: torch.Tensor,  # per-group token count
    scale: float,
) -> torch.Tensor:
    """Reference shrink: y[t] = x[t] @ w[lora_of_t].T * scale"""
    if w.ndim == 4:
        w = w.squeeze(1)
    T, K = x.shape
    N = w.shape[1]
    y = torch.zeros(T, N, dtype=torch.float32, device=x.device)
    offset = 0
    for g in range(len(seq_lengths)):
        sl = seq_lengths[g].item()
        idx = lora_indices[g].item()
        if idx >= 0:
            # w[idx] is [N, K], x_chunk is [sl, K], result is [sl, N]
            y[offset:offset + sl] = (x[offset:offset + sl].float() @ w[idx].float().T) * scale
        offset += sl
    return y


def torch_expand_slice(
    y: torch.Tensor,           # [T, out_dim] — modified in-place
    x: torch.Tensor,           # [T, N_rank]
    w: torch.Tensor,           # [num_loras, (1,) N_rank, out_slice]
    lora_indices: torch.Tensor,
    seq_lengths: torch.Tensor,
    y_offset: int,
    y_slice_size: int,
    add_inputs: bool,
) -> None:
    """Reference expand_slice: y[t, off:off+s] += x[t] @ w[lora_of_t]"""
    if w.ndim == 4:
        w = w.squeeze(1)
    offset = 0
    for g in range(len(seq_lengths)):
        sl = seq_lengths[g].item()
        idx = lora_indices[g].item()
        if idx >= 0:
            result = x[offset:offset + sl].float() @ w[idx].float().T
            target = y[offset:offset + sl, y_offset:y_offset + y_slice_size]
            if add_inputs:
                target.add_(result.to(target.dtype))
            else:
                target.copy_(result.to(target.dtype))
        offset += sl


# ---------------------------------------------------------------------------
# GMM kernels (extracted from all.py, no custom_op wrapper needed for test)
# ---------------------------------------------------------------------------

_TRANSPOSED_WEIGHT_CACHE: dict[int, torch.Tensor] = {}


def _get_transposed_weight(w: torch.Tensor) -> torch.Tensor:
    key = w.data_ptr()
    if (cached := _TRANSPOSED_WEIGHT_CACHE.get(key)) is not None:
        return cached
    if w.ndim == 4:
        w = w.squeeze(1)
    w_t = w.transpose(1, 2).contiguous()
    _TRANSPOSED_WEIGHT_CACHE[key] = w_t
    return w_t


def _gather_weights_for_gmm(w: torch.Tensor, lora_indices: torch.Tensor) -> torch.Tensor:
    w_t = _get_transposed_weight(w)
    safe_indices = lora_indices.clamp(min=0)
    gathered = w_t[safe_indices]
    inactive = (lora_indices < 0).unsqueeze(1).unsqueeze(2)
    gathered = gathered.masked_fill(inactive, 0.0)
    return gathered


def gmm_shrink(
    x: torch.Tensor,
    w: torch.Tensor,
    lora_indices: torch.Tensor,
    seq_lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """GMM-based shrink using npu_grouped_matmul."""
    gathered_w = _gather_weights_for_gmm(w, lora_indices)
    x_in = x if x.dtype == gathered_w.dtype else x.to(gathered_w.dtype)
    result = torch_npu.npu_grouped_matmul(
        x=[x_in], weight=[gathered_w],
        split_item=2, group_list_type=1, group_type=0,
        group_list=seq_lengths,
    )[0]
    if scale != 1.0:
        result = result * scale
    return result


def gmm_expand_slice(
    y: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    lora_indices: torch.Tensor,
    seq_lengths: torch.Tensor,
    y_offset: int,
    y_slice_size: int,
    add_inputs: bool,
) -> None:
    """GMM-based expand_slice using npu_grouped_matmul."""
    gathered_w = _gather_weights_for_gmm(w, lora_indices)
    x_in = x if x.dtype == gathered_w.dtype else x.to(gathered_w.dtype)
    result = torch_npu.npu_grouped_matmul(
        x=[x_in], weight=[gathered_w],
        split_item=2, group_list_type=1, group_type=0,
        group_list=seq_lengths,
    )[0]
    target = y[:, y_offset:y_offset + y_slice_size]
    if add_inputs:
        target.add_(result.to(target.dtype))
    else:
        target.copy_(result.to(target.dtype))


# ---------------------------------------------------------------------------
# SGMV wrappers
# ---------------------------------------------------------------------------

def sgmv_shrink_wrapper(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_lengths: torch.Tensor,
    lora_indices: torch.Tensor,
    batch_size: int,
    max_length: int,
    token_nums: int,
    scale: float,
):
    from vllm_ascend.lora.lora_ops import sgmv_shrink
    sgmv_shrink(x, w, y, b_seq_start_loc, seq_lengths, lora_indices,
                batch_size, max_length, token_nums, scale)


def sgmv_expand_slice_wrapper(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_lengths: torch.Tensor,
    lora_indices: torch.Tensor,
    batch_size: int,
    max_length: int,
    token_nums: int,
    y_offset: int,
    y_slice_size: int,
    add_inputs: bool,
):
    from vllm_ascend.lora.lora_ops import sgmv_expand_slice
    sgmv_expand_slice(x, w, y, b_seq_start_loc, seq_lengths, lora_indices,
                      batch_size, max_length, token_nums,
                      y_offset, y_slice_size, add_inputs)


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------

def generate_token_lora_tensor(
    num_tokens: int,
    num_loras: int,
    num_requests: int,
    device: torch.device,
    include_no_lora: bool = False,
) -> torch.Tensor:
    """Generate a realistic interleaved token-to-lora mapping.

    Creates num_requests sequences, each assigned a random LoRA (or -1).
    Tokens are distributed roughly evenly across requests.
    """
    tokens_per_req = num_tokens // max(num_requests, 1)
    remainder = num_tokens - tokens_per_req * num_requests

    lora_ids = list(range(num_loras))
    if include_no_lora:
        lora_ids.append(-1)

    mapping = []
    for i in range(num_requests):
        lora_id = lora_ids[i % len(lora_ids)]
        count = tokens_per_req + (1 if i < remainder else 0)
        mapping.extend([lora_id] * count)

    return torch.tensor(mapping[:num_tokens], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def check_close(name: str, ref: torch.Tensor, test: torch.Tensor, atol: float, rtol: float) -> bool:
    ref_f = ref.float()
    test_f = test.float()
    diff = (ref_f - test_f).abs()
    max_diff = diff.max().item()
    denom = ref_f.abs().clamp(min=1e-8)
    max_rel = (diff / denom).max().item()
    ok = torch.allclose(ref_f, test_f, atol=atol, rtol=rtol)
    status = "OK  " if ok else "FAIL"
    print(f"  {status} {name:20s}  max_abs={max_diff:.6f}  max_rel={max_rel:.6f}")
    return ok


def test_shrink(
    num_tokens: int, num_loras: int, num_slices: int, rank: int,
    hidden_size: int, num_requests: int, device: torch.device,
    atol: float, rtol: float, dtype: torch.dtype,
):
    tag = f"SHRINK  T={num_tokens} L={num_loras} S={num_slices} R={rank} reqs={num_requests}"
    print(f"\n{tag}")

    # Clear weight cache between tests
    _TRANSPOSED_WEIGHT_CACHE.clear()

    torch.manual_seed(42)
    # LoRA A weights: [num_loras, rank, hidden_size] (vllm layout: [num_loras, N, K])
    lora_a_stacked = [
        torch.randn(num_loras, rank, hidden_size, dtype=dtype, device=device).contiguous()
        for _ in range(num_slices)
    ]
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device).contiguous()

    token_lora = generate_token_lora_tensor(num_tokens, num_loras, num_requests, device)
    (b_seq_start, seq_lengths, lora_indices, batch_size,
     max_length, token_nums, no_lora) = compute_meta(token_lora)

    print(f"  groups={batch_size}  lora_pattern={lora_indices.tolist()[:10]}{'...' if batch_size > 10 else ''}")
    print(f"  seq_lengths={seq_lengths.tolist()[:10]}{'...' if batch_size > 10 else ''}  sum={seq_lengths.sum().item()}")

    all_ok = True
    for s in range(num_slices):
        # --- Reference (torch, fp32) ---
        ref = torch_shrink(x, lora_a_stacked[s], lora_indices, seq_lengths, 1.0)

        # --- SGMV ---
        sgmv_out = torch.zeros(num_tokens, rank, dtype=torch.float32, device=device)
        sgmv_shrink_wrapper(
            x, lora_a_stacked[s], sgmv_out,
            b_seq_start, seq_lengths, lora_indices,
            batch_size, max_length, token_nums, 1.0,
        )

        # --- GMM ---
        gmm_out = gmm_shrink(x, lora_a_stacked[s], lora_indices, seq_lengths, 1.0)

        all_ok &= check_close(f"sgmv  slice={s}", ref, sgmv_out, atol, rtol)
        all_ok &= check_close(f"gmm   slice={s}", ref, gmm_out, atol, rtol)

    return all_ok


def test_expand(
    num_tokens: int, num_loras: int, num_slices: int, rank: int,
    output_size: int, num_requests: int, device: torch.device,
    atol: float, rtol: float, dtype: torch.dtype,
):
    tag = f"EXPAND  T={num_tokens} L={num_loras} S={num_slices} R={rank} out={output_size} reqs={num_requests}"
    print(f"\n{tag}")

    _TRANSPOSED_WEIGHT_CACHE.clear()

    torch.manual_seed(42)
    slice_size = output_size // num_slices
    # LoRA B weights: [num_loras, out_slice, rank] (vllm layout: [num_loras, N, K] where K=rank)
    lora_b_stacked = [
        torch.randn(num_loras, slice_size, rank, dtype=dtype, device=device).contiguous()
        for _ in range(num_slices)
    ]
    # sgmv_expand expects float32 input (it's the output of shrink which accumulates in fp32)
    x_slices = [
        torch.randn(num_tokens, rank, dtype=torch.float32, device=device).contiguous()
        for _ in range(num_slices)
    ]

    token_lora = generate_token_lora_tensor(num_tokens, num_loras, num_requests, device)
    (b_seq_start, seq_lengths, lora_indices, batch_size,
     max_length, token_nums, no_lora) = compute_meta(token_lora)

    print(f"  groups={batch_size}  lora_pattern={lora_indices.tolist()[:10]}{'...' if batch_size > 10 else ''}")

    # --- SGMV reference (fp32 input, fp32 matmul, overwrite semantics) ---
    sgmv_ref_y = torch.zeros(num_tokens, output_size, dtype=dtype, device=device)
    offset = 0
    for s in range(num_slices):
        torch_expand_slice(sgmv_ref_y, x_slices[s], lora_b_stacked[s],
                           lora_indices, seq_lengths, offset, slice_size, False)
        offset += slice_size

    # --- GMM reference (cast x to weight dtype first, then matmul — matches gmm behavior) ---
    gmm_ref_y = torch.zeros(num_tokens, output_size, dtype=dtype, device=device)
    offset = 0
    for s in range(num_slices):
        torch_expand_slice(gmm_ref_y, x_slices[s].to(dtype), lora_b_stacked[s],
                           lora_indices, seq_lengths, offset, slice_size, True)
        offset += slice_size

    # --- SGMV ---
    sgmv_y = torch.zeros(num_tokens, output_size, dtype=dtype, device=device)
    offset = 0
    for s in range(num_slices):
        sgmv_expand_slice_wrapper(
            x_slices[s], lora_b_stacked[s], sgmv_y,
            b_seq_start, seq_lengths, lora_indices,
            batch_size, max_length, token_nums,
            offset, slice_size, True,
        )
        offset += slice_size

    # --- GMM ---
    gmm_y = torch.zeros(num_tokens, output_size, dtype=dtype, device=device)
    offset = 0
    for s in range(num_slices):
        gmm_expand_slice(gmm_y, x_slices[s], lora_b_stacked[s],
                         lora_indices, seq_lengths, offset, slice_size, True)
        offset += slice_size

    all_ok = True
    all_ok &= check_close("sgmv  expand", sgmv_ref_y, sgmv_y, atol, rtol)
    all_ok &= check_close("gmm   expand", gmm_ref_y, gmm_y, atol, rtol)
    return all_ok


def test_shrink_with_no_lora(
    num_tokens: int, num_loras: int, rank: int,
    hidden_size: int, num_requests: int, device: torch.device,
    atol: float, rtol: float, dtype: torch.dtype,
):
    """Test with some groups having no LoRA (index = -1)."""
    tag = f"SHRINK (mixed no-lora)  T={num_tokens} L={num_loras} R={rank} reqs={num_requests}"
    print(f"\n{tag}")

    _TRANSPOSED_WEIGHT_CACHE.clear()

    torch.manual_seed(42)
    w = torch.randn(num_loras, rank, hidden_size, dtype=dtype, device=device).contiguous()
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device).contiguous()

    token_lora = generate_token_lora_tensor(
        num_tokens, num_loras, num_requests, device, include_no_lora=True
    )
    (b_seq_start, seq_lengths, lora_indices, batch_size,
     max_length, token_nums, no_lora) = compute_meta(token_lora)

    print(f"  groups={batch_size}  lora_pattern={lora_indices.tolist()[:10]}")

    ref = torch_shrink(x, w, lora_indices, seq_lengths, 1.0)

    sgmv_out = torch.zeros(num_tokens, rank, dtype=torch.float32, device=device)
    sgmv_shrink_wrapper(
        x, w, sgmv_out,
        b_seq_start, seq_lengths, lora_indices,
        batch_size, max_length, token_nums, 1.0,
    )

    gmm_out = gmm_shrink(x, w, lora_indices, seq_lengths, 1.0)

    all_ok = True
    all_ok &= check_close("sgmv  (no-lora)", ref, sgmv_out, atol, rtol)
    all_ok &= check_close("gmm   (no-lora)", ref, gmm_out, atol, rtol)
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test gmm vs sgmv correctness")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--output-size", type=int, default=4096, help="Total output dim for expand tests")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 32, 128, 512, 2048, 8192])
    parser.add_argument("--num-loras", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--num-slices", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--num-requests", type=int, nargs="+", default=[1, 4, 16, 32, 64])
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--shrink-only", action="store_true")
    parser.add_argument("--expand-only", action="store_true")
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

    # --- Shrink tests ---
    if not args.expand_only:
        print("\n" + "=" * 60)
        print("SHRINK TESTS")
        print("=" * 60)
        for (T, L, S, R, reqs) in itertools.product(
            args.tokens, args.num_loras, args.num_slices, args.ranks, args.num_requests,
        ):
            if reqs > T:
                continue
            total += 1
            ok = test_shrink(T, L, S, R, args.hidden_size, reqs, device, args.atol, args.rtol, dtype)
            if not ok:
                fails += 1

        # Mixed no-lora tests
        print("\n" + "=" * 60)
        print("SHRINK WITH NO-LORA GROUPS")
        print("=" * 60)
        for (T, L, R, reqs) in itertools.product(
            [32, 512, 4096], [2, 4], [16, 32], [4, 16],
        ):
            if reqs > T:
                continue
            total += 1
            ok = test_shrink_with_no_lora(T, L, R, args.hidden_size, reqs, device, args.atol, args.rtol, dtype)
            if not ok:
                fails += 1

    # --- Expand tests ---
    if not args.shrink_only:
        print("\n" + "=" * 60)
        print("EXPAND TESTS")
        print("=" * 60)
        for (T, L, S, R, reqs) in itertools.product(
            args.tokens, args.num_loras, args.num_slices, args.ranks, args.num_requests,
        ):
            if reqs > T:
                continue
            total += 1
            ok = test_expand(T, L, S, R, args.output_size, reqs, device, args.atol, args.rtol, dtype)
            if not ok:
                fails += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {total - fails}/{total} passed, {fails} failed")
    print("=" * 60)
    sys.exit(1 if fails > 0 else 0)


if __name__ == "__main__":
    main()
