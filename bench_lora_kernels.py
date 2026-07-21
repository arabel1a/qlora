#!/usr/bin/env python3
"""Crossover microbenchmark: bgmv (decode kernel) vs gmm (npu_grouped_matmul,
prefill kernel) vs sgmv, at the Qwen3-30B-A3B geometry (hidden=2048, rank=16).

Proves the effect behind "MoE-LoRA (bgmv) costs more than the MoE FFN":
  - bgmv runs on AI_VECTOR_CORE, per-token gather -> cost ~ linear in tokens,
    memory-bound, ~flat in rank.
  - gmm runs on the cube cores -> wins above ~GMM_TOKEN_THRESHOLD (1024) tokens.
The MoE-LoRA path (add_lora_fused_moe) is hardcoded to bgmv, so at prefill token
counts it sits on the wrong side of this crossover.
"""
import argparse, time
import torch, torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401  registers _C_ascend ops
import vllm_ascend.meta_registration  # noqa: F401
from vllm.lora.punica_wrapper.utils import compute_meta
from vllm_ascend.lora.lora_ops import bgmv_shrink, bgmv_expand_slice
from test_kernels import (gmm_shrink, gmm_expand_slice, generate_token_lora_tensor,
                          _TRANSPOSED_WEIGHT_CACHE)

def bench(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us/call

def run(tokens, rank, hidden, out_size, num_requests, num_loras, dev, dtype):
    print(f"\n{'='*96}")
    print(f"hidden={hidden} rank={rank} out={out_size} num_loras={num_loras} reqs={num_requests} dtype={dtype}")
    print(f"{'='*96}")
    hdr = f"{'tokens':>8} | {'bgmv_shr':>9} {'gmm_shr':>9} {'shr_win':>8} | {'bgmv_exp':>9} {'gmm_exp':>9} {'exp_win':>8}"
    print(hdr); print("-"*len(hdr))
    for T in tokens:
        _TRANSPOSED_WEIGHT_CACHE.clear()  # keyed by data_ptr(), which torch recycles across iters
        torch.manual_seed(42)
        x = torch.randn(T, hidden, dtype=dtype, device=dev).contiguous()
        a = torch.randn(num_loras, rank, hidden, dtype=dtype, device=dev).contiguous()
        tl = generate_token_lora_tensor(T, num_loras, min(num_requests, T), dev)
        (_bss, seq_lengths, lora_indices, _bs, _ml, _tn, _nl) = compute_meta(tl)
        # shrink
        so = torch.zeros(T, rank, dtype=torch.float32, device=dev)
        a_flat = a.view(-1, rank, hidden)
        t_bgmv_s = bench(lambda: bgmv_shrink(x, a_flat, so, tl, 1.0))
        t_gmm_s  = bench(lambda: gmm_shrink(x, a, lora_indices, seq_lengths, 1.0))
        # expand
        b = torch.randn(num_loras, out_size, rank, dtype=dtype, device=dev).contiguous()
        xr = torch.randn(T, rank, dtype=torch.float32, device=dev).contiguous()
        y = torch.zeros(T, out_size, dtype=dtype, device=dev)
        b_flat = b.view(-1, out_size, rank)
        t_bgmv_e = bench(lambda: bgmv_expand_slice(xr, b_flat, y, tl, 0, out_size, add_inputs=True))
        t_gmm_e  = bench(lambda: gmm_expand_slice(y, xr, b, lora_indices, seq_lengths, 0, out_size, True))
        sw = "gmm" if t_gmm_s < t_bgmv_s else "bgmv"
        ew = "gmm" if t_gmm_e < t_bgmv_e else "bgmv"
        print(f"{T:>8} | {t_bgmv_s:9.1f} {t_gmm_s:9.1f} {sw:>8} | {t_bgmv_e:9.1f} {t_gmm_e:9.1f} {ew:>8}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--out-size", type=int, default=768)  # moe_intermediate_size
    p.add_argument("--tokens", type=int, nargs="+",
                   default=[16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    p.add_argument("--ranks", type=int, nargs="+", default=[16])
    p.add_argument("--num-requests", type=int, default=16)
    p.add_argument("--num-loras", type=int, default=1)
    p.add_argument("--device", type=str, default="npu:0")
    a = p.parse_args()
    dev = torch.device(a.device); dtype = torch.bfloat16
    print(f"Device {dev}  hidden={a.hidden}  out={a.out_size}  ranks={a.ranks}  tokens={a.tokens}")
    for r in a.ranks:
        run(a.tokens, r, a.hidden, a.out_size, a.num_requests, a.num_loras, dev, dtype)
    # rank sweep at a fixed large prefill token count -> FLOP-vs-overhead proof
    print(f"\n\n### RANK SWEEP @ T=8192 (bgmv ~flat in rank => memory-bound, not FLOP-bound) ###")
    for r in [8, 16, 32, 64, 128]:
        run([8192], r, a.hidden, a.out_size, a.num_requests, a.num_loras, dev, dtype)

if __name__ == "__main__":
    main()
