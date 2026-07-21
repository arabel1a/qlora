#!/usr/bin/env python3
"""Correctness of the grouped-matmul MoE-LoRA apply (add_lora_fused_moe_gmm)
against a torch per-expert reference, at the Qwen3-30B-A3B geometry.

Mirrors the two npu_grouped_matmul calls (shrink then expand) that
PunicaWrapperNPU.add_lora_fused_moe_gmm performs for one slice, on the
expert-permuted activation with the base per-expert group_list (type 1 = counts).
"""
import argparse
import torch, torch_npu  # noqa: F401


def ref_per_expert(x, a, b, counts):
    """x[M,hidden], a[E,rank,hidden], b[E,out,rank], counts[E] -> ref[M,out] (fp32 math)."""
    M, out = x.shape[0], b.shape[1]
    ref = torch.zeros(M, out, dtype=torch.float32, device=x.device)
    start = 0
    for e in range(counts.shape[0]):
        n = int(counts[e].item())
        if n == 0:
            continue
        xe = x[start:start + n].to(torch.float32)          # [n, hidden]
        de = xe @ a[e].to(torch.float32).T @ b[e].to(torch.float32).T  # [n,out]
        ref[start:start + n] = de
        start += n
    return ref


def gmm_apply(x, a, b, group_list, group_list_type):
    """Exactly what add_lora_fused_moe_gmm does for one slice."""
    a_t = a.transpose(1, 2).contiguous()   # [E, hidden, rank]
    b_t = b.transpose(1, 2).contiguous()   # [E, rank, out]
    shrink = torch_npu.npu_grouped_matmul(
        x=[x], weight=[a_t], split_item=2, group_type=0,
        group_list_type=group_list_type, group_list=group_list)[0]
    delta = torch_npu.npu_grouped_matmul(
        x=[shrink], weight=[b_t], split_item=2, group_type=0,
        group_list_type=group_list_type, group_list=group_list)[0]
    return delta


def make_counts(M, E, device, empty_frac=0.0):
    """Random per-expert token counts summing to M (some experts may be empty)."""
    torch.manual_seed(0)
    probs = torch.rand(E)
    if empty_frac > 0:
        probs[torch.rand(E) < empty_frac] = 0.0
    probs = probs / probs.sum()
    counts = (probs * M).floor().to(torch.int64)
    counts[0] += M - int(counts.sum().item())  # fix rounding
    assert int(counts.sum().item()) == M and (counts >= 0).all()
    return counts.to(device)


def run(M, E, rank, hidden, out, device, dtype, empty_frac, tag):
    torch.manual_seed(42)
    x = torch.randn(M, hidden, dtype=dtype, device=device).contiguous()
    a = torch.randn(E, rank, hidden, dtype=dtype, device=device).contiguous() * 0.02
    b = torch.randn(E, out, rank, dtype=dtype, device=device).contiguous() * 0.02
    counts = make_counts(M, E, device, empty_frac)
    ref = ref_per_expert(x, a, b, counts)
    got = gmm_apply(x, a, b, counts, group_list_type=1).to(torch.float32)
    max_abs = (got - ref).abs().max().item()
    # L2 relative error is the right criterion for a bf16 matmul: it certifies the
    # GROUPING (each row hit its expert's weight) -- a wrong group_list would give
    # O(1) rel_rms, bf16 rounding gives ~1e-2. Per-element rel is meaningless on the
    # many near-zero delta entries.
    rel_rms = (torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref)).item()
    ok = rel_rms < 3e-2
    print(f"[{'PASS' if ok else 'FAIL'}] {tag:34s} M={M:>6} E={E} r={rank} h={hidden} out={out} "
          f"empty={empty_frac:.1f}  rel_rms={rel_rms:.4e} max_abs={max_abs:.2e}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="npu:0")
    a = p.parse_args()
    dev = torch.device(a.device); dt = torch.bfloat16
    print(f"Device {dev}  dtype {dt}")
    E, rank, hidden = 128, 16, 2048
    inter = 768  # moe_intermediate_size (w13 slice width / w2 input)
    ok = True
    # w13-shaped (hidden -> inter) and w2-shaped (inter -> hidden), prefill sizes
    ok &= run(16384, E, rank, hidden, inter, dev, dt, 0.0, "w13 gate/up (prefill)")
    ok &= run(16384, E, rank, inter, hidden, dev, dt, 0.0, "w2 down (prefill)")
    ok &= run(4096,  E, rank, hidden, inter, dev, dt, 0.3, "w13 with empty experts")
    ok &= run(1024,  E, rank, hidden, inter, dev, dt, 0.0, "w13 small batch")
    ok &= run(131072, E, rank, hidden, inter, dev, dt, 0.0, "w13 full expanded rows")
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))


if __name__ == "__main__":
    main()
