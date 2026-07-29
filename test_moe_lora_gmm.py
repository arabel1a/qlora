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


def ref_multi(x, a, b, expert_per_row, lora_per_row, active):
    """Per-row reference for the multi-adapter path.

    x[M,hidden], a[L,E,rank,hidden], b[L,E,out,rank]. Row i uses weight
    (lora_per_row[i], expert_per_row[i]); inactive rows contribute 0.
    """
    M, out = x.shape[0], b.shape[2]
    ref = torch.zeros(M, out, dtype=torch.float32, device=x.device)
    xf = x.to(torch.float32)
    for i in range(M):
        if not bool(active[i]):
            continue
        e = int(expert_per_row[i].item()); l = int(lora_per_row[i].item())
        ref[i] = xf[i] @ a[l, e].to(torch.float32).T @ b[l, e].to(torch.float32).T
    return ref


def gmm_apply_multi(x, a, b, expert_per_row, lora_per_row, adapter_enabled, active_slots):
    """Exactly _build_moe_gmm_multi_plan + add_lora_fused_moe_gmm_multi (one slice),
    with compaction over the active slots (grouping over num_experts * n_active)."""
    max_loras, num_experts = a.shape[0], a.shape[1]
    rank, hidden = a.shape[2], a.shape[3]
    out_s = b.shape[2]
    m = x.shape[0]
    n_active = len(active_slots)
    dev = x.device
    # --- plan (mirror _build_moe_gmm_multi_plan) ---
    lora_safe = lora_per_row.clamp(min=0)
    active = (lora_per_row >= 0) & adapter_enabled[lora_safe].bool()
    active_slots_dev = torch.tensor(active_slots, dtype=torch.long, device=dev)
    slot_to_compact = torch.zeros(max_loras, dtype=torch.long, device=dev)
    slot_to_compact[active_slots_dev] = torch.arange(n_active, dtype=torch.long, device=dev)
    compact = slot_to_compact[lora_safe]
    combined = expert_per_row.to(torch.long) * n_active + compact
    order = torch.argsort(combined.to(torch.float32))
    fine_group_list = torch.bincount(combined, minlength=num_experts * n_active)
    active_mask_sorted = active.index_select(0, order)
    # --- apply (mirror add_lora_fused_moe_gmm_multi) ---
    x_sorted = x.index_select(0, order)
    a_sel = a.index_select(0, active_slots_dev)  # [n_active, E, rank, hidden]
    b_sel = b.index_select(0, active_slots_dev)  # [n_active, E, out, rank]
    a_g = a_sel.permute(1, 0, 2, 3).reshape(num_experts * n_active, rank, hidden).transpose(1, 2).contiguous()
    b_g = b_sel.permute(1, 0, 2, 3).reshape(num_experts * n_active, out_s, rank).transpose(1, 2).contiguous()
    shrink = torch_npu.npu_grouped_matmul(
        x=[x_sorted], weight=[a_g], split_item=2, group_type=0,
        group_list_type=1, group_list=fine_group_list)[0]
    delta = torch_npu.npu_grouped_matmul(
        x=[shrink], weight=[b_g], split_item=2, group_type=0,
        group_list_type=1, group_list=fine_group_list)[0]
    delta = delta * active_mask_sorted.view(-1, 1).to(delta.dtype)
    out = torch.zeros((m, out_s), dtype=delta.dtype, device=delta.device)
    out.index_copy_(0, order, delta)
    return out, active


def run_multi(M, E, max_loras, n_active, rank, hidden, out, device, dtype, inactive_frac, tag):
    """max_loras = stack capacity (allocated slots); n_active = adapters actually
    present in the batch (compaction target). E*n_active must be <= 1024 for the
    fused gmm; larger falls back to bgmv in production (not exercised here)."""
    torch.manual_seed(7)
    x = torch.randn(M, hidden, dtype=dtype, device=device).contiguous()
    a = (torch.randn(max_loras, E, rank, hidden, dtype=dtype, device=device) * 0.02).contiguous()
    b = (torch.randn(max_loras, E, out, rank, dtype=dtype, device=device) * 0.02).contiguous()
    # x is expert-permuted -> expert ids are non-decreasing (built from counts).
    counts = make_counts(M, E, device)
    expert_per_row = torch.repeat_interleave(torch.arange(E, device=device), counts)
    # active slots: n_active distinct slots out of the max_loras capacity.
    g = torch.Generator().manual_seed(3)
    active_slots = sorted(torch.randperm(max_loras, generator=g)[:n_active].tolist())
    # each row picks one of the active slots; a fraction go inactive (id -1).
    pick = torch.randint(0, n_active, (M,), device=device)
    lora_per_row = torch.tensor(active_slots, device=device)[pick]
    inactive = torch.rand(M, device=device) < inactive_frac
    lora_per_row = torch.where(inactive, torch.full_like(lora_per_row, -1), lora_per_row)
    adapter_enabled = torch.ones(max_loras, dtype=torch.bool, device=device)
    ref = ref_multi(x, a, b, expert_per_row, lora_per_row, ~inactive)
    got, _ = gmm_apply_multi(x, a, b, expert_per_row, lora_per_row, adapter_enabled, active_slots)
    got = got.to(torch.float32)
    max_abs = (got - ref).abs().max().item()
    denom = torch.linalg.vector_norm(ref).item()
    rel_rms = (torch.linalg.vector_norm(got - ref).item() / denom) if denom > 0 else max_abs
    ok = rel_rms < 3e-2
    print(f"[{'PASS' if ok else 'FAIL'}] {tag:36s} M={M:>6} E={E} cap={max_loras} act={n_active} "
          f"h={hidden} out={out} inact={inactive_frac:.1f}  grp={E*n_active} rel_rms={rel_rms:.4e}")
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
    print("\n--- multi-adapter (expert, active-lora) grouping, compaction + 1024 cap ---")
    # run_multi(M, E, cap=max_loras, n_active, rank, hidden, out, ..., inactive_frac, tag)
    ok &= run_multi(16384, E, 8,  4, rank, hidden, inter, dev, dt, 0.0,  "w13 4 active / cap 8")
    ok &= run_multi(16384, E, 8,  4, rank, inter, hidden, dev, dt, 0.0,  "w2  4 active / cap 8")
    ok &= run_multi(16384, E, 8,  8, rank, hidden, inter, dev, dt, 0.25, "w13 8 active (grp=1024 cap) + no-lora")
    ok &= run_multi(4096,  E, 4,  2, rank, hidden, inter, dev, dt, 0.5,  "w13 2 active half no-lora")
    ok &= run_multi(16384, E, 32, 4, rank, hidden, inter, dev, dt, 0.0,  "w13 cap 32, only 4 active (compaction)")
    ok &= run_multi(1024,  E, 8,  6, rank, hidden, inter, dev, dt, 0.1,  "w13 6 active small batch")
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))


if __name__ == "__main__":
    main()
