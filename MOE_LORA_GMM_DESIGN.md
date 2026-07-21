# MoE-LoRA: single-GEMM path by reusing the experts' routing metadata

Goal: replace the per-row `bgmv` MoE-LoRA with **grouped-matmul on the cube cores**, and compute the
LoRA metadata by **reusing what the base expert GMM already built**, so we also delete the per-layer
`argsort` / `FloorDiv` / `Index` / `And`/`Or` churn. Target: one `npu_grouped_matmul` per shrink and
per expand, sharing the base `group_list`.

## 0. What about decode?

The kernel crossover is real: at tiny token counts bgmv beats gmm (microbench: T=16 → bgmv 26 µs vs
gmm 159 µs). In decode there are ~`num_seqs * top_k / num_experts` ≈ **<1 row per expert**, which is
pathological for a per-expert grouped matmul. So **for the kernel, decode should keep bgmv** — there
is no free lunch on the matmul itself.

BUT the decode problem today is not mainly the kernel — it's the **metadata**: `_recover_moe_lora_routing`
runs `argsort` + `// top_k` (FloorDiv on AI_CPU) + gathers **every layer** (48x), costing ~215 ms in
the b16 trace (FloorDiv 82 ms + Index 98 ms + Sort 13 ms + Cast/Mul/GreaterEqual/And). That we *can*
kill for both regimes by reusing the base routing. So:

- **prefill**: gmm (reuse `group_list`) — the 25x kernel win.
- **decode**: keep bgmv, but fed from the reused base routing — kills the argsort/floordiv/gather.
- One `token_num > threshold` switch (the dense path already has `LORA_GMM`/`GMM_TOKEN_THRESHOLD=1024`).

The user's "use the gmm metadata on both" is worth *measuring*: a gmm-only unified path pays the flat
~160 µs gmm overhead per call in decode (× 48 layers × 6 calls ≈ several ms/step) but removes the
~215 ms/window of metadata ops. Net decode TPOT is an empirical question — the draft below makes both
the gmm apply and the metadata reuse available so we can A/B it. My prior: gmm-only regresses decode
TPOT; threshold-switch wins both. We should confirm with a decode-only profile.

## 1. What the base expert path already computes (moe_mlp.py `unquant_apply_mlp`)

```
gate_up_out = npu_grouped_matmul(x=[hidden_states], weight=[w1], group_list=group_list,
                                 group_list_type=group_list_type, group_type=0)[0]
...
down_out    = npu_grouped_matmul(x=[gate_up_out],  weight=[w2], group_list=group_list, ...)[0]
```

So at the LoRA hook we already hold, for free, the exact tensors a grouped LoRA matmul needs:
- `hidden_states` — the activation **already permuted into expert-contiguous order** (w13 shrink input).
- `gate_up_out` — the swiglu output, same expert order (w2 shrink input).
- `group_list` (+ `group_list_type`) — per-expert token counts, **the group_list for LoRA too**.

The current LoRA hook ignores all three and instead recovers per-row `(expert, lora)` via argsort of
`expanded_row_idx`. That's the redundant work.

## 2. Weight layout

`create_lora_weights` gives, per slice `s`:
- `w13_lora_a_stacked[s]`: `[max_loras, num_experts, rank, hidden]`
- `w13_lora_b_stacked[s]`: `[max_loras, num_experts, out_s, rank]`  (out_s = w13 slice width)
- `w2_lora_a_stacked[0]`: `[max_loras, num_experts, rank, moe_inter]`
- `w2_lora_b_stacked[0]`: `[max_loras, num_experts, hidden, rank]`

`npu_grouped_matmul(group_type=0)` wants `x:[M,K]`, `weight:[num_experts,K,N]`. So per expert:
- shrink: `weight_a = lora_a[lora].transpose(1,2)` → `[num_experts, hidden, rank]`, out `[M, rank]`.
- expand: `weight_b = lora_b[lora].transpose(1,2)` → `[num_experts, rank, out_s]`, out `[M, out_s]`.

(Base already does `w1.transpose(1,2)`; identical trick.)

## 3. Draft — single active adapter (the common serving case; our benchmark)

When exactly one adapter is active in the batch, every row routed to expert `e` uses that adapter's
`(lora, e)` weight → grouping by expert is **exactly** the base `group_list`. No per-row gather, no
combined_idx, no routing recovery.

### 3a. `all.py` (PunicaWrapperNPU): new gmm apply

```python
def add_lora_fused_moe_gmm(
    self, y, x, lora_a_stacked, lora_b_stacked, *,
    group_list, group_list_type, lora_id: int,
    topk_weights=None, mul_routed_weight=False, offset=0,
):
    """Grouped-matmul MoE-LoRA for a single active adapter. x is the expert-
    permuted activation; group_list is the base per-expert token count. One
    npu_grouped_matmul for shrink and one for expand, per slice, on the cube
    cores -- no per-row bgmv, no routing recovery."""
    x2d = x.view(-1, x.shape[-1])
    y2d = y.view(-1, y.shape[-1])
    cur = offset
    for s in range(len(lora_a_stacked)):
        a = lora_a_stacked[s][lora_id]          # [num_experts, rank, hidden]
        b = lora_b_stacked[s][lora_id]          # [num_experts, out_s, rank]
        out_s = b.shape[-2]
        a_t = a.transpose(1, 2).contiguous()    # [num_experts, hidden, rank]
        b_t = b.transpose(1, 2).contiguous()    # [num_experts, rank, out_s]
        shrink = torch_npu.npu_grouped_matmul(
            x=[x2d], weight=[a_t], split_item=2, group_type=0,
            group_list_type=group_list_type, group_list=group_list)[0]   # [M, rank]
        if mul_routed_weight and topk_weights is not None:
            shrink = shrink * topk_weights.view(-1, 1)
        delta = torch_npu.npu_grouped_matmul(
            x=[shrink], weight=[b_t], split_item=2, group_type=0,
            group_list_type=group_list_type, group_list=group_list)[0]   # [M, out_s]
        y2d[:, cur:cur + out_s] += delta.to(y2d.dtype)
        cur += out_s
```

(`a_t`/`b_t` transposes should be cached on the stacked weight like `_TRANSPOSED_WEIGHT_CACHE` /
the base `need_trans` — they are static per adapter, computed once, not per forward.)

### 3b. `moe/ops/moe_mlp.py`: thread `group_list` into the hook (it's already in scope)

```python
lora_routing = moe_lora_apply_w13(
    lora_context, gate_up_out=gate_up_out, hidden_states=hidden_states,
    expanded_row_idx=expanded_row_idx, topk_ids=topk_ids,
    group_list=group_list, group_list_type=group_list_type,   # <-- add
)
...
moe_lora_apply_w2(lora_context, down_out=hidden_states, silu_out=gate_up_out,
                  lora_routing=lora_routing,
                  group_list=group_list, group_list_type=group_list_type)  # <-- add
```

### 3c. `moe/lora/fused_moe.py`: pick gmm vs bgmv, skip routing recovery on the gmm path

```python
def moe_lora_apply_w13(lora_context, *, gate_up_out, hidden_states,
                       expanded_row_idx, topk_ids, group_list, group_list_type):
    pw = lora_context.punica_wrapper
    single_lora = pw.single_active_lora_id  # host int or None, set in update_metadata
    if pw.use_moe_gmm and single_lora is not None:
        pw.add_lora_fused_moe_gmm(
            y=gate_up_out, x=hidden_states,
            lora_a_stacked=lora_context.w13_lora_a_stacked,
            lora_b_stacked=lora_context.w13_lora_b_stacked,
            group_list=group_list, group_list_type=group_list_type,
            lora_id=single_lora)
        return ("gmm", group_list, group_list_type)     # w2 reuses; no argsort at all
    # else: existing bgmv path (decode / multi-lora) -- unchanged
    routing = _recover_moe_lora_routing(lora_context, expanded_row_idx, topk_ids)
    ... existing add_lora_fused_moe(...) ...
    return ("bgmv",) + routing
```

`single_active_lora_id` and `use_moe_gmm` are set once per step in `update_metadata` (host-side, from
`token_lora_indices` + `adapter_enabled` + `token_num > GMM_TOKEN_THRESHOLD`) — no per-layer cost, no
device sync, graph-capture safe (values change, shapes don't; same as the base `group_list`).

**Ops eliminated on this path (per layer):** `argsort`(Sort), `// top_k`(FloorDiv, AI_CPU), the two
`Index` gathers, `combined_idx` (`GreaterEqual`+`LogicalAnd`+`Mul`+`Where`). ~215 ms/window in the
trace → ~0.

## 4. Multi-adapter extension (general correctness)

When >1 adapter is active, rows within one expert block belong to different loras, so per-expert
grouping is insufficient. Reuse the base expert permutation and add a **stable sub-sort by lora
within expert** to get `(expert, lora)` groups:

1. `combined = expert_per_row * max_loras + lora_per_row`  (expert-major, lora-minor; `expert_per_row`
   is free from the base routing, `lora_per_row` from the constant `token_lora_indices`).
2. `order = argsort(combined.float())` (value-independent shape → capturable, the existing float trick).
3. `fine_group_list = bincount(combined, minlength=num_experts*max_loras)` (fixed shape → capturable).
4. gmm with `weight = lora_*_stacked.permute to [num_experts, max_loras, ...].reshape(E*L, K, N)` and
   `group_list = fine_group_list`, on `x[order]`; scatter the delta back with `order`.

Still one gmm per shrink/expand on the cube cores; the only extra work is one small sort + one bincount
+ one gather/scatter per layer — orders of magnitude below the current per-row bgmv. (`max_loras` is
small, so `E*L` groups stay modest.)

## 5. Validation plan

1. Numerical: extend `test_kernels.py` with a MoE case — compare gmm-apply vs current bgmv-apply
   output on random routing (atol/rtol as existing).
2. Prefill microbench already done (`bench_lora_kernels.py`): gmm 25x at prefill row counts.
3. End-to-end: profile b16 with the gmm path — expect LoRA `bgmv` ~4804 ms → grouped-matmul ~150-300 ms,
   plus the ~215 ms metadata ops gone; total rank0 ~5693 ms → ~800 ms (near the 604 ms base).
4. Decode: decode-only profile (prompt 8, gen 128, batch 16) gmm-only vs bgmv+reused-metadata, to
   decide the threshold. Confirms whether decode wants gmm at all.
5. Correctness of the multi-lora path: 2 adapters in one batch, compare to bgmv reference.

## Status — IMPLEMENTED & VALIDATED (single-adapter path)

The single-adapter path (3a-3c) is implemented and validated on `va18_misha` (v0.18 + backport),
Qwen3-30B-A3B TP4, `LORA_GMM=threshold`.

**Correctness**
- Kernel: `test_moe_lora_gmm.py` — gmm apply vs torch per-expert reference, rel_rms ~2.3e-3 (bf16 noise)
  across w13/w2 shapes, empty experts, and the full 131072-row case. All PASS. (Wrong grouping would be
  O(1) rel_rms.)
- End-to-end: coherent temp-0 generation with the gmm path live.

**Measured (b16 profile, rank0, prefill 1024 / gen 16):**

| | BEFORE (bgmv) | AFTER (gmm MoE-LoRA) |
|---|---:|---:|
| total rank0 kernel | 5693 ms | **2065 ms (2.76x)** |
| LoRA `bgmv` (vector) | 4804 ms (84%) | 472 ms (decode-only, by design) |
| `GroupedMatmul` (cube) | 179 ms | 674 ms (MoE FFN + LoRA gmm) |
| routing-metadata ops | 218 ms | 134 ms |
| batch of 16 (e2e) | 6.37 s | **2.62 s (2.4x)** |

Prefill MoE-LoRA (~4.3 s bgmv) moved onto the cube cores (+495 ms GroupedMatmul ≈ 9x for that compute);
decode correctly stayed on bgmv (the 472 ms residual, token_num < threshold). Trace saved at
`traces/lora_moegmm/`.

**Follow-ups (not yet done):** cache the weight transpose per (adapter, slice) (currently inline
`.contiguous()` each call — some of the +495 ms and the Cast bump is this); the multi-adapter path (§4);
a decode-only A/B to decide whether decode ever wants gmm (my prior: no). Files changed: `all.py`,
`moe/lora/fused_moe.py`, `moe/ops/moe_mlp.py`; test `test_moe_lora_gmm.py`.
