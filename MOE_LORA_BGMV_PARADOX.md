# Why MoE-LoRA (bgmv) costs more than the MoE FFN (GroupedMatmul)

Qwen3-30B-A3B, TP4, Atlas 910B4, r16 adapter. Traces: 16 concurrent requests, 1024 prefill / 16 gen.

## The paradox

From the profiler, with LoRA enabled the top operators are LoRA `bgmv`, not the MoE experts:

```
bgmv_shrink_bfloat16_t   AI_VECTOR_CORE   786   3,459,680 us   avg 4402  max 20333
bgmv_expand_bfloat16_t   AI_VECTOR_CORE   788   1,162,974 us   avg 1476  max  8830
GroupedMatmul            MIX_AIC         1728     179,232 us   avg  104  max  1498   <- the actual MoE FFN
```

Without LoRA the MoE `GroupedMatmul` (184 ms) is the top compute op and everything is fine.

So a **rank-16** LoRA update — ~3% of the FLOPs of the expert FFN it decorates — takes **~27x longer
than the entire MoE FFN**. That seems to contradict two things we know:

1. On this hardware grouped-matmul only beats bgmv **above ~1024 tokens/batch**.
2. MoE has *larger* matrices and dynamic per-layer routing, and there grouped-matmul clearly wins
   over bgmv.

If grouped-matmul wins for the big MoE matrices, how can the tiny LoRA update (on bgmv) cost more
than the MoE FFN (on grouped-matmul)?

## The answer (one line)

Both facts are true. The MoE FFN correctly runs on grouped-matmul (cube cores) because it is above
the 1024-token threshold. **The MoE-LoRA update does not** — `add_lora_fused_moe` is hardcoded to
`bgmv` regardless of token count, so during prefill it runs the *decode* kernel on the *prefill*
token set: the losing side of the exact crossover in fact #1. It runs on the vector cores, per token,
while the FFN it supplements runs on the cube cores, batched.

## Root cause in the code (`punica_npu.py` / `all.py`)

- Dense-linear LoRA (`add_shrink`/`add_expand`) **has** a bgmv→gmm switch:
  `enabled = token_num > GMM_TOKEN_THRESHOLD` (=**1024**), gated by env `LORA_GMM` (default `"off"`).
- MoE-expert LoRA (`add_lora_fused_moe`) calls `self.bgmv_shrink(...)` and
  `self.bgmv_expand_slice(...)` **unconditionally** — no threshold, no gmm path, ever.

The adapter targets both `q/k/v/o_proj` (dense) and `gate/up/down_proj` (MoE experts), so only the
dense half can ever switch; the MoE half — which processes `num_tokens × top_k` = 8x more rows — is
permanently on bgmv.

## Experiments

### E1 — Trace decomposition (end-to-end, rank0)

| | base (no LoRA) | LoRA |
|---|---:|---:|
| total kernel time | 604 ms | **5693 ms** (9.4x) |
| MoE FFN `GroupedMatmul` (cube) | 184 ms (30%) | 179 ms (3.1%) — **unchanged**, count 1728 both |
| LoRA `bgmv` (vector) | 0 | **4804 ms (84%)** |

LoRA adds **zero** grouped-matmul work; the entire cost is bgmv on vector cores. The big prefill
bgmv averages 4405 µs (max 20338 µs) vs the decode bgmv at 19 µs.

### E2 — Kernel crossover microbenchmark (`bench_lora_kernels.py`, hidden=2048, out=768, r16)

Isolated bgmv vs gmm (`npu_grouped_matmul`), µs/call:

| tokens | bgmv_shrink | gmm_shrink | winner | bgmv_expand | gmm_expand | winner |
|---:|---:|---:|:--:|---:|---:|:--:|
| 16   |   25.6 | 158.8 | bgmv |  25.4 | 209.1 | bgmv |
| 256  |   50.1 | 157.7 | bgmv |  25.5 | 208.6 | bgmv |
| 512  |   88.9 | 158.0 | bgmv |  31.9 | 207.6 | bgmv |
| **1024** | **172.6** | **158.5** | **gmm** | 57.5 | 205.8 | bgmv |
| 2048 |  340.4 | 159.1 | gmm | 108.2 | 205.7 | bgmv |
| 4096 |  668.9 | 293.1 | gmm | 208.5 | 209.3 | bgmv |
| 8192 | 1325.9 | 258.0 | gmm | 407.1 | 209.9 | gmm |
| 16384| 2649.2 | 469.2 | gmm | 808.1 | 257.8 | gmm |
| 32768| 5429.2 | 833.0 | gmm | 1615.5| 270.3 | gmm |

- `bgmv` scales **linearly** with tokens; `gmm` is **flat ~160 µs** (cube core saturated) until it
  finally has enough work to grow slowly.
- Shrink crossover is at **T=1024 — exactly `GMM_TOKEN_THRESHOLD`.** (Fact #1, confirmed on-device.)
- The MoE-LoRA processes `num_tokens × top_k` ≈ 131k rows per layer. Extrapolating bgmv_shrink to
  131k rows ≈ **21 ms** — matching the trace's `max=20338 µs`. gmm would do it in ≈ **0.8 ms → ~25x.**

### E3 — Rank sweep @ T=8192 (FLOPs vs kernel-efficiency)

| rank | bgmv_shrink | gmm_shrink |
|---:|---:|---:|
| 8   |  700 |  440 |
| 16  | 1325 |  305 |
| 32  | 2578 |  191 |
| 64  | 5086 |  192 |
| 128 |10099 |  150 |

`bgmv` scales linearly with rank (per-token × per-rank vector work); `gmm` is flat/faster (cube core,
more work = better utilization). The bgmv cost is a kernel-efficiency problem, not a FLOP problem.

### E4 — System A/B: enable the existing gmm switch (`LORA_GMM=threshold`)

Only the dense q/k/v/o LoRA can switch; MoE gate/up/down cannot.

| | LoRA bgmv (vector) | GroupedMatmul (cube) | batch of 16 |
|---|---:|---:|---:|
| `LORA_GMM=off`        | 4804 ms (84%) | 179 ms (count 1728 = MoE FFN) | 6.37 s |
| `LORA_GMM=threshold`  | **4031 ms (79%)** | 286 ms (count 2208 = 1728 + **480 dense-LoRA gmm**) | 5.78 s |

480 dense bgmv calls (~773 ms) became 480 gmm calls (~107 ms) — a ~7x local win, confirming gmm≫bgmv
at prefill scale. **bgmv is still 79%**: the residual is the fused-MoE LoRA, which has no gmm path.

## Resolution of the two "contradicting" facts

- "gmm beats bgmv above 1024 tokens" — **true**, crossover confirmed at T=1024 (E2).
- "for the big MoE matrices gmm wins over bgmv" — **true**, and it's exactly why the MoE FFN is cheap.
- No contradiction: the MoE-LoRA simply never makes the switch. It runs the decode kernel (bgmv,
  vector cores) on the prefill token set, on the losing side of a crossover that the MoE FFN is on
  the winning side of. Rank-16, ~3% of the FLOPs, ~27x the time ⇒ ~900x lower compute efficiency —
  entirely a wrong-kernel-for-the-regime effect, not a compute effect.

## Why bgmv is slow in absolute terms (not just "wrong side of a crossover")

The regime variable for bgmv-vs-gmm is **M = the number of tokens that share one weight**, not the
matrix's N/K (rank, hidden). bgmv wins only when M is tiny (decode, ~1 token/seq). MoE does NOT move
into that regime: prefill has num_tokens × top_k = 131072 routed rows (~1024 tokens/expert), and
`add_lora_fused_moe` processes all 131072 in one bgmv with a per-row `(lora,expert)` gather — it never
batches the ~1024 tokens that share an expert's LoRA weight (the FFN's GroupedMatmul does).

Per-kernel hardware counters (rank0, b16) show why the (smaller) LoRA matmul is slower than the
(larger) expert matmul:

| kernel | core | avg µs | block_dim | MTE2% (mem) | mac% | cube_util% |
|---|---|---:|---:|---:|---:|---:|
| `bgmv_shrink` (real prefill calls) | AI_VECTOR_CORE | ~4500 | 40 | 41.4 | 0 | 0 |
| `bgmv_expand` | AI_VECTOR_CORE | ~1475 | 40 | 15.2 | 0 | 0 |
| `GroupedMatmul` (MoE FFN) | MIX_AIC (cube) | 104 | 20 | 0 | 10.2 | 82.8 |

Three compounding reasons, all data-backed:
1. **Arithmetic intensity.** shrink reads [M,2048] to make a rank-16 output ⇒ ~2048·16/(2048·2) = 8
   MAC/byte. The expert gate/up reads the same activation to make 768-wide output ⇒ 384 MAC/byte.
   That is a **48x (=768/16) gap** — the LoRA is intrinsically memory-bound, the expert is
   compute-bound. Counters confirm: bgmv_shrink 41% MTE2 with mac=0; GroupedMatmul 83% cube util.
   So the *smaller* matrix is exactly what makes LoRA bandwidth-bound.
2. **Wrong cores.** bgmv runs on the vector units (mac=0), not the cube units the experts use.
3. **Scattered gather.** the per-row `combined_idx` gather gives poor memory coalescing (the 41% MTE2
   stall). block_dim=40 shows it already uses all vector cores — it's not an occupancy bug.

The microbench is the control: the same rank-16 shrink via gmm (cube, batched) is 25x faster than
bgmv (vector, per-row) at prefill row counts. So the cost is the execution strategy, not matrix size.
"Smaller matrices" (N,K) do not move you toward bgmv's favorable regime — only small M does.

## Fix

Give `add_lora_fused_moe` a grouped-matmul path, mirroring the dense `add_shrink`/`add_expand` switch.
The grouping is already computed: `combined_idx = lora_id * num_experts + expert_id` is exactly the
per-row group index a grouped matmul needs (sort rows by `combined_idx` → `group_list` → one
`npu_grouped_matmul` for shrink and one for expand). Expected: the ~4 s of MoE-LoRA bgmv collapses
toward the ~0.2 s scale of the equivalent gmm, i.e. the LoRA overhead drops from ~8x the base to a
small fraction — the same win the dense path already gets.

## Artifacts
- Traces: `traces/lora/prof_lora_v018_b16` (off), `traces/lora_gmm/prof_lora_v018_b16_gmm` (threshold),
  `traces/base/prof_base_v018_b16` (no LoRA). Each rank's `ASCEND_PROFILER_OUTPUT/op_statistic.csv`.
- Scripts: `bench_lora_kernels.py` (E2/E3), `analyze_moe_lora_trace.py` (E1), profile scripts in the vault.
