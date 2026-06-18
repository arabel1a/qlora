# Fused LoRA — gmm prefill / bgmv decode (vllm-ascend)

**Problem.** In vLLM V1 `is_prefill` is always `True`, so the upstream `PunicaWrapperNPU`
runs **sgmv for every batch**. sgmv is the segmented-GEMM kernel; on long-prefill
workloads it dominates latency. Driving `npu_grouped_matmul` (gmm) for prefill from a
Python `torch.library` opaque op fixed prefill but regressed decode (per-call Python
wrapper tax + `compute_meta` device→host syncs every step).

**Change.** One native op `torch.ops._C_ascend.add_lora_{shrink,expand}` that picks the
kernel **in C++** per batch:

- `token_num > THRESHOLD` (prefill) → **gmm** (`aclnnGroupedMatmulV4`)
- else (decode) → **bgmv** (per-token weight indexing in-kernel — no gather, no `compute_meta`)

The branch is a compiled `if` on a **CPU bool tensor** (`use_gmm`), so it survives
`torch.compile` (not constant-folded) and the op is called once per `add_*` like any
native kernel — no Python dispatch/sync overhead.

### `punica_npu.py` (the only part you touch)

```python
def add_shrink(self, y, x, lora_a_stacked, scale, **kwargs):
    x = x.view(-1, x.shape[-1])
    y_views = [yi.view(-1, yi.shape[-1]) for yi in y]
    _, seq_len, lora_indices, *_ = self.prefill_metadata
    torch.ops._C_ascend.add_lora_shrink(
        y_views, x, list(lora_a_stacked),
        lora_indices, seq_len, self.token_lora_indices,   # gmm needs seq_len/lora_indices; bgmv needs token_lora_indices
        scale, self._use_gmm_shrink_cpu, self._no_lora_cpu,
    )
```

Gating, set once per step — **decode skips `compute_meta` entirely** (its `.max().item()`
+ `.sum().item()` syncs):

```python
def update_metadata(self, mapping, lora_index_to_id, max_loras, vocab_size, **kw):
    self._update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)  # -> token_lora_indices (cheap)
    enabled = len(mapping.index_mapping) > GMM_TOKEN_THRESHOLD                    # gmm only for big batches
    if enabled:
        self._update_prefill_metadata(self.token_lora_indices)                   # compute_meta (2 syncs) — prefill only
        no_lora = self.no_lora
    else:
        no_lora = not any(mapping.index_mapping)                                  # decode: no syncs
    self._use_gmm_shrink_cpu.fill_(enabled); self._use_gmm_expand_cpu.fill_(enabled)
    self._no_lora_cpu.fill_(no_lora)
```

Op interface (C++ internals omitted):

```
add_lora_shrink(Tensor(a!)[] y, Tensor x, Tensor[] lora_a, Tensor lora_indices,
                Tensor seq_len, Tensor token_lora_indices, float scale,
                Tensor use_gmm, Tensor no_lora) -> ()
```

### Build / deploy

```bash
./build_op.sh           # rebuild vllm_ascend_C + deploy punica_npu.py   (revert: ./build_op.sh revert)
```

### Results — Qwen3-32B TP4, lora-adapter1, input 2–7K (avg 4385), **output 204**, 80 prompts @ parallel 8

| metric | stock (sgmv) | ours (gmm/bgmv) | Δ |
|---|---|---|---|
| Total test time | 290.2 s | **210.4 s** | **−27 %** |
| TTFT p50 | 14.12 s | **7.06 s** | **−50 %** |
| TTFT p99 | 17.59 s | **8.88 s** | **−50 %** |
| TPOT p50 | 71.8 ms | **68.0 ms** | −5 % |
| Output throughput | 56.2 tok/s | **77.6 tok/s** | **+38 %** |
| Total throughput | 1265 tok/s | **1745 tok/s** | **+38 %** |

Both 80/80 success, identical inputs (same seed). The win is gmm halving prefill TTFT on
these long inputs; decode TPOT also improves slightly (bgmv + no `compute_meta` syncs + no
Python-op tax).
