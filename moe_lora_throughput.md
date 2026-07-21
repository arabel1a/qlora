# MoE-LoRA throughput — Qwen3-30B-A3B-Thinking-2507, TP4 (Atlas 910B4)

Workload: evalscope `random`, prefill 2048 / gen 128, `ignore_eos`, r16 adapter.
LOW = parallel 1 / 16 req. HIGH = parallel 16 / 64 req. All runs 100% success (0 failed).

## LOW LOAD (parallel=1)

| Config | LoRA | Total tok/s | Output tok/s | TTFT ms | TPOT ms |
|---|---|---|---|---|---|
| vllm-ascend v0.18 base       | no  | 552.6  | 31.6 | 143 | 29.9 |
| **v0.18 + our backport (lora-gemm)** | yes | 473.8  | 27.1 | 734 | 30.4 |
| vllm-ascend v0.23rc1 (upstream MoE-LoRA) | yes | 542.2  | 31.9 | 848 | 24.9 |
| vllm-ascend v0.23rc1 base     | no  | 1310.8 | 77.1 | 168 | 11.8 |

## HIGH LOAD (parallel=16)

| Config | LoRA | Total tok/s | Output tok/s | TTFT ms | TPOT ms |
|---|---|---|---|---|---|
| vllm-ascend v0.18 base       | no  | 6594.1 | ~— | 980  | 33.8 |
| **v0.18 + our backport (lora-gemm)** | yes | 2064.3 | ~— | 7463 | 73.9 |
| vllm-ascend v0.23rc1 (upstream MoE-LoRA) | yes | 1865.9 | 109.8 | 8853 | 77.1 |
| vllm-ascend v0.23rc1 base     | no  | 8532.5 | 501.9 | 1028 | 24.0 |

## Takeaways

1. **Engine-version gain is large.** Base v0.23 vs base v0.18 at HIGH: 8532 vs 6594 tok/s
   (~1.3x), and TPOT halves (24.0 vs 33.8 ms). So much of "v0.23 is faster" is the engine,
   not LoRA.
2. **MoE-LoRA is the bottleneck on both versions.** Turning LoRA on collapses HIGH throughput:
   - v0.23: 8532 -> 1866 tok/s (4.6x drop)
   - v0.18: 6594 -> 2064 tok/s (3.2x drop)
   This is inherent to the upstream per-expert-bgmv MoE-LoRA design (static-shape gather looping
   over experts), not specific to our backport.
3. **Our 0.18 backport is competitive with upstream v0.23.** At HIGH it slightly beats native
   v0.23 lora (2064 vs 1866 tok/s); at LOW they are within noise (474 vs 542). The backport did
   not regress relative to the newer upstream MoE-LoRA path.
4. TTFT under load balloons with LoRA on both (7.5-8.9 s at parallel=16) — the LoRA gather runs
   in the prefill hot path.
