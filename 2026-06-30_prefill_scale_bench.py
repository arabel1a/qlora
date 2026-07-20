"""Kernel-level repro of the 16-parallel PREFILL slowdown (gmm vs sgmv vs bgmv).

At 16 parallel requests with ONE adapter, the scheduler packs a chunked-prefill
step of up to max_num_batched_tokens=32768 tokens, all using the same LoRA ->
compute_meta collapses them to 1 group. So the gmm path is a single
npu_grouped_matmul over T (up to 32768) tokens, plus a per-call weight
gather/transpose/contiguous and an fp32 expand. sgmv runs its segmented GEMM
over the same T. This sweeps T and reports per-module + per-layer totals so we
can see WHERE (and at what T) gmm crosses sgmv, and whether the kernel-level
ratio matches the ~2x end-to-end prefill (TTFT) regression observed at 16/128.

Qwen3-32B TP4 per-rank shapes (vllm merges qkv & gate_up); adapter rank=32.

Run: docker exec va18_misha bash -lc \
  "cd /home/russia_mmo/misha/qlora && python 2026-06-30_prefill_scale_bench.py"
Env: NUM_LORAS (default 1), DEVICE (default npu:7).
"""
import os
import time
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # noqa
import vllm_ascend.meta_registration  # noqa
from vllm.lora.punica_wrapper.utils import compute_meta
from vllm_ascend.lora.lora_ops import (
    bgmv_shrink, bgmv_expand_slice, sgmv_shrink, sgmv_expand_slice,
)

DEVICE = os.environ.get("DEVICE", "npu:7")
RANK = 32
NUM_LORAS = int(os.environ.get("NUM_LORAS", "1"))
DTYPE = torch.bfloat16
NLAYERS = 64
WARMUP, ITERS = 20, 50

# (name, shrink_K, [expand slice sizes])  -- TP4 per-rank
MODULES = [
    ("qkv_proj",     5120, [2048, 256, 256]),
    ("o_proj",       2048, [5120]),
    ("gate_up_proj", 5120, [6400, 6400]),
    ("down_proj",    6400, [5120]),
]

# Prefill step sizes seen at 16 parallel (chunked-prefill caps at 32768).
T_SWEEP = [2048, 4096, 8192, 16384, 24576, 32768]


def gather_gmm(w, lora_idx):
    """Mirror all.py _gather_weights_for_gmm exactly (per-call cost included)."""
    if w.ndim == 4:
        w = w.squeeze(1)
    g = w[lora_idx.clamp(min=0)].transpose(1, 2).contiguous()
    return g.masked_fill((lora_idx < 0).unsqueeze(1).unsqueeze(2), 0.0)


def bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e6  # us


def gen_token_lora(T, L, reqs, dev):
    """Contiguous per-sequence lora ids (like a real prefill batch of `reqs` seqs)."""
    per = T // reqs
    out = []
    for i in range(reqs):
        out += [i % L] * per
    out = out[:T] + [0] * (T - len(out[:T]))
    return torch.tensor(out, dtype=torch.long, device=dev)


def run_T(T, reqs, dev):
    tl = gen_token_lora(T, NUM_LORAS, reqs, dev)
    bss, sl, li, bs, ml, tn, nl = compute_meta(tl)
    ngroups = int((li >= 0).sum().item())
    print(f"\n{'='*86}\nPREFILL T={T:6d}  seqs={reqs:2d}  loras={NUM_LORAS}  "
          f"-> compute_meta groups={ngroups}  (group_list={sl.tolist()[:6]}...)\n{'='*86}")

    tot = {"bgmv": 0.0, "sgmv": 0.0, "gmm": 0.0, "gather": 0.0}
    for name, K, slices in MODULES:
        torch.manual_seed(0)
        x = torch.randn(T, K, dtype=DTYPE, device=dev).contiguous()
        A = torch.randn(NUM_LORAS, RANK, K, dtype=DTYPE, device=dev).contiguous()
        ys = torch.zeros(T, RANK, dtype=torch.float32, device=dev)
        buf = torch.randn(T, RANK, dtype=torch.float32, device=dev).contiguous()
        Bw = [torch.randn(NUM_LORAS, n, RANK, dtype=DTYPE, device=dev).contiguous() for n in slices]
        ye = torch.zeros(T, sum(slices), dtype=DTYPE, device=dev)

        # ---- SHRINK ----
        t_sgmv_s = bench(lambda: sgmv_shrink(x, A, ys, bss, sl, li, bs, ml, tn, 1.0))
        t_bgmv_s = bench(lambda: bgmv_shrink(x, A, ys, tl, 1.0))
        t_gather_s = bench(lambda: gather_gmm(A, li))

        def f_gmm_s():
            gw = gather_gmm(A, li)
            xin = x.to(gw.dtype)
            torch_npu.npu_grouped_matmul(x=[xin], weight=[gw], split_item=2,
                                         group_list_type=1, group_type=0, group_list=sl)[0]
        t_gmm_s = bench(f_gmm_s)

        # ---- EXPAND (loop slices; gmm/expand run in fp32 like all.py) ----
        def f_sgmv_e():
            off = 0
            for j, n in enumerate(slices):
                sgmv_expand_slice(buf, Bw[j], ye, bss, sl, li, bs, ml, tn, off, n, True)
                off += n

        def f_bgmv_e():
            off = 0
            for j, n in enumerate(slices):
                bgmv_expand_slice(buf, Bw[j], ye, tl, off, n, True)
                off += n

        def f_gmm_e():
            for j, n in enumerate(slices):
                gw = gather_gmm(Bw[j], li).to(buf.dtype)
                torch_npu.npu_grouped_matmul(x=[buf], weight=[gw], split_item=2,
                                             group_list_type=1, group_type=0, group_list=sl)[0]

        t_sgmv_e = bench(f_sgmv_e)
        t_bgmv_e = bench(f_bgmv_e)
        t_gmm_e = bench(f_gmm_e)

        m_bgmv, m_sgmv, m_gmm = t_bgmv_s + t_bgmv_e, t_sgmv_s + t_sgmv_e, t_gmm_s + t_gmm_e
        tot["bgmv"] += m_bgmv; tot["sgmv"] += m_sgmv; tot["gmm"] += m_gmm
        tot["gather"] += t_gather_s * (1 + len(slices))  # gather per shrink + per expand-slice
        print(f"  {name:13s} K={K:5d} sl={str(slices):18s} "
              f"shrink[sgmv {t_sgmv_s:7.1f} gmm {t_gmm_s:7.1f} (gather {t_gather_s:5.1f})]  "
              f"expand[sgmv {t_sgmv_e:7.1f} gmm {t_gmm_e:7.1f}]  bgmv={m_bgmv:7.1f}")

    r = tot["gmm"] / tot["sgmv"] if tot["sgmv"] else float("nan")
    print(f"  ---- per-LAYER (us): sgmv={tot['sgmv']:8.1f}  gmm={tot['gmm']:8.1f}  "
          f"bgmv={tot['bgmv']:8.1f}  | gmm/sgmv={r:.2f}x  (gather share of gmm: "
          f"{100*tot['gather']/tot['gmm']:.0f}%)")
    print(f"  ---- whole-model x{NLAYERS} (ms): sgmv={tot['sgmv']*NLAYERS/1e3:6.2f}  "
          f"gmm={tot['gmm']*NLAYERS/1e3:6.2f}  bgmv={tot['bgmv']*NLAYERS/1e3:6.2f}")
    return r


def main():
    torch.npu.set_device(DEVICE)
    dev = torch.device(DEVICE)
    print(f"torch={torch.__version__} torch_npu={torch_npu.__version__} dev={DEVICE} "
          f"| Qwen3-32B TP4 rank={RANK} loras={NUM_LORAS}")
    ratios = {}
    for T in T_SWEEP:
        # 16-parallel prefill: pack up to 16 seqs into the step (capped by T)
        reqs = min(16, max(1, T // 2048))
        ratios[T] = run_T(T, reqs, dev)
    print(f"\n{'#'*86}\nSUMMARY gmm/sgmv per-layer ratio vs T (loras={NUM_LORAS}):")
    for T in T_SWEEP:
        bar = "#" * int(ratios[T] * 20)
        print(f"  T={T:6d}:  {ratios[T]:.2f}x  {bar}")
    print(f"{'#'*86}")


if __name__ == "__main__":
    main()
