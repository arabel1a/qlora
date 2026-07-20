"""Kernel shapes for the EMBED + LM_HEAD LoRA modules (the fsdp-rank32 adapter on
the OTHER server). These are the modules the blocks-only adapter lacks, and the
prime suspect for why ours regresses there but not here.

Qwen3-32B TP4, rank=32, vocab=151936 -> vocab_shard=37984, hidden=5120.

  module        shrink                         expand                       T (prefill)
  embed_tokens  lookup A[vocab,rank]->[T,r]    [T,r]@B[hidden,r]->[T,5120]  FULL batch (<=32768)
  lm_head       x[S,5120]@A[r,5120]->[S,r]     [S,r]@B[37984,r]->[S,37984]  S = num_seqs (<=16)

For embed we sweep T (it runs over the whole prefill batch); for lm_head T=num_seqs.
Compare sgmv vs bgmv vs gmm so we can see whether THESE shapes (esp. the 37984-wide
lm_head expand and the full-batch embed expand) are where gmm/sgmv diverge.

Run on a FREE device (A/B holds 0-3): DEVICE=npu:7 python 2026-06-30_embed_lmhead_bench.py
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
HIDDEN = 5120
VOCAB_SHARD = 37984
WARMUP, ITERS = 20, 50


def gather_gmm(w, lora_idx):
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
    return (time.perf_counter() - t0) / ITERS * 1e6


def gen_token_lora(T, L, reqs, dev):
    per = T // reqs
    out = []
    for i in range(reqs):
        out += [i % L] * per
    out = out[:T] + [0] * (T - len(out[:T]))
    return torch.tensor(out, dtype=torch.long, device=dev)


def bench_expand(label, T, reqs, out_dim, dev):
    """Expand [T,rank] -> [T,out_dim] across sgmv/bgmv/gmm (B = [L, out_dim, rank])."""
    tl = gen_token_lora(T, NUM_LORAS, reqs, dev)
    bss, sl, li, bs, ml, tn, nl = compute_meta(tl)
    torch.manual_seed(0)
    buf = torch.randn(T, RANK, dtype=torch.float32, device=dev).contiguous()
    B = torch.randn(NUM_LORAS, out_dim, RANK, dtype=DTYPE, device=dev).contiguous()
    ye = torch.zeros(T, out_dim, dtype=DTYPE, device=dev)

    t_sgmv = bench(lambda: sgmv_expand_slice(buf, B, ye, bss, sl, li, bs, ml, tn, 0, out_dim, True))
    t_bgmv = bench(lambda: bgmv_expand_slice(buf, B, ye, tl, 0, out_dim, True))

    def f_gmm():
        gw = gather_gmm(B, li).to(buf.dtype)
        torch_npu.npu_grouped_matmul(x=[buf], weight=[gw], split_item=2,
                                     group_list_type=1, group_type=0, group_list=sl)[0]
    t_gmm = bench(f_gmm)
    win = min(t_sgmv, t_bgmv, t_gmm)
    tag = {t_sgmv: "sgmv", t_bgmv: "bgmv", t_gmm: "gmm"}[win]
    print(f"  {label:26s} T={T:6d} out={out_dim:6d}  "
          f"sgmv {t_sgmv:8.1f}  bgmv {t_bgmv:8.1f}  gmm {t_gmm:8.1f}  us   -> best={tag}")
    return t_sgmv, t_bgmv, t_gmm


def bench_lmhead_shrink(S, dev):
    tl = gen_token_lora(S, NUM_LORAS, S, dev)
    bss, sl, li, bs, ml, tn, nl = compute_meta(tl)
    x = torch.randn(S, HIDDEN, dtype=DTYPE, device=dev).contiguous()
    A = torch.randn(NUM_LORAS, RANK, HIDDEN, dtype=DTYPE, device=dev).contiguous()
    ys = torch.zeros(S, RANK, dtype=torch.float32, device=dev)
    t_sgmv = bench(lambda: sgmv_shrink(x, A, ys, bss, sl, li, bs, ml, tn, 1.0))
    t_bgmv = bench(lambda: bgmv_shrink(x, A, ys, tl, 1.0))
    print(f"  lm_head shrink            S={S:6d} out={RANK:6d}  "
          f"sgmv {t_sgmv:8.1f}  bgmv {t_bgmv:8.1f}  (gmm n/a, S tiny)")


def embed_lookup_cost(T, dev):
    """Embedding shrink = index_select of A[vocab,rank] by T token ids (not a gemm)."""
    A = torch.randn(VOCAB_SHARD, RANK, dtype=DTYPE, device=dev).contiguous()
    ids = torch.randint(0, VOCAB_SHARD, (T,), device=dev)
    t = bench(lambda: torch.index_select(A, 0, ids))
    print(f"  embed lookup (index_select) T={T:6d}                 {t:8.1f} us")


def main():
    torch.npu.set_device(DEVICE)
    dev = torch.device(DEVICE)
    print(f"dev={DEVICE} | Qwen3-32B TP4 rank={RANK} loras={NUM_LORAS} | vocab_shard={VOCAB_SHARD}")

    print("\n############ EMBED_TOKENS (expand runs over FULL prefill batch T) ############")
    for T in (2048, 8192, 16384, 32768):
        reqs = min(16, max(1, T // 2048))
        embed_lookup_cost(T, dev)
        bench_expand("embed expand ->hidden", T, reqs, HIDDEN, dev)

    print("\n############ LM_HEAD (T = num_seqs, expand is 37984-wide) ############")
    for S in (8, 16):
        bench_lmhead_shrink(S, dev)
        bench_expand("lm_head expand ->vocab", S, S, VOCAB_SHARD, dev)

    print("\n############ context: for comparison, lm_head expand if it ran over FULL T ############")
    print("(it does NOT in vllm — logits are last-token-only — but shows the wide-output scaling)")
    for T in (2048, 8192):
        reqs = min(16, max(1, T // 2048))
        bench_expand("lm_head expand @fullT", T, reqs, VOCAB_SHARD, dev)


if __name__ == "__main__":
    main()
