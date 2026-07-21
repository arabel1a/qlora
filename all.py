import os
from collections.abc import Callable

import torch
import torch_npu
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm_ascend.lora.utils import refresh_all_lora_classes
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

GMM_TOKEN_THRESHOLD = int(os.environ.get("LORA_GMM_THRESHOLD", "1024"))

# Runtime gate for the gmm path.
#   off       -> gmm disabled, pure bgmv (committed-safe default)
#   threshold -> gmm on when token_nums > GMM_TOKEN_THRESHOLD (the prod regime)
#   force     -> gmm on for every batch (decode too) so it can be tested on short prompts
_GMM_MODE = os.environ.get("LORA_GMM", "off").lower()


class PunicaWrapperNPU(PunicaWrapperBase):
    """
    PunicaWrapperNPU: gmm (npu_grouped_matmul) for prefill, bgmv for decode.

    add_shrink / add_expand call the native torch.ops._C_ascend.add_lora_{shrink,
    expand} ops, which branch gmm-vs-bgmv on a CPU bool flag (use_gmm) and
    short-circuit on a no_lora CPU flag. Being native ops, the choice survives
    torch.compile: they are never traced into nor constant-folded, and the flags
    are read live at runtime from CPU tensors (host read, no device sync ->
    aclgraph-safe).
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int, device: torch.device | str, **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches, device)
        refresh_all_lora_classes()
        self.lora_config = kwargs.get("lora_config")
        if get_ascend_device_type() == AscendDeviceType._310P or (
            self.lora_config is not None and self.lora_config.max_lora_rank >= 128
        ):
            from vllm.lora.ops.torch_ops import (
                bgmv_expand,
                bgmv_expand_slice,
                bgmv_shrink,
                sgmv_expand,
            )
        else:
            from vllm_ascend.lora.lora_ops import (
                bgmv_expand,
                bgmv_expand_slice,
                bgmv_shrink,
                sgmv_expand,
            )
        # bgmv_* serve add_lora_embedding/add_lora_logits/add_lora_fused_moe;
        # sgmv_expand serves add_lora_embedding's prefill path. The Linear
        # shrink/expand kernels live in the native _C_ascend ops.
        self.bgmv_expand = bgmv_expand
        self.bgmv_expand_slice = bgmv_expand_slice
        self.bgmv_shrink = bgmv_shrink
        self.sgmv_expand = sgmv_expand

        # gmm-vs-bgmv switch (set per batch in update_metadata). CPU tensors so
        # the native ops can read them with .item() without a device->host sync.
        self._use_gmm_shrink_cpu = torch.tensor(False, dtype=torch.bool)
        self._use_gmm_expand_cpu = torch.tensor(False, dtype=torch.bool)
        # no-lora short-circuit. Default True (skip) until metadata says a lora
        # is active, mirroring upstream's `if self.no_lora: return` fast path.
        self._no_lora_cpu = torch.tensor(True, dtype=torch.bool)
        # whether compute_meta (prefill_metadata) was computed this step
        self._prefill_meta_ready = False
        # MoE-LoRA single-adapter gmm fast path (see add_lora_fused_moe_gmm).
        # Both are host-only (no device tensor) and set per step in
        # update_metadata: safe to branch on in Python because the gmm path only
        # activates when `enabled` (token_num > threshold), which is the eager
        # prefill regime -- decode (captured by aclgraph) always stays on bgmv.
        self._single_active_lora_id: int | None = None
        self._use_moe_gmm = False

    def update_metadata(self, mapping, lora_index_to_id, max_loras, vocab_size, **kwargs):
        # Base metadata (token_lora_indices) is needed by the bgmv decode path and is
        # cheap (no device->host syncs). compute_meta (prefill_metadata) is needed ONLY
        # by the gmm prefill path and costs TWO syncs (.max().item()+.sum().item()), so
        # we skip it on decode. token_num / no_lora come from the host mapping (no sync).
        self._update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)
        token_num = len(mapping.index_mapping)
        if _GMM_MODE == "force":
            enabled = True
        elif _GMM_MODE == "threshold":
            enabled = token_num > GMM_TOKEN_THRESHOLD
        else:
            enabled = False
        if enabled:
            # prefill/gmm: compute seq_len + lora_indices (+ no_lora) via compute_meta;
            # the 2 syncs are amortized over a large (>threshold) batch.
            self._update_prefill_metadata(self.token_lora_indices)
            no_lora = bool(self.no_lora)
        else:
            # decode/bgmv: skip compute_meta entirely. no_lora from the host mapping
            # (index_mapping uses 0 for no-lora). add_shrink/add_expand still pass
            # prefill_metadata's lora_indices/seq_len but the bgmv branch ignores them.
            no_lora = not any(mapping.index_mapping)
        self._use_gmm_expand_cpu.fill_(enabled)
        self._use_gmm_shrink_cpu.fill_(enabled)
        self._no_lora_cpu.fill_(no_lora)
        # MoE-LoRA gmm fast path is only correct when a single adapter is active
        # in the batch (then every row routed to expert e uses that adapter's
        # (lora, e) weight, so the LoRA groups == the expert groups). Detect the
        # sole active slot host-side: index_mapping holds lora ids (>0), the slot
        # is lora_index_to_id.index(id) -- same convention as token_lora_indices,
        # no device sync. Gate on `enabled` so it never fires in captured decode.
        active_ids = {x for x in mapping.index_mapping if x > 0}
        if len(active_ids) == 1:
            self._single_active_lora_id = lora_index_to_id.index(active_ids.pop())
        else:
            self._single_active_lora_id = None
        self._use_moe_gmm = enabled and self._single_active_lora_id is not None
        # prefill_metadata is only valid when we ran compute_meta (enabled). Consumers
        # that use it (add_lora_embedding's sgmv path) must fall back to bgmv otherwise.
        self._prefill_meta_ready = enabled
        self.is_prefill = bool(getattr(mapping, "is_prefill", True))

    def add_shrink(
        self,
        y: tuple[torch.Tensor, ...] | torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ):
        """
        Semantics:
            y[i] += (x @ lora_a_stacked[i]) * scale

        Dispatched by the native op: gmm needs lora_indices/seq_len, bgmv needs
        token_lora_indices; both are passed, the op reads what its branch uses.
        """
        x = x.view(-1, x.shape[-1])
        y_views = [y[i].view(-1, y[i].shape[-1]) for i in range(len(lora_a_stacked))]
        _, seq_len, lora_indices, _, _, _ = self.prefill_metadata
        torch.ops._C_ascend.add_lora_shrink(
            y_views, x, list(lora_a_stacked),
            lora_indices, seq_len, self.token_lora_indices,
            scale, self._use_gmm_shrink_cpu, self._no_lora_cpu,
        )

    def add_expand(
        self,
        y: torch.Tensor,
        x: tuple[torch.Tensor, ...] | torch.Tensor,
        lora_b_stacked: tuple[torch.Tensor, ...],
        lora_bias_stacked: tuple[torch.Tensor, ...] | None,
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> None:
        """
        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i]
                offset += slice
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        if lora_bias_stacked is not None:
            self._apply_bias(self.token_lora_indices, y, output_slices, lora_bias_stacked)
        _, seq_len, lora_indices, _, _, _ = self.prefill_metadata
        torch.ops._C_ascend.add_lora_expand(
            y, [x[i] for i in range(len(lora_b_stacked))], list(lora_b_stacked),
            lora_indices, seq_len, self.token_lora_indices,
            list(output_slices), offset_start, add_inputs,
            self._use_gmm_expand_cpu, self._no_lora_cpu,
        )
        y = y.view_as(y_org)

    # --- embedding expand: sgmv when prefill metadata exists, else bgmv ---

    def _expand_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_inputs: bool = True,
    ) -> None:
        if self.no_lora:
            return
        self.sgmv_expand(x, w_t_all, y, *self.prefill_metadata, add_inputs)

    def _expand_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_inputs: bool = True,
    ) -> None:
        self.bgmv_expand(x, w_t_all, y, self.token_lora_indices, add_inputs)

    def add_lora_embedding(
        self, y: torch.Tensor, x: torch.Tensor, lora_b_stacked: torch.Tensor, add_inputs: bool = True, **kwargs
    ) -> None:
        """
        Applies lora specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked
        """
        # Use the sgmv prefill path only when prefill_metadata was actually computed
        # this step; on a metadata-stripped decode step fall back to bgmv (per-token).
        expand_fun: Callable = self._expand_prefill if self._prefill_meta_ready else self._expand_decode
        x = x.to(torch.float32)
        expand_fun(y, x, lora_b_stacked, add_inputs)

    def add_lora_fused_moe(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple,
        lora_b_stacked: tuple,
        *,
        topk_weights: torch.Tensor | None = None,
        sorted_token_ids: torch.Tensor | None = None,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor | None = None,
        max_lora_rank: int = 0,
        top_k_num: int = 1,
        shrink_config=None,
        expand_config=None,
        adapter_enabled: torch.Tensor,
        mul_routed_weight: bool = False,
        fully_sharded: bool = False,
        offset: int = 0,
        token_lora_mapping: torch.Tensor | None = None,
    ) -> None:
        """Ascend-native fused MoE LoRA (v2, PR #10977 backport): static-shape
        per-row gather via the same bgmv_shrink/bgmv_expand AscendC kernels used
        by the dense Linear LoRA layers. Each row needs the LoRA slot for
        (lora_id, expert_id); fold both into one gather index into a
        [max_loras * num_experts, ...] view of the per-(lora, expert) weight
        stacks: combined_idx = lora_id * num_experts + expert_id, or -1 when the
        row has no active adapter. bgmv skips negative rows, so inactive rows get
        a zero delta for free. Every tensor's shape depends only on input shapes,
        never on values -> ACL Graph capturable (the torch.unique version was
        not, failing with aclnnUnique2 under enforce_eager=False)."""
        del sorted_token_ids, num_tokens_post_padded, max_lora_rank
        del shrink_config, expand_config, fully_sharded
        assert top_k_num == 1, "Ascend MoE LoRA v1 expects pre-expanded rows (top_k_num=1)."
        if token_lora_mapping is None:
            token_lora_mapping = self.token_lora_indices

        x2d = x.view(-1, x.shape[-1])
        y2d = y.view(-1, y.shape[-1])
        expert_idx = expert_ids.view(-1).to(torch.long)
        num_experts = lora_a_stacked[0].shape[1]

        lora_idx_safe = token_lora_mapping.clamp(min=0)
        enabled = (token_lora_mapping >= 0) & adapter_enabled[lora_idx_safe].bool()
        combined_idx = torch.where(
            enabled,
            lora_idx_safe * num_experts + expert_idx,
            torch.full_like(token_lora_mapping, -1),
        ).contiguous()

        # bgmv_shrink writes fp32 (Y_T); bgmv_expand reads fp32 (X_T).
        rank = lora_a_stacked[0].shape[-2]
        shrink_out = torch.zeros((x2d.shape[0], rank), dtype=torch.float32, device=x2d.device)

        cur_offset = offset
        for slice_idx in range(len(lora_a_stacked)):
            # lora_a/b_stacked[s]: [max_loras, num_experts, rank, *]; flattening
            # the leading two dims turns the (lora, expert) gather into the plain
            # per-row gather bgmv_shrink/bgmv_expand already implement.
            a = lora_a_stacked[slice_idx]
            b = lora_b_stacked[slice_idx]
            out_size = b.shape[-2]
            a_flat = a.view(-1, rank, a.shape[-1])
            b_flat = b.view(-1, out_size, rank)

            self.bgmv_shrink(x2d, a_flat, shrink_out, combined_idx, 1.0)

            delta = shrink_out
            if mul_routed_weight and topk_weights is not None:
                delta = shrink_out * topk_weights.view(-1, 1)

            self.bgmv_expand_slice(delta, b_flat, y2d, combined_idx, cur_offset, out_size, add_inputs=True)
            cur_offset += out_size

    def add_lora_fused_moe_gmm(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple,
        lora_b_stacked: tuple,
        *,
        group_list: torch.Tensor,
        group_list_type: int,
        lora_id: int,
        offset: int = 0,
    ) -> None:
        """Single-adapter grouped-matmul MoE-LoRA (the cube-core path).

        ``x`` is the expert-permuted activation the base expert GMM already
        consumes (``hidden_states`` for w13, the swiglu output for w2) and
        ``group_list`` is the base per-expert token count. Because a single
        adapter is active, every row routed to expert ``e`` uses the same
        ``(lora_id, e)`` weight, so the LoRA groups are exactly the expert groups
        -- no per-row gather, no ``combined_idx``, no routing recovery. One
        ``npu_grouped_matmul`` for shrink and one for expand per slice, on the
        cube cores, reusing the base ``group_list`` verbatim. The delta is added
        into ``y`` before the caller's routed-weight scaling, mirroring the bgmv
        path's insertion point.
        """
        x2d = x.view(-1, x.shape[-1])
        y2d = y.view(-1, y.shape[-1])
        cur = offset
        for s in range(len(lora_a_stacked)):
            a = lora_a_stacked[s][lora_id]        # [num_experts, rank, hidden]
            b = lora_b_stacked[s][lora_id]        # [num_experts, out_s, rank]
            out_s = b.shape[-2]
            # npu_grouped_matmul(group_type=0) wants weight [E, K, N]; the stacks
            # are [E, N, K]. TODO: cache the transpose per (adapter, slice) -- the
            # stacks are updated in place on adapter swap so key on values, not id.
            a_t = a.transpose(1, 2).contiguous()  # [E, hidden, rank]
            b_t = b.transpose(1, 2).contiguous()  # [E, rank, out_s]
            shrink = torch_npu.npu_grouped_matmul(
                x=[x2d], weight=[a_t], split_item=2, group_type=0,
                group_list_type=group_list_type, group_list=group_list,
            )[0]                                  # [M, rank]
            delta = torch_npu.npu_grouped_matmul(
                x=[shrink], weight=[b_t], split_item=2, group_type=0,
                group_list_type=group_list_type, group_list=group_list,
            )[0]                                  # [M, out_s]
            y2d[:, cur:cur + out_s] += delta.to(y2d.dtype)
            cur += out_s

    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        *,
        buffer: tuple[torch.Tensor, ...] | None = None,
        **kwargs,
    ) -> None:
        """
        Applicable to linear-related lora.

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)
        """
        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)
        if buffer is None:
            r = lora_b_stacked[0].size(-1)
            # We set the buffer to be float32 by default, consistent with the
            # triton op
            buffer = tuple(
                torch.zeros((x.size(0), r), dtype=torch.float32, device=x.device)
                for _ in range(len(output_slices))
            )
        self.add_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
        self.add_expand(y, buffer, lora_b_stacked, None, output_slices, add_inputs=True, **kwargs)

    def add_lora_logits(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        scale,
        *,
        buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """
        Applies lora specifically for LogitsProcessorWithLoRA.

        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)
        if buffer is None:
            buffer = torch.zeros((x.size(0), r), dtype=torch.float32, device=x.device)
        indices = self.sampler_indices
        self.bgmv_shrink(x, lora_a_stacked, buffer, indices, scale)
        self.bgmv_expand(buffer, lora_b_stacked, y, indices, add_inputs=True)
        y = y.view_as(y_org)
