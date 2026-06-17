import os
from collections.abc import Callable
from typing import List

import torch
import torch_npu
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm_ascend.lora.utils import refresh_all_lora_classes
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

# Back-compat shim: test_e2e.py references this. No longer used as a cache —
# see _gather_weights_for_gmm for why caching the transpose was unsafe.
_TRANSPOSED_WEIGHT_CACHE: dict[int, torch.Tensor] = {}
_SGMV_SHRINK_FN = None
_SGMV_EXPAND_SLICE_FN = None
# Decode path uses bgmv (per-token in-kernel weight indexing): == sgmv speed but needs
# no compute_meta (only token_lora_indices) and no weight gather. See FINAL design.
_BGMV_SHRINK_FN = None
_BGMV_EXPAND_SLICE_FN = None


def _gather_weights_for_gmm(w: torch.Tensor, lora_indices: torch.Tensor) -> torch.Tensor:
    # IMPORTANT: do not cache the transposed weight by w.data_ptr(). The stacked
    # LoRA weight buffers are allocated up front (zeros) and the adapter weights
    # are copied into the *same* storage later. A data_ptr-keyed cache populated
    # during the zero-init / warmup pass returns a STALE all-zeros transpose once
    # the real adapter loads -> the gmm LoRA delta becomes exactly 0 (output ==
    # base). sgmv reads the live buffer and is unaffected, which is why only the
    # gmm path silently dropped the LoRA.
    #
    # Gather by index first (per-group, small), then transpose the gathered
    # result. Gathering on dim 0 commutes with transpose(1, 2), so this is
    # identical to transpose-then-gather but avoids transposing the full stacked
    # weight and reads the live buffer every call.
    if w.ndim == 4:
        w = w.squeeze(1)                              # [num_slots, rank, hidden]
    safe_indices = lora_indices.clamp(min=0)
    gathered = w[safe_indices]                        # [num_groups, rank, hidden]
    gathered = gathered.transpose(1, 2).contiguous()  # [num_groups, hidden, rank]
    inactive = (lora_indices < 0).unsqueeze(1).unsqueeze(2)
    gathered = gathered.masked_fill(inactive, 0.0)
    return gathered


# --- custom ops: each wraps a WHOLE add_* call as one opaque operator ---
#
# Why one combined op per add_* instead of the old per-slice dual-launch of
# separate gmm_* and sgmv_* ops: under torch.compile the custom-op body is
# OPAQUE. Inductor never traces into it, so it cannot constant-fold the kernel
# choice away nor reorder/fuse the two kernels, and the slice loop + the
# gmm-vs-sgmv branch run as live Python at runtime. The switch is driven by two
# CPU bool tensors passed as op inputs:
#   use_gmm  -> gmm (npu_grouped_matmul, prefill) vs sgmv (decode)
#   no_lora  -> short-circuit when no adapter is active in the batch
# Reading them with .item() is a host-side load (no device->host sync, so it is
# legal under ACL-graph capture) and, being tensor inputs rather than Python
# constants, their values are NOT baked into the FX graph at trace time. A
# single op also mutates the output buffer exactly once, instead of two ops
# aliasing the same buffer (which functionalization handles poorly).

@torch.library.custom_op("lora::add_shrink", mutates_args=("y",))
def _add_shrink_op(
    y: List[torch.Tensor],
    x: torch.Tensor,
    lora_a_stacked: List[torch.Tensor],
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batch_size: int,
    max_length: int,
    token_nums: int,
    scaling: float,
    token_lora_indices: torch.Tensor,
    use_gmm: torch.Tensor,
    no_lora: torch.Tensor,
) -> None:
    if no_lora.item():
        return
    if use_gmm.item():
        for slice_idx in range(len(lora_a_stacked)):
            out = y[slice_idx]
            gathered_w = _gather_weights_for_gmm(
                lora_a_stacked[slice_idx], lora_indices_tensor)
            x_in = x if x.dtype == gathered_w.dtype else x.to(gathered_w.dtype)
            result = torch_npu.npu_grouped_matmul(
                x=[x_in], weight=[gathered_w],
                split_item=2, group_list_type=1, group_type=0,
                group_list=seq_len_tensor,
            )[0]
            if scaling != 1.0:
                result = result * scaling
            out.add_(result.to(out.dtype))
    else:
        # decode: bgmv indexes the stacked weight per-token in-kernel (no gather,
        # no per-group metadata) -> uses token_lora_indices, not seq/group meta.
        for slice_idx in range(len(lora_a_stacked)):
            _BGMV_SHRINK_FN(
                x, lora_a_stacked[slice_idx], y[slice_idx],
                token_lora_indices, scaling,
            )


@_add_shrink_op.register_fake
def _add_shrink_fake(y, x, lora_a_stacked, b_seq_start_loc, seq_len_tensor,
                     lora_indices_tensor, batch_size, max_length, token_nums,
                     scaling, token_lora_indices, use_gmm, no_lora):
    return None


@torch.library.custom_op("lora::add_expand", mutates_args=("y",))
def _add_expand_op(
    y: torch.Tensor,
    x: List[torch.Tensor],
    lora_b_stacked: List[torch.Tensor],
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batch_size: int,
    max_length: int,
    token_nums: int,
    output_slices: List[int],
    offset_start: int,
    add_inputs: bool,
    token_lora_indices: torch.Tensor,
    use_gmm: torch.Tensor,
    no_lora: torch.Tensor,
) -> None:
    if no_lora.item():
        return
    offset_left = offset_start
    if use_gmm.item():
        for slice_idx in range(len(lora_b_stacked)):
            xi = x[slice_idx]
            size = output_slices[slice_idx]
            gathered_w = _gather_weights_for_gmm(
                lora_b_stacked[slice_idx], lora_indices_tensor)
            # Cast weights up to match x (fp32 from shrink), not x down to bf16
            w_in = gathered_w if gathered_w.dtype == xi.dtype else gathered_w.to(xi.dtype)
            result = torch_npu.npu_grouped_matmul(
                x=[xi], weight=[w_in],
                split_item=2, group_list_type=1, group_type=0,
                group_list=seq_len_tensor,
            )[0]
            target = y[:, offset_left:offset_left + size]
            if add_inputs:
                target.add_(result.to(target.dtype))
            else:
                target.copy_(result.to(target.dtype))
            offset_left += size
    else:
        # decode: bgmv expand_slice, per-token weight indexing.
        for slice_idx in range(len(lora_b_stacked)):
            size = output_slices[slice_idx]
            _BGMV_EXPAND_SLICE_FN(
                x[slice_idx], lora_b_stacked[slice_idx], y,
                token_lora_indices, offset_left, size, add_inputs,
            )
            offset_left += size


@_add_expand_op.register_fake
def _add_expand_fake(y, x, lora_b_stacked, b_seq_start_loc, seq_len_tensor,
                     lora_indices_tensor, batch_size, max_length, token_nums,
                     output_slices, offset_start, add_inputs, token_lora_indices,
                     use_gmm, no_lora):
    return None


GMM_TOKEN_THRESHOLD = 1024

# Runtime gate for the gmm path.
#   off       -> gmm disabled, pure sgmv (committed-safe default, =97% acc)
#   threshold -> gmm on when token_nums > GMM_TOKEN_THRESHOLD (the prod regime)
#   force     -> gmm on for every batch (decode too) so it can be tested on short prompts
_GMM_MODE = os.environ.get("LORA_GMM", "off").lower()


class PunicaWrapperNPU(PunicaWrapperBase):
    """
    PunicaWrapperNPU: gmm (npu_grouped_matmul) for prefill, sgmv for decode.

    add_shrink / add_expand are each wrapped in a SINGLE opaque custom op that
    branches gmm-vs-sgmv on a CPU bool flag (use_gmm) and short-circuits on a
    no_lora CPU flag. Being opaque, the choice survives torch.compile: the op is
    never traced into nor constant-folded, and the flags are read live at
    runtime from CPU tensors (host read, no device sync -> aclgraph-safe).
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
                sgmv_expand_slice,
                sgmv_shrink,
            )
        else:
            from vllm_ascend.lora.lora_ops import (
                bgmv_expand,
                bgmv_expand_slice,
                bgmv_shrink,
                sgmv_expand,
                sgmv_expand_slice,
                sgmv_shrink,
            )
        self.bgmv_expand = bgmv_expand
        self.bgmv_expand_slice = bgmv_expand_slice
        self.bgmv_shrink = bgmv_shrink
        self.sgmv_expand = sgmv_expand
        self.sgmv_expand_slice = sgmv_expand_slice
        self.sgmv_shrink = sgmv_shrink

        global _SGMV_SHRINK_FN, _SGMV_EXPAND_SLICE_FN
        global _BGMV_SHRINK_FN, _BGMV_EXPAND_SLICE_FN
        _SGMV_SHRINK_FN = sgmv_shrink
        _SGMV_EXPAND_SLICE_FN = sgmv_expand_slice
        _BGMV_SHRINK_FN = bgmv_shrink
        _BGMV_EXPAND_SLICE_FN = bgmv_expand_slice

        # gmm-vs-sgmv switch (set per batch in update_metadata). CPU tensors so
        # the opaque ops can read them with .item() without a device->host sync.
        self._use_gmm_shrink_cpu = torch.tensor(False, dtype=torch.bool)
        self._use_gmm_expand_cpu = torch.tensor(False, dtype=torch.bool)
        # no-lora short-circuit. Default True (skip) until metadata says a lora
        # is active, mirroring upstream's `if self.no_lora: return` fast path.
        self._no_lora_cpu = torch.tensor(True, dtype=torch.bool)
        # whether compute_meta (prefill_metadata) was computed this step
        self._prefill_meta_ready = False

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

    # --- kept for add_lora_embedding / add_lora_logits ---

    def _expand_prefill(self, y, x, w_t_all, add_inputs):
        if self.no_lora:
            return
        self.sgmv_expand(x, w_t_all, y, *self.prefill_metadata, add_inputs)

    def _expand_decode(self, y, x, w_t_all, add_inputs):
        self.bgmv_expand(x, w_t_all, y, self.token_lora_indices, add_inputs)

    def add_lora_embedding(self, y, x, lora_b_stacked, add_inputs=True, **kwargs):
        # Use the sgmv prefill path only when prefill_metadata was actually computed
        # this step; on a metadata-stripped decode step fall back to bgmv (per-token).
        expand_fun: Callable = self._expand_prefill if self._prefill_meta_ready else self._expand_decode
        x = x.to(torch.float32)
        expand_fun(y, x, lora_b_stacked, add_inputs)

    def add_lora_linear(self, y, x, lora_a_stacked, lora_b_stacked, scale,
                        output_slices, *, buffer=None, **kwargs):
        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)
        if buffer is None:
            r = lora_b_stacked[0].size(-1)
            buffer = tuple(
                torch.zeros((x.size(0), r), dtype=torch.float32, device=x.device)
                for _ in range(len(output_slices))
            )
        self.add_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
        self.add_expand(y, buffer, lora_b_stacked, None, output_slices, add_inputs=True, **kwargs)

    def add_lora_logits(self, y, x, lora_a_stacked, lora_b_stacked, scale,
                        *, buffer=None, **kwargs):
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
