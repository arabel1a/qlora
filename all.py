import os
from collections.abc import Callable
from typing import List

import torch
import torch_npu
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm_ascend.lora.utils import refresh_all_lora_classes
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

print("\nhui"*10)

# --- TEMP debug trace (compile/decode routing investigation) ---
_TRACE = os.environ.get("LORA_TRACE", "") != ""
_TRACE_PATH = os.environ.get("LORA_TRACE_PATH", "/tmp/lora_trace.log")
_trace_n = [0]


def _trace(tag: str, **kv):
    if not _TRACE:
        return
    _trace_n[0] += 1
    parts = " ".join(f"{k}={v}" for k, v in kv.items())
    try:
        with open(_TRACE_PATH, "a") as f:
            f.write(f"[{_trace_n[0]:05d}] {tag} {parts}\n")
    except Exception:
        pass


# Back-compat shim: test_e2e.py references this. No longer used as a cache —
# see _gather_weights_for_gmm for why caching the transpose was unsafe.
_TRANSPOSED_WEIGHT_CACHE: dict[int, torch.Tensor] = {}
_SGMV_SHRINK_FN = None
_SGMV_EXPAND_SLICE_FN = None


def _sanitize_group_list(seq_len_tensor: torch.Tensor, num_rows: int) -> torch.Tensor:
    # Under torch.compile the punica metadata slices are baked at trace time
    # (sliced by the trace-time batch_size, e.g. 256 from the warmup run), so at
    # runtime seq_len_tensor carries STALE counts past the live batch: e.g.
    # [2019, 32, 32, ...] for a single 2019-token prefill. The ascend sgmv
    # kernel tolerates the garbage tail; npu_grouped_matmul does not —
    # sum(group_list) overshoots the actual row count and the kernel reads/
    # writes out of bounds -> garbage output. Zero every group whose cumulative
    # count exceeds the real number of input rows. Pure tensor ops, no
    # device->host sync, so this is also legal under aclgraph capture.
    csum = torch.cumsum(seq_len_tensor, dim=0)
    valid = csum <= num_rows
    return seq_len_tensor * valid


def _stream_is_capturing() -> bool:
    # npu_grouped_matmul is not aclgraph-safe (data-dependent group_list; crashes
    # or corrupts under capture/replay). If this op body runs while the current
    # stream is being captured into an ACL graph, the graph must record the sgmv
    # kernels instead — sgmv under capture is verified correct.
    try:
        return torch.npu.is_current_stream_capturing()
    except (AttributeError, RuntimeError):
        return False


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


# --- custom ops: opaque to torch.compile, safe to branch inside ---

@torch.library.custom_op("lora::gmm_shrink", mutates_args=("output_tensor",))
def _gmm_shrink_op(
    inputs: torch.Tensor,
    lora_a_weight: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batch_size: int,
    max_length: int,
    token_nums: int,
    scaling: float,
    no_lora: torch.Tensor,
) -> None:
    _trace("gmm_shrink", no_lora=no_lora.item(), tok=int(inputs.shape[0]),
           tn=token_nums, cap=_stream_is_capturing(),
           sl=tuple(seq_len_tensor.shape))
    if no_lora.item():
        return
    if _stream_is_capturing():
        _SGMV_SHRINK_FN(
            inputs, lora_a_weight, output_tensor,
            b_seq_start_loc, seq_len_tensor, lora_indices_tensor,
            batch_size, max_length, token_nums, scaling,
        )
        return
    gathered_w = _gather_weights_for_gmm(lora_a_weight, lora_indices_tensor)
    x_in = inputs if inputs.dtype == gathered_w.dtype else inputs.to(gathered_w.dtype)
    group_list = _sanitize_group_list(seq_len_tensor, inputs.shape[0])
    result = torch_npu.npu_grouped_matmul(
        x=[x_in], weight=[gathered_w],
        split_item=2, group_list_type=1, group_type=0,
        group_list=group_list,
    )[0]
    if scaling != 1.0:
        result = result * scaling
    output_tensor.add_(result.to(output_tensor.dtype))


@_gmm_shrink_op.register_fake
def _gmm_shrink_fake(inputs, lora_a_weight, output_tensor,
                     b_seq_start_loc, seq_len_tensor, lora_indices_tensor,
                     batch_size, max_length, token_nums, scaling, no_lora):
    return None


@torch.library.custom_op("lora::sgmv_shrink", mutates_args=("output_tensor",))
def _sgmv_shrink_op(
    inputs: torch.Tensor,
    lora_a_weight: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batch_size: int,
    max_length: int,
    token_nums: int,
    scaling: float,
    no_lora: torch.Tensor,
) -> None:
    _trace("sgmv_shrink", no_lora=no_lora.item(), tok=int(inputs.shape[0]),
           tn=token_nums)
    if no_lora.item():
        return
    _SGMV_SHRINK_FN(
        inputs, lora_a_weight, output_tensor,
        b_seq_start_loc, seq_len_tensor, lora_indices_tensor,
        batch_size, max_length, token_nums, scaling,
    )


@_sgmv_shrink_op.register_fake
def _sgmv_shrink_fake(inputs, lora_a_weight, output_tensor,
                      b_seq_start_loc, seq_len_tensor, lora_indices_tensor,
                      batch_size, max_length, token_nums, scaling, no_lora):
    return None


@torch.library.custom_op("lora::gmm_expand_slice", mutates_args=("y",))
def _gmm_expand_slice_op(
    y: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batch_size: int,
    max_length: int,
    token_nums: int,
    y_offset: int,
    y_slice_size: int,
    add_inputs: bool,
    no_lora: torch.Tensor,
) -> None:
    _trace("gmm_expand", no_lora=no_lora.item(), tok=int(x.shape[0]),
           tn=token_nums, cap=_stream_is_capturing(),
           sl=tuple(seq_len_tensor.shape))
    if no_lora.item():
        return
    if _stream_is_capturing():
        _SGMV_EXPAND_SLICE_FN(
            x, w, y,
            b_seq_start_loc, seq_len_tensor, lora_indices_tensor,
            batch_size, max_length, token_nums,
            y_offset, y_slice_size, add_inputs,
        )
        return
    gathered_w = _gather_weights_for_gmm(w, lora_indices_tensor)
    # Cast weights up to match x (fp32 from shrink), not x down to bf16
    w_in = gathered_w if gathered_w.dtype == x.dtype else gathered_w.to(x.dtype)
    group_list = _sanitize_group_list(seq_len_tensor, x.shape[0])
    result = torch_npu.npu_grouped_matmul(
        x=[x], weight=[w_in],
        split_item=2, group_list_type=1, group_type=0,
        group_list=group_list,
    )[0]
    target = y[:, y_offset:y_offset + y_slice_size]
    if add_inputs:
        target.add_(result.to(target.dtype))
    else:
        target.copy_(result.to(target.dtype))


@_gmm_expand_slice_op.register_fake
def _gmm_expand_slice_fake(y, x, w, b_seq_start_loc, seq_len_tensor,
                           lora_indices_tensor, batch_size, max_length,
                           token_nums, y_offset, y_slice_size, add_inputs, no_lora):
    return None


@torch.library.custom_op("lora::sgmv_expand_slice", mutates_args=("y",))
def _sgmv_expand_slice_op(
    y: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batch_size: int,
    max_length: int,
    token_nums: int,
    y_offset: int,
    y_slice_size: int,
    add_inputs: bool,
    no_lora: torch.Tensor,
) -> None:
    _trace("sgmv_expand", no_lora=no_lora.item(), tok=int(x.shape[0]),
           tn=token_nums)
    if no_lora.item():
        return
    _SGMV_EXPAND_SLICE_FN(
        x, w, y,
        b_seq_start_loc, seq_len_tensor, lora_indices_tensor,
        batch_size, max_length, token_nums,
        y_offset, y_slice_size, add_inputs,
    )


@_sgmv_expand_slice_op.register_fake
def _sgmv_expand_slice_fake(y, x, w, b_seq_start_loc, seq_len_tensor,
                            lora_indices_tensor, batch_size, max_length,
                            token_nums, y_offset, y_slice_size, add_inputs, no_lora):
    return None


GMM_TOKEN_THRESHOLD = 1024

# Runtime gate for the gmm path (debugging the gmm accuracy regression).
#   off       -> gmm disabled, pure sgmv (committed-safe default, =97% acc)
#   threshold -> gmm on when token_nums > GMM_TOKEN_THRESHOLD (the prod regime)
#   force     -> gmm on for every batch (decode too) so it can be tested on short prompts
_GMM_MODE = os.environ.get("LORA_GMM", "off").lower()


class PunicaWrapperNPU(PunicaWrapperBase):
    """
    PunicaWrapperNPU: dual-launch gmm + sgmv.
    Both wrapped in custom_ops with early-exit flags (opaque to compile).
    Prefill: gmm runs, sgmv disabled. Decode: sgmv runs, gmm disabled.
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
        _SGMV_SHRINK_FN = sgmv_shrink
        _SGMV_EXPAND_SLICE_FN = sgmv_expand_slice

        # self._use_gmm_device = torch.tensor(False, dtype=torch.bool, device=device)
        self._use_gmm_shrink_cpu = torch.tensor(False, dtype=torch.bool)
        self._use_gmm_expand_cpu = torch.tensor(False, dtype=torch.bool)

    def update_metadata(self, mapping, lora_index_to_id, max_loras, vocab_size, **kwargs):
        super().update_metadata(mapping, lora_index_to_id, max_loras, vocab_size, **kwargs)
        if _GMM_MODE == "force":
            enabled = True
        elif _GMM_MODE == "threshold":
            enabled = bool(self.token_nums > GMM_TOKEN_THRESHOLD)
        else:
            enabled = False
        # self._use_gmm_device.fill_(enabled)
        self._use_gmm_expand_cpu.fill_(enabled)
        self._use_gmm_shrink_cpu.fill_(enabled)
        _trace("update_metadata", is_prefill=getattr(self, "is_prefill", "?"),
               tn=self.token_nums, enabled=enabled, bsz=self.batch_size)

    def add_shrink(
        self,
        y: tuple[torch.Tensor, ...] | torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ):
        x = x.view(-1, x.shape[-1])
        gmm_no_lora = self._use_gmm_shrink_cpu.logical_not()
        sgmv_no_lora = self._use_gmm_shrink_cpu.clone()
        for slice_idx in range(len(lora_a_stacked)):
            torch.ops.lora.gmm_shrink(
                x, lora_a_stacked[slice_idx],
                y[slice_idx].view(-1, y[slice_idx].shape[-1]),
                *self.prefill_metadata, scale,
                gmm_no_lora,
            )
            torch.ops.lora.sgmv_shrink(
                x, lora_a_stacked[slice_idx],
                y[slice_idx].view(-1, y[slice_idx].shape[-1]),
                *self.prefill_metadata, scale,
                sgmv_no_lora,
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
        gmm_no_lora = self._use_gmm_expand_cpu.logical_not()
        sgmv_no_lora = self._use_gmm_expand_cpu.clone()
        offset_left = offset_start
        if lora_bias_stacked is not None:
            self._apply_bias(self.token_lora_indices, y, output_slices, lora_bias_stacked)
        for slice_idx in range(len(lora_b_stacked)):
            torch.ops.lora.gmm_expand_slice(
                y, x[slice_idx], lora_b_stacked[slice_idx],
                *self.prefill_metadata,
                offset_left, output_slices[slice_idx], add_inputs,
                gmm_no_lora,
            )
            torch.ops.lora.sgmv_expand_slice(
                y, x[slice_idx], lora_b_stacked[slice_idx],
                *self.prefill_metadata,
                offset_left, output_slices[slice_idx], add_inputs,
                sgmv_no_lora,
            )
            offset_left += output_slices[slice_idx]
        y = y.view_as(y_org)

    # --- kept for add_lora_embedding / add_lora_logits ---

    def _expand_prefill(self, y, x, w_t_all, add_inputs):
        if self.no_lora:
            return
        self.sgmv_expand(x, w_t_all, y, *self.prefill_metadata, add_inputs)

    def _expand_decode(self, y, x, w_t_all, add_inputs):
        self.bgmv_expand(x, w_t_all, y, self.token_lora_indices, add_inputs)

    def add_lora_embedding(self, y, x, lora_b_stacked, add_inputs=True, **kwargs):
        expand_fun: Callable = self._expand_prefill if self.is_prefill else self._expand_decode
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
