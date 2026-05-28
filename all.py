from collections.abc import Callable
from typing import List

import torch
import torch_npu
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm_ascend.lora.utils import refresh_all_lora_classes
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

print("\nhui"*10)

_TRANSPOSED_WEIGHT_CACHE: dict[int, torch.Tensor] = {}
_SGMV_SHRINK_FN = None
_SGMV_EXPAND_SLICE_FN = None


def _get_transposed_weight(w: torch.Tensor) -> torch.Tensor:
    key = w.data_ptr()
    if (cached := _TRANSPOSED_WEIGHT_CACHE.get(key)) is not None:
        return cached
    if w.ndim == 4:
        w = w.squeeze(1)
    w_t = w.transpose(1, 2).contiguous()
    _TRANSPOSED_WEIGHT_CACHE[key] = w_t
    return w_t


def _gather_weights_for_gmm(w: torch.Tensor, lora_indices: torch.Tensor) -> torch.Tensor:
    w_t = _get_transposed_weight(w)
    safe_indices = lora_indices.clamp(min=0)
    gathered = w_t[safe_indices]
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
    if no_lora.item():
        return
    gathered_w = _gather_weights_for_gmm(lora_a_weight, lora_indices_tensor)
    x_in = inputs if inputs.dtype == gathered_w.dtype else inputs.to(gathered_w.dtype)
    result = torch_npu.npu_grouped_matmul(
        x=[x_in], weight=[gathered_w],
        split_item=2, group_list_type=1, group_type=0,
        group_list=seq_len_tensor,
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
    if no_lora.item():
        return
    gathered_w = _gather_weights_for_gmm(w, lora_indices_tensor)
    # Cast weights up to match x (fp32 from shrink), not x down to bf16
    w_in = gathered_w if gathered_w.dtype == x.dtype else gathered_w.to(x.dtype)
    result = torch_npu.npu_grouped_matmul(
        x=[x], weight=[w_in],
        split_item=2, group_list_type=1, group_type=0,
        group_list=seq_len_tensor,
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
        val = self.token_nums > GMM_TOKEN_THRESHOLD
        # self._use_gmm_device.fill_(val)
        self._use_gmm_expand_cpu.fill_(False)
        self._use_gmm_shrink_cpu.fill_(False)

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
