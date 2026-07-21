#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Ascend MoE-LoRA wrapper (backport of vllm-ascend PR #10977 to v0.18.0).

Design:
  - Inherit weight allocation / set_lora / slice helpers from upstream
    ``FusedMoEWithLoRA``. Only the injection mechanism differs: upstream wraps
    Triton modular-kernel internals (``TritonExperts``) which do not exist on
    Ascend. We instead publish a per-layer ``MoELoRAContext`` on the base layer
    (``_ascend_moe_lora_context``); the Ascend unquant MoE path threads it
    through ``MoEFusedExpertsInput`` -> ``MoEMlpComputeInput`` and applies the
    LoRA delta natively inside ``unquant_apply_mlp`` (``moe_lora_apply_w13`` /
    ``moe_lora_apply_w2``) -- no runtime monkey-patch of ``comm._apply_mlp``.

  - v1 scope: unquant + AllGather + TP-only + no shared experts + no FusedMC2 +
    no dynamic EPLB. These are the exact conditions under which
    ``Qwen3-30B-A3B-Thinking-2507`` runs with TP=4 EP=1. Other paths assert
    early so users get a clear error rather than silently wrong outputs.

  v0.18.0 backport notes vs the PR (which targets main / v0.23.0):
  - The per-layer context host is always ``self.base_layer`` here: in v0.18 the
    unquant ``apply`` is called with ``layer == the FusedMoE module`` (the same
    object we wrap), so reading ``getattr(layer, "_ascend_moe_lora_context")``
    on it works directly -- there is no ``routed_experts`` runner split.
  - ``MoELoRAContext`` / ``_build_lora_context`` are provided here (they are NOT
    in upstream vllm 0.18 nor in the PR head); the fields read upstream's
    ``w13_lora_a_stacked`` / ``w2_lora_a_stacked`` / ``adapter_enabled`` stacks
    populated by ``FusedMoEWithLoRA.create_lora_weights``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from vllm import envs
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
from vllm.lora.layers.utils import _get_lora_device

import vllm_ascend.envs as envs_ascend


@dataclass
class MoELoRAContext:
    """Per-layer MoE-LoRA state published on the base FusedMoE layer.

    Holds stable references only (the in-place-updated LoRA weight stacks, the
    adapter_enabled flag tensor and the punica wrapper), so it is built once in
    ``set_mapping`` and reused every forward.
    """

    top_k: int
    punica_wrapper: Any
    w13_lora_a_stacked: tuple
    w13_lora_b_stacked: tuple
    w2_lora_a_stacked: tuple
    w2_lora_b_stacked: tuple
    adapter_enabled: torch.Tensor


def _assert_ascend_moe_lora_supported(base_layer: nn.Module) -> None:
    if getattr(base_layer, "use_ep", False):
        raise AssertionError(
            "Ascend MoE LoRA v1 does not support expert parallelism. "
            "Launch with `--enable-expert-parallel=false` and use TP only "
            "(e.g. TP=4 for Qwen3-30B-A3B on 4x64GB)."
        )
    if getattr(base_layer, "dynamic_eplb", False):
        raise AssertionError(
            "Ascend MoE LoRA v1 is incompatible with dynamic EPLB "
            "(expert migration would break the per-expert LoRA layout)."
        )
    if int(envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2) != 0:
        raise AssertionError(
            "Ascend MoE LoRA v1 cannot patch FusedMC2 path "
            "(dispatch_ffn_combine is a single fused C++ op). "
            "Set VLLM_ASCEND_ENABLE_FUSED_MC2=0."
        )
    if getattr(base_layer, "_shared_experts", None) is not None:
        raise AssertionError(
            "Ascend MoE LoRA v1 does not wrap the shared_experts path "
            "(it runs outside quant_method.apply). The target model "
            "Qwen3-30B-A3B-Thinking-2507 has no shared experts; models "
            "like DeepSeek-V3 are not yet supported."
        )
    if getattr(base_layer, "multistream_overlap_gate", False):
        raise AssertionError(
            "multistream_overlap_gate=True interleaves quant_method.apply "
            "calls on multiple streams; the MoE LoRA path has not been "
            "validated under this overlap. Disable it for MoE LoRA."
        )


def _recover_moe_lora_routing(lora_context, expanded_row_idx, topk_ids):
    """Recover per-permuted-row (expert_id, lora_slot) for the dispatched rows.

    npu_moe_init_routing semantics (verified empirically): ``expanded_row_idx``
    is indexed by the ORIGINAL flat (token, k) position and gives where that
    pair landed in the expert-sorted array -- not the reverse. So recovering
    "which (token, k) pair does sorted row i hold" needs the inverse permutation
    of ``expanded``, not a direct gather by it. ``argsort`` output shape ==
    input shape (value-independent), so this stays graph-capturable -- no
    ``.item()`` / data-dependent host sync.
    """
    top_k = lora_context.top_k
    expanded = torch.abs(expanded_row_idx)
    # argsort on int32/int64 falls back to AiCPU (serial, ~9x TTFT hit). The sort
    # key is a permutation of row ids < max_num_batched_tokens*top_k (<< 2^24), so
    # float32 represents every value EXACTLY -> identical order, but the kernel now
    # runs on AiCore. (indices returned are still int64, used for the gathers.)
    inv_perm = torch.argsort(expanded.to(torch.float32))
    expert_per_row = topk_ids.reshape(-1)[inv_perm].to(torch.long)

    # token_lora_indices is a 1D LongTensor sized to max_num_batched_tokens
    # (host-known constant). Clamping defensively to the last index is a no-op
    # in normal operation but keeps the gather graph-safe.
    orig_token = inv_perm // top_k
    token_lora_indices = lora_context.punica_wrapper.token_lora_indices
    orig_token = orig_token.clamp_(max=token_lora_indices.numel() - 1)
    lora_per_row = token_lora_indices[orig_token]
    return expert_per_row, lora_per_row


# Marker returned by moe_lora_apply_w13 when it took the grouped-matmul path, so
# moe_lora_apply_w2 takes the same path (and skips the bgmv routing tuple).
_GMM_ROUTING = ("gmm",)


def moe_lora_apply_w13(lora_context, *, gate_up_out, hidden_states, expanded_row_idx,
                       topk_ids, group_list=None, group_list_type=1):
    """Add the w13 LoRA delta into ``gate_up_out`` (in place), before activation.

    Called from ``unquant_apply_mlp`` right after the base gate_up GMM. When a
    single adapter is active and the batch is in the gmm regime (``_use_moe_gmm``,
    i.e. token_num > threshold / eager prefill), apply the delta with a grouped
    matmul that reuses the base ``group_list`` -- no per-row routing recovery.
    Otherwise fall back to the per-row bgmv path. Returns a marker/routing that
    tells the w2 hook which path to mirror.
    """
    pw = lora_context.punica_wrapper
    if getattr(pw, "_use_moe_gmm", False) and getattr(pw, "_single_active_lora_id", None) is not None:
        pw.add_lora_fused_moe_gmm(
            y=gate_up_out,
            x=hidden_states,
            lora_a_stacked=lora_context.w13_lora_a_stacked,
            lora_b_stacked=lora_context.w13_lora_b_stacked,
            group_list=group_list,
            group_list_type=group_list_type,
            lora_id=pw._single_active_lora_id,
        )
        return _GMM_ROUTING
    routing = _recover_moe_lora_routing(lora_context, expanded_row_idx, topk_ids)
    expert_per_row, lora_per_row = routing
    pw.add_lora_fused_moe(
        y=gate_up_out,
        x=hidden_states,
        lora_a_stacked=lora_context.w13_lora_a_stacked,
        lora_b_stacked=lora_context.w13_lora_b_stacked,
        expert_ids=expert_per_row,
        adapter_enabled=lora_context.adapter_enabled,
        token_lora_mapping=lora_per_row,
    )
    return routing


def moe_lora_apply_w2(lora_context, *, down_out, silu_out, lora_routing,
                      group_list=None, group_list_type=1):
    """Add the w2 LoRA delta into ``down_out`` (in place), after the down GMM.

    Mirrors the path w13 took: grouped matmul (reusing ``group_list``) when
    ``lora_routing is _GMM_ROUTING``, else the per-row bgmv reusing the routing
    tuple computed by w13. ``silu_out`` is the activation that fed the base down
    GMM (same expert-permuted order as ``group_list``).
    """
    pw = lora_context.punica_wrapper
    if lora_routing is _GMM_ROUTING:
        pw.add_lora_fused_moe_gmm(
            y=down_out,
            x=silu_out,
            lora_a_stacked=lora_context.w2_lora_a_stacked,
            lora_b_stacked=lora_context.w2_lora_b_stacked,
            group_list=group_list,
            group_list_type=group_list_type,
            lora_id=pw._single_active_lora_id,
        )
        return
    expert_per_row, lora_per_row = lora_routing
    pw.add_lora_fused_moe(
        y=down_out,
        x=silu_out,
        lora_a_stacked=lora_context.w2_lora_a_stacked,
        lora_b_stacked=lora_context.w2_lora_b_stacked,
        expert_ids=expert_per_row,
        adapter_enabled=lora_context.adapter_enabled,
        token_lora_mapping=lora_per_row,
    )


class AscendFusedMoEWithLoRA(FusedMoEWithLoRA):
    """Ascend-native MoE-LoRA wrapper.

    Reuses upstream weight allocation, set_lora, reset_lora, and slicing.
    Instead of the GPU modular-kernel injection, it publishes a per-layer
    ``MoELoRAContext`` onto the base layer (``_ascend_moe_lora_context``) that
    the Ascend unquant MoE path applies natively.
    """

    def __init__(self, base_layer: nn.Module) -> None:
        # Skip FusedMoEWithLoRA.__init__: it immediately asserts Triton
        # internals and calls _inject_lora_into_fused_moe which is GPU-only.
        BaseLayerWithLoRA.__init__(self)
        self.base_layer = base_layer
        _assert_ascend_moe_lora_supported(base_layer)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.device = _get_lora_device(base_layer)
        # VLLM_LORA_ENABLE_DUAL_STREAM is newer than vllm 0.18.0 -> optional.
        self._enable_aux_cuda_stream = getattr(envs, "VLLM_LORA_ENABLE_DUAL_STREAM", False)
        self.moe_config = base_layer.moe_config
        self._w13_slices = 2 if base_layer.moe_config.is_act_and_mul else 1

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------
    def set_mapping(self, punica_wrapper):
        # Upstream FusedMoEWithLoRA.set_mapping chains into the GPU modular
        # kernel (``self._moe_kernel...``) which we deliberately skip. Instead,
        # once punica_wrapper is available, build the per-layer MoELoRAContext
        # and publish it on ``self.base_layer`` -- the exact object v0.18's
        # unquant ``apply`` receives as ``layer`` and reads via
        # ``getattr(layer, "_ascend_moe_lora_context", None)``.
        BaseLayerWithLoRA.set_mapping(self, punica_wrapper)
        self.base_layer._ascend_moe_lora_context = self._build_lora_context()

    def _build_lora_context(self) -> MoELoRAContext:
        base = self.base_layer
        top_k = getattr(base, "top_k", None)
        if top_k is None:
            top_k = getattr(base.moe_config, "top_k", None)
        assert top_k is not None, "FusedMoE base layer exposes no top_k"
        return MoELoRAContext(
            top_k=int(top_k),
            punica_wrapper=self.punica_wrapper,
            w13_lora_a_stacked=self.w13_lora_a_stacked,
            w13_lora_b_stacked=self.w13_lora_b_stacked,
            w2_lora_a_stacked=self.w2_lora_a_stacked,
            w2_lora_b_stacked=self.w2_lora_b_stacked,
            adapter_enabled=self.adapter_enabled,
        )


class AscendFusedMoE3DWithLoRA(AscendFusedMoEWithLoRA, FusedMoE3DWithLoRA):
    """For checkpoints that already fuse w1+w3 into a 3D weight (single slice)."""

    def __init__(self, base_layer: nn.Module) -> None:
        AscendFusedMoEWithLoRA.__init__(self, base_layer)
        # Override: 3D MoE LoRA uses a single w13 slice.
        self._w13_slices = 1


# ----------------------------------------------------------------------
# Upstream compatibility shim: vllm/lora/model_manager.py:create_dummy_lora
# branches on ``module.__class__.__name__ == "FusedMoEWithLoRA"`` (and the 3D
# variant). Overriding only __name__ keeps the class objects distinct (so
# isinstance / type identity are unaffected) while letting the upstream string
# compare hit our subclasses (otherwise set_lora fails with
# "too many values to unpack (expected 3)").
# ----------------------------------------------------------------------
AscendFusedMoEWithLoRA.__name__ = "FusedMoEWithLoRA"
AscendFusedMoE3DWithLoRA.__name__ = "FusedMoE3DWithLoRA"
