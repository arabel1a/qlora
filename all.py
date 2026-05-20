from typing import List
import torch
import torch_npu
import triton
import triton.language as tl
import torch
import os
_LORA_A_PTR_DICT: dict[tuple[int, ...], tuple[torch.tensor, ...]] = {}
NUM_AI_CORES=20
import torch._dynamo
from collections.abc import Callable
import torch
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm_ascend.lora.utils import refresh_all_lora_classes
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type



# os.environ["MLIR_ENABLE_DUMP"] = "1"
# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
# os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_debug"
# torch._dynamo.config.repro_after="dynamo"
"""Baseline implementations for LoRA shrink: native torch and C++ sgmv_shrink.

Both accept compute_meta-style metadata (production format from vllm).
The sgmv_shrink signature matches vllm_ascend/lora/lora_ops.py exactly.
"""

def sort_metadata(token_lora_tensor, b_seq_start_loc, seq_len_tensor, lora_indices_tensor, batch_size, max_length, token_nums, no_lora):      
    return (
        b_seq_start_loc, seq_len_tensor, lora_indices_tensor,  
        batch_size, max_length, token_nums, 
    )


@torch.no_grad()
def stub_shrink(
    inputs: torch.Tensor,  # (num_tokens_in_batch, hidden_size)
    lora_a_weights: torch.Tensor,  # (num_loras, rank, hidden_size)
    output_tensor: torch.Tensor,  # (num_tokens_in_batch, hidden_size)
    b_seq_start_loc: torch.Tensor,  # (N_groups) - token indices where lora adapter changes. N_groups < N_requests
    seq_len_tensor: torch.Tensor,  # exactly b_seq_start_loc[1:] - b_seq_start_loc[:1]
    lora_indices_tensor: torch.Tensor,  # (N_groups) group idx -> lora idx
    batches: int,  # N_groups
    max_seq_length: int,  # seq_len_tensor.max.items()
    token_nums: int,  # num_tokens_in_batch
    scaling: float,
) -> None:
    pass
    
@torch.no_grad()
def torch_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int, 
    scaling: float,
) -> None:
    token_indices = torch.arange(inputs.shape[0], device=inputs.device)
    group_id_per_token = torch.bucketize(token_indices, b_seq_start_loc, right=True) - 1
    token_to_lora_idx = lora_indices_tensor[group_id_per_token]
    for slice_idx in range(len(lora_a_weights)):
        token_to_lora = lora_a_weights[slice_idx][token_to_lora_idx].squeeze(1)
        # lora_output = torch.einsum("th,tlh->tl", inputs, token_to_lora)
        lora_output = torch.matmul(inputs.unsqueeze(1), token_to_lora.transpose(-1, -2)).squeeze(1)
        output_tensor[slice_idx, ...] = lora_output * scaling

def _get_lora_a_ptr(lora_a_weights: list[torch.Tensor], device: torch.device):
    """
    `_LORA_A_PTR_DICT` collects the required information during `profile_run`,
    After this, it remains constant and subsequent usage is through LUT.
    Refer to:
    https://github.com/triton-lang/triton/blob/release/3.1.x/python/tutorials/08-grouped-gemm.py
    """
    key = tuple(lora_weight.data_ptr() for lora_weight in lora_a_weights)

    if values := _LORA_A_PTR_DICT.get(key):
        return values

    lora_strides_d0 = []
    lora_strides_d1 = []
    lora_strides_d2 = []
    tensor_ptrs = []
    for lora_a_weight in lora_a_weights:
        if lora_a_weight.ndim == 4:  # shape:(lora_num,1,size,rank)
            assert lora_a_weight.size(1) == 1
            lora_a_weight = lora_a_weight.squeeze(dim=1)
        else:
            assert lora_a_weight.ndim == 3  # shape:(lora_num,size,rank)
        assert lora_a_weight.is_contiguous()
        tensor_ptrs.append(lora_a_weight.data_ptr())
        lora_strides_d0.append(lora_a_weight.stride(0))
        lora_strides_d1.append(lora_a_weight.stride(1))
        lora_strides_d2.append(lora_a_weight.stride(2))
    if len(lora_a_weights) > 1:
        lora_ptr_tensor = torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64)
    else:
        lora_ptr_tensor = lora_a_weights[0]

    if (
        len(set(lora_strides_d0)) > 1
        or len(set(lora_strides_d1)) > 1
        or len(set(lora_strides_d2)) > 1
    ):
        raise ValueError("All LoRA weights must have the same stride.")

    _LORA_A_PTR_DICT[key] = (
        lora_ptr_tensor,
        lora_strides_d0[0],
        lora_strides_d1[0],
        lora_strides_d2[0],
    )
    return _LORA_A_PTR_DICT.get(key)



@triton.jit
def _lora_shrink_kernel(
    input_ptr, lora_ptr, out_ptr,  # data
    b_seq_start_loc, seq_len_tensor, lora_indices_tensor,  # indices
    M, N, K,  # sizes
    scaling,  # fp32 scale
    NUM_GROUPS: tl.constexpr, # and this
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    N_NUM_BLOCKS: tl.constexpr,
    MAX_TILES_PER_CORE: tl.constexpr, # not super sure whethter this should be constexpr
    K_NUM_BLOCKS: tl.constexpr,
    NUM_AI_CORES: tl.constexpr,
    DTYPE: tl.constexpr,
    CAST:  tl.constexpr,
):
    pid = tl.program_id(axis=0)

    total = 0
    for g in range(NUM_GROUPS):
        total += tl.load(seq_len_tensor + g).to(tl.int32)

    # scheduligng: each core gets single lora, 
    # each lora block is gets cores proportinally to length
    used_cores = 0
    my_first_token = 0
    my_last_token = 0
    lora_id = tl.zeros((), dtype=tl.int64)
    for g in range(NUM_GROUPS):
        seq_len = tl.load(seq_len_tensor + g)
        # split cores proportionally to seq_at least one core each
        # assumes NUM_AI_CORES > NUM_GROUPS !
        cores_for_g = 1 + (NUM_AI_CORES - NUM_GROUPS) * seq_len // total
        # cores_for_g = tl.maximum(1, (seq_len * NUM_AI_CORES + total - 1) // total)
        chunk = seq_len // cores_for_g
        after_this_used_cores = used_cores + cores_for_g

        # check if current pid within this group
        local = pid - used_cores
        is_my_group = (pid >= used_cores) & (pid < after_this_used_cores)

        # find tokens
        grp_start = tl.load(b_seq_start_loc + g)
        if is_my_group:
            my_first_token = (grp_start + local * chunk).to(tl.int32)
            my_last_token = tl.where(local < cores_for_g - 1,grp_start + local * chunk + chunk, grp_start + seq_len).to(tl.int32)
        # my_first_token = tl.where(
        #     is_my_group, (grp_start + local * chunk).to(tl.int32), 0
        # )
        # my_last_token = tl.where(
        #     is_my_group, tl.where(local < cores_for_g - 1,grp_start + local * chunk + chunk,grp_start + seq_len), 0
        # ).to(tl.int32)
        used_cores = after_this_used_cores.to(tl.int32)
        lora_id = tl.where(is_my_group, tl.load(lora_indices_tensor + g), lora_id).to(tl.int64)

    if my_last_token == my_first_token:
        return
    for slice_id in tl.static_range(SLICE_NUM):
        if SLICE_NUM == 1:
            slice_base = lora_ptr
        else:
            slice_base = tl.load(lora_ptr + slice_id).to(
                 tl.pointer_type(DTYPE), bitcast=True
             )

        for tile_idx in range(MAX_TILES_PER_CORE):
            row_start = my_first_token + tile_idx * BLOCK_M
        
            a_block_ptr = tl.make_block_ptr(
                base=input_ptr,
                shape=(my_last_token, K),
                strides=(K, 1),
                offsets=(row_start, 0),
                block_shape=(BLOCK_M, BLOCK_K),
                order=(1, 0),
            )
            
            for n_blk in range(N_NUM_BLOCKS):
                lora_base = slice_base + lora_id * K * N
                b_block_ptr = tl.make_block_ptr(
                    base=lora_base,
                    shape=(K, N),
                    strides=(1, K),
                    offsets=(0, n_blk * BLOCK_N),
                    block_shape=(BLOCK_K, BLOCK_N),
                    order=(0, 1),
                )
                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k_idx in range(K_NUM_BLOCKS):
                    a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))
                    b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))
                    accumulator += tl.dot(a_tile, b_tile)
                    a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
                    b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

                accumulator = (accumulator * scaling)
                if CAST:
                    accumulator = accumulator.to(DTYPE)

                # [num_slices, num_tokens, rank] -> [num_slices * num_tokens, rank]
                c_block_ptr = tl.make_block_ptr(
                    base=out_ptr,
                    shape=(slice_id * M + my_last_token, N),
                    strides=(N, 1),
                    offsets=(slice_id * M + row_start, n_blk * BLOCK_N),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))


# @torch.inference_mode()
# @torch.no_grad()
known_signatures = set()
@torch.library.custom_op("misha::_shrink", mutates_args=("output_tensor",))
def _shrink_op(
    inputs: torch.Tensor, # M x K
    lora_a_weights: List[torch.Tensor], # slices x loras x K x N
    output_tensor: torch.Tensor, # slices x M x N
    b_seq_start_loc: torch.Tensor,  # start of each group (of tokens sharing the adapter)
    seq_len_tensor: torch.Tensor,  # token cnt per group
    lora_indices_tensor: torch.Tensor,  # group -> lora mapping
    NUM_GROUPS: int, # number of consecutive pieces of the same lora
    max_length: int,  # longest group
    token_nums: int,  # total tokens
    scaling: float,
) -> None:
    # torch._check(NUM_GROUPS <= NUM_AI_CORES)
    # assert NUM_GROUPS <= NUM_AI_CORES
    # assert inputs.dtype == lora_a_weights[0].dtype
    # assert inputs.dtype in [torch.float16, torch.bfloat16]
    TRITON_DTYPE = tl.float16 if inputs.dtype == torch.float16 else tl.bfloat16
    # assert inputs.is_contiguous()
    # assert output_tensor.is_contiguous()
    
    # constants
    M = inputs.size(0)
    if len(lora_a_weights[0].shape) == 3:
        NUM_LORAS, N, K = lora_a_weights[0].shape
    else:
        NUM_LORAS, _, N, K = lora_a_weights[0].shape
        # assert _ == 1

    NUM_SLICES = len(lora_a_weights)
    # kernel_config = get_lora_op_configs()
    BLOCK_M = 32 # = kernel_config["block_m"]
    BLOCK_N = 32 # = kernel_config["block_n"]
    BLOCK_K = 32 # = kernel_config["block_k"]

    # assert inputs.shape == (M, K)
    # assert output_tensor.shape == (NUM_SLICES, M, N)

    lora_ptr_tensor, lora_strides_d0, lora_strides_d1, lora_strides_d2 = (
        _get_lora_a_ptr(lora_a_weights, inputs.device)
    )
    # lora_ptr_tensor = None
    # assert (lora_strides_d0, lora_strides_d1, lora_strides_d2) == (K * N, K, 1)

    cpg = max(1, NUM_AI_CORES // NUM_GROUPS) # guaranteed cores per group    
    tpc = triton.cdiv(max_length, cpg) # worst case token per group
    MAX_TILES_PER_CORE = 2 * triton.next_power_of_2(triton.cdiv(tpc, BLOCK_M)) # worst case tile per group

    # perfect cshedule will get something like
    # MAX_TILES_PER_CORE = triton.next_power_of_2(token_nums // NUM_AI_CORES //BLOCK_M)
    
    N_NUM_BLOCKS = triton.cdiv(N, BLOCK_N)
    K_NUM_BLOCKS = triton.cdiv(K, BLOCK_K)
    grid = (NUM_AI_CORES,)

    constexprs = (NUM_GROUPS, # at most 20 -> not that much of compilation burden
        BLOCK_M, BLOCK_N, BLOCK_K,
        NUM_SLICES,
        N_NUM_BLOCKS,
        MAX_TILES_PER_CORE, # powers of 2 -> not that much
        K_NUM_BLOCKS,
        NUM_AI_CORES,
        TRITON_DTYPE,
        output_tensor.dtype != torch.float32,
    )
    if constexprs not in known_signatures:
        known_signatures.add(constexprs)
        print(constexprs)
 
    # output_tensor.zero_()
    _lora_shrink_kernel[grid](
        inputs, lora_ptr_tensor, output_tensor,
        b_seq_start_loc, seq_len_tensor, lora_indices_tensor,
        M, N, K,
        scaling,
        # constexprs
        NUM_GROUPS, # at most 20 -> not that much of compilation burden
        BLOCK_M, BLOCK_N, BLOCK_K,
        NUM_SLICES,
        N_NUM_BLOCKS,
        MAX_TILES_PER_CORE, # powers of 2 -> not that much
        K_NUM_BLOCKS,
        NUM_AI_CORES,
        TRITON_DTYPE,
        output_tensor.dtype != torch.float32,
    )


@_shrink_op.register_fake
def _shrink_fake(
    inputs: torch.Tensor,
    lora_a_weights: List[torch.Tensor],
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    NUM_GROUPS: int,
    max_length: int,
    token_nums: int,
    scaling: float,
) -> None:
    """
    Meta-kernel for torch.compile to trace shapes and dtypes.
    No actual computation happens here.
    """
    # torch._check(inputs.dim() == 2, lambda: f"Expected inputs to be 2D, got {inputs.dim()}D")
    # M, K = inputs.shape
    
    # torch._check(len(lora_a_weights) > 0, lambda: "lora_a_weights cannot be empty")
    
    # w_shape = lora_a_weights[0].shape
    # torch._check(len(w_shape) in (3, 4), lambda: f"Expected 3D or 4D lora weights, got {len(w_shape)}D")
    
    # if len(w_shape) == 3:
    #     NUM_LORAS, N, K_weight = w_shape
    # else:
    #     NUM_LORAS, _, N, K_weight = w_shape
        
    # torch._check(K == K_weight, lambda: f"K mismatch: inputs K={K}, weights K={K_weight}")
    
    # NUM_SLICES = len(lora_a_weights)
    
    # out_shape = output_tensor.shape
    # expected_shape_3d = (NUM_SLICES, M, N)
    # expected_shape_2d = (NUM_SLICES * M, N)
    
    # torch._check(
    #     out_shape == expected_shape_3d or out_shape == expected_shape_2d,
    #     lambda: f"Unexpected output_tensor shape {out_shape}. Expected {expected_shape_3d}"
    # )
    
    return None

triton_shrink = torch.ops.misha._shrink.default


print ("hui " * 10)


USE_STUB_KERNEL=int(os.environ.get("VLLM_LORA_USE_STUB_KERNEL", 0))
USE_TRITON_KERNEL=int(os.environ.get("VLLM_LORA_USE_TRITON_KERNEL", 0))
USE_TORCH_KERNEL=int(os.environ.get("VLLM_LORA_USE_TORCH_KERNEL", 0))
SPLIT_PREFILL_DECODE=os.environ.get("VLLM_LORA_SPLIT_PREFILL_DECODE")

if sum([USE_STUB_KERNEL, USE_TORCH_KERNEL, USE_TRITON_KERNEL]) > 1:
    raise ValueError("Choose one kernel, man.")
    
PATCHED_KERNEL = True # any([USE_STUB_KERNEL, USE_TORCH_KERNEL, USE_TRITON_KERNEL])
if PATCHED_KERNEL:
    print("\n" * 10)
    print(f"WARNING: Misha patched lora kernels {USE_STUB_KERNEL=} {USE_TORCH_KERNEL=} {USE_TRITON_KERNEL=}")
    print("\n" * 10)

class PunicaWrapperNPU(PunicaWrapperBase):
    """
    PunicaWrapperNPU is designed to manage and provide metadata for the punica
    kernel. The main function is to maintain the state information for
    Multi-LoRA, and to provide the interface for the pytorch punica ops.
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
        if PATCHED_KERNEL:
            self.triton_shrink = triton_shrink
            self.torch_shrink = torch_shrink
            self.stub_shrink = stub_shrink

    def _shrink_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        # No LoRA request, so return directly
        if self.no_lora:
            return
        self.sgmv_shrink(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            scale,
        )

    def _shrink_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        # print("decode")
        self.bgmv_shrink(x, w_t_all, y, self.token_lora_indices, scale)

    def _expand_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_inputs: bool,
    ):
        # No LoRA request, so return directly
        if self.no_lora:
            return
        self.sgmv_expand(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            add_inputs,
        )

    def _expand_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_inputs: bool,
    ):
        self.bgmv_expand(x, w_t_all, y, self.token_lora_indices, add_inputs)

    def _expand_slice_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ):
        # No LoRA request, so return directly
        if self.no_lora:
            return
        self.sgmv_expand_slice(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            y_offset,
            y_slice_size,
            add_inputs,
        )

    def _expand_slice_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ):
        self.bgmv_expand_slice(x, w_t_all, y, self.token_lora_indices, y_offset, y_slice_size, add_inputs)

    def _apply_expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool = True,
    ):
        """
        Perform the ` y[:,y_offset:y_offset+y_slice_size]+=x@w_t_all`
        computation, which is suitable for the
        GEMM of lora'b.
        """

        expand_slice_fun: Callable = self._expand_slice_prefill if self.is_prefill else self._expand_slice_decode
        expand_slice_fun(y, x, w_t_all, y_offset, y_slice_size, add_inputs)

    def _apply_shrink(self, y: torch.Tensor, x: torch.Tensor, w_t_all: torch.Tensor, scale: float):
        """
        Perform the ` y+=x@w_t_all` computation, which is suitable for the
        GEMM of lora'a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        shrink_fun: Callable = self._shrink_prefill if self.is_prefill else self._shrink_decode
        shrink_fun(y, x, w_t_all, scale)
        y = y.view_as(y_org)
    
    def add_shrink(
            self,
            y: tuple[torch.Tensor, ...] | torch.Tensor,
            x: torch.Tensor,
            lora_a_stacked: tuple[torch.Tensor, ...],
            scale: float,
            **kwargs,
        ):
        """
        Performs GEMM  for multiple slices of lora_a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale

        Args:
            y (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """
        x = x.view(-1, x.shape[-1])
        if USE_TRITON_KERNEL and x.shape[0] > 1024:
            for slice_idx in range(len(lora_a_stacked)):
                torch.ops.misha._shrink.default(
                        x, lora_a_stacked[slice_idx:slice_idx+1], y[slice_idx].unsqueeze(0),
                    *self.prefill_metadata, scale
                )
        else:
            for slice_idx in range(len(lora_a_stacked)):
                self._apply_shrink(y[slice_idx], x, lora_a_stacked[slice_idx], scale)

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
        Performs GEMM and bias addition for multiple slices of lora_b.

        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i] +
                    lora_bias_stacked[i]
                offset += slice

        Args:
            y (torch.Tensor): Output tensor.
            x (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Input tensors
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]):
                bias's weight
            output_slices (Tuple[int, ...]): Every slice's size
            add_inputs (bool):  Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        offset_left = offset_start
        if lora_bias_stacked is not None:
            self._apply_bias(self.token_lora_indices, y, output_slices, lora_bias_stacked)
        for slice_idx in range(len(lora_b_stacked)):
            self._apply_expand(
                y,
                x[slice_idx],
                lora_b_stacked[slice_idx],
                offset_left,
                output_slices[slice_idx],
                add_inputs=add_inputs,
            )
            offset_left += output_slices[slice_idx]
        y = y.view_as(y_org)

    def add_lora_embedding(
        self, y: torch.Tensor, x: torch.Tensor, lora_b_stacked: torch.Tensor, add_inputs: bool = True, **kwargs
    ) -> None:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """

        # Embedding layer only need expand op
        expand_fun: Callable = self._expand_prefill if self.is_prefill else self._expand_decode
        x = x.to(torch.float32)
        expand_fun(y, x, lora_b_stacked, add_inputs)

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
                    ).squeeze(0)+lora_bias_stacked[i]

        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (Tuple[int, ...]): Every slice's size.
            buffer (Optional[Tuple[torch.Tensor, ...]]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)

        if buffer is None:
            r = lora_b_stacked[0].size(-1)
            # We set the buffer to be float32 by default, consistent with the
            # triton op
            buffer = tuple(
                torch.zeros((x.size(0), r), dtype=torch.float32, device=x.device) for _ in range(len(output_slices))
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
        Applies lora  specifically for LogitsProcessorWithLoRA.

        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor):lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]):Default to None.
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
