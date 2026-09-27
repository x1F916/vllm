# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU Triton HyperConnection kernels for Qwen4Exp."""

# TODO(refactor): Share the HC kernel math once launch hooks cover CUDA PDL,
# Triton-CPU thread configuration, and backend-local custom-op registration.

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .launch import grid_num_threads, launch


@triton.jit
def _grouped_gemma_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_x,
    stride_y,
    DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
) -> None:
    GROUP_DIM: tl.constexpr = DIM // NUM_GROUPS
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(GROUP_DIM)
    pid = tl.program_id(0)
    group_id = pid % NUM_GROUPS
    row = pid // NUM_GROUPS
    offs_g = tl.arange(0, BLOCK_SIZE)
    offsets = group_id * GROUP_DIM + offs_g
    mask = offs_g < GROUP_DIM
    w_offs = offs_g if W_SHARED else offsets

    x = tl.load(x_ptr + row * stride_x + offsets, mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + w_offs, mask, other=0.0)
    rrms = tl.rsqrt(tl.sum(x * x) / GROUP_DIM + EPS)
    y = x * rrms
    y += y * w.to(tl.float32)
    tl.store(y_ptr + row * stride_y + offsets, y, mask)


def _grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    N, DIM = x.shape
    assert x.stride(1) == 1, "grouped Gemma RMSNorm requires unit inner stride"
    assert weight.is_contiguous(), "grouped Gemma RMSNorm weight must be contiguous"
    assert DIM % num_groups == 0
    group_dim = DIM // num_groups
    assert weight.numel() in (group_dim, DIM)

    y = x.new_empty(x.shape)
    grid = (N * num_groups,)
    launch(
        _grouped_gemma_rmsnorm_kernel,
        grid,
        x,
        weight,
        y,
        x.stride(0),
        y.stride(0),
        DIM,
        num_groups,
        W_SHARED=weight.numel() == group_dim,
        EPS=eps,
        num_cpu_threads=grid_num_threads(*grid),
    )
    return y


@triton.jit
def _hc_combine_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    out_ptr,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)
    row = tl.program_id(0)
    tile_id = tl.program_id(1)
    offs_inner = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_inner = offs_inner < HC_DIM
    offs_hc = tl.arange(0, HC_PAD)
    mask_hc = offs_hc < HC
    offs = offs_hc[:, None] * HC_DIM + offs_inner[None, :]
    mask = mask_hc[:, None] & mask_inner[None, :]

    if inj_ptr is not None:
        inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
    block = tl.load(block_ptr + row * stride_block + offs_inner, mask_inner, other=0.0)
    res = tl.load(res_ptr + row * stride_res + offs, mask, other=0.0)
    if inj_ptr is not None:
        inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
        out = res.to(tl.float32) + block.to(tl.float32)[None, :] * inj[:, None]
    else:
        out = res.to(tl.float32) + block.to(tl.float32)
    tl.store(out_ptr + row * stride_out + offs, out, mask=mask)


def _hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    N, DIM = residual.shape
    assert DIM % hc_count == 0
    hc_dim = DIM // hc_count
    assert block_output.shape == (N, hc_dim)
    assert residual.stride(1) == 1
    assert block_output.stride(1) == 1
    if injection_logits is not None:
        assert injection_logits.shape == (N, hc_count)
        assert injection_logits.stride(1) == 1
    stride_injection = injection_logits.stride(0) if injection_logits is not None else 0

    out = residual.new_empty(residual.shape)
    BLOCK_SIZE = 512
    grid = (N, triton.cdiv(hc_dim, BLOCK_SIZE))
    launch(
        _hc_combine_kernel,
        grid,
        block_output,
        residual,
        injection_logits,
        out,
        block_output.stride(0),
        residual.stride(0),
        stride_injection,
        out.stride(0),
        hc_dim,
        hc_count,
        BLOCK_SIZE,
        num_cpu_threads=grid_num_threads(*grid),
    )
    return out


def _same_shape_fake(x: torch.Tensor, *args) -> torch.Tensor:
    return x.new_empty(x.shape)


def _hc_combine_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    del block_output, injection_logits, hc_count
    return residual.new_empty(residual.shape)


direct_register_custom_op(
    op_name="qwen4_exp_cpu_grouped_gemma_rmsnorm",
    op_func=_grouped_gemma_rmsnorm,
    fake_impl=_same_shape_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_cpu_hc_combine",
    op_func=_hc_combine,
    fake_impl=_hc_combine_fake,
)


def grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_cpu_grouped_gemma_rmsnorm(
        x, weight, eps, num_groups
    )


# silu, gate_mix and combine_norm run once or twice per HC module on a few
# KiB per token. As plain torch ops Inductor fuses them into the surrounding
# graph, which is cheaper than a Triton launch per op.
def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    x_f32 = x.float() / hc_count
    return (x_f32 * torch.sigmoid(x_f32)).to(x.dtype)


def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    gated = torch.sigmoid(gate.float()) * x.float()
    return gated.unflatten(-1, (hc_count, -1)).sum(-2).div(hc_count).to(x.dtype)


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_cpu_hc_combine(
        residual, block_output, injection_logits, hc_count
    )


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_f32 = residual.float().unflatten(-1, (hc_count, -1))
    block_f32 = block_output.float().unsqueeze(-2)
    if injection_logits is not None:
        scale = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)
        block_f32 = block_f32 * scale.unsqueeze(-1)
    # Preserve the unfused combine -> RMSNorm rounding boundary.
    combined = (residual_f32 + block_f32).to(residual.dtype)
    combined_f32 = combined.float()
    variance = combined_f32.square().mean(-1, keepdim=True)
    normed = combined_f32 * torch.rsqrt(variance + eps)
    weight = norm_weight.float().view(-1, combined.shape[-1])
    normed = normed + normed * weight
    return combined.flatten(-2), normed.to(residual.dtype).flatten(-2)


__all__ = [
    "grouped_gemma_rmsnorm",
    "hc_combine",
    "hc_combine_norm",
    "hc_gate_mix",
    "hc_silu",
]
