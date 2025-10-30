# Copyright (c) 2025 Duyue Ma

from typing import Tuple

import torch
import triton
import triton.language as tl

from ..utils import NSAHelper

@triton.jit
def _construct_block_fwd_kernel(
    X,
    Y,
    X_CU_SEQLENS,
    Y_CU_SEQLENS,
    sxn, sxh, sxd,
    sym, syh, syk, syd,
    stride:tl.constexpr,
    kernel_size:tl.constexpr,
    D1:tl.constexpr,
    D2:tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    start_n = tl.program_id(2) * stride

    x_bos, x_eos = tl.load(X_CU_SEQLENS+off_b), tl.load(X_CU_SEQLENS+off_b+1)
    x_len = x_eos - x_bos
    if start_n >= x_len:
        return
    y_bos, y_eos = tl.load(Y_CU_SEQLENS+off_b), tl.load(Y_CU_SEQLENS+off_b+1)
    num_blocks = y_eos - y_bos

    block_idx = start_n // stride

    X += x_bos * sxn + off_h * sxh
    Y += y_bos * sym + off_h * syh

    off_n = start_n + tl.arange(0, stride)
    off_k = tl.arange(0, stride)

    load_mask = off_n[:, None] < x_len
    x_ptrs = X + off_n[:, None] * sxn + tl.arange(0, D1)[None, :] * sxd
    x = tl.load(x_ptrs, mask=load_mask, other=0.0)
    if D2>0:
        x_ptrs2 = X + off_n[:, None] * sxn + tl.arange(0, D2)[None, :] * sxd + D1
        x2 = tl.load(x_ptrs2, mask=load_mask, other=0.0)

    store_mask = (block_idx >= 0) & (block_idx < num_blocks)
    if store_mask:
        tl.store(Y + block_idx * sym + off_k[:, None] * syk + tl.arange(0, D1)[None, :], x)
        if D2 > 0:
            tl.store(Y + block_idx * sym + off_k[:, None] * syk + tl.arange(0, D2)[None, :] + D1, x2)


@triton.jit
def _construct_block_bwd_kernel(
    DX,
    DY,
    X_CU_SEQLENS,
    Y_CU_SEQLENS,
    sxn, sxh, sxd,
    sym, syh, syk, syd,
    stride: tl.constexpr,
    kernel_size,
    ATOMIC: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    start_n = tl.program_id(2) * stride
    off_n = start_n + tl.arange(0, stride)

    x_bos, x_eos = tl.load(X_CU_SEQLENS+off_b), tl.load(X_CU_SEQLENS+off_b+1)
    x_len = x_eos - x_bos

    if start_n >= x_len:
        return

    y_bos, y_eos = tl.load(Y_CU_SEQLENS+off_b), tl.load(Y_CU_SEQLENS+off_b+1)
    num_blocks = y_eos - y_bos

    DX += x_bos * sxn + off_h * sxh
    DY += y_bos * sym + off_h * syh

    block_idx = start_n // stride

    dx = tl.zeros((stride, D1), dtype=tl.float32)
    if D2 > 0:
        dx2 = tl.zeros((stride, D2), dtype=tl.float32)

    off_k = tl.arange(0, stride)
    
    mask = (block_idx >= 0) & (block_idx < num_blocks)
    dy = tl.load(DY + block_idx * sym + off_k[:, None] * syk + tl.arange(0, D1)[None, :], mask)
    dx += dy
    if D2 > 0:
        dy2 = tl.load(DY + block_idx * sym + off_k[:, None] * syk + tl.arange(0, D2)[None, :] + D1, mask)
        dx2 += dy2

    mask = off_n[:, None] < x_len
    if not ATOMIC:
        tl.store(DX + off_n[:, None] * sxn + tl.arange(0, D1)[None, :], dx, mask)
        if D2 > 0:
            tl.store(DX + off_n[:, None] * sxn + tl.arange(0, D2)[None, :] + D1, dx2, mask)
    else:
        tl.atomic_add(DX + off_n[:, None] * sxn + tl.arange(0, D1)[None, :], dx, mask)
        if D2 > 0:
            tl.atomic_add(DX + off_n[:, None] * sxn + tl.arange(0, D2)[None, :] + D1, dx2, mask)


def construct_block_fwd(x, helper=NSAHelper):
    T, H, D = x.shape
    D1, D2 = NSAHelper.split_d(D)
    kernel_size, stride = NSAHelper.kernel_size, NSAHelper.stride
    x_cu_seqlens, y_cu_seqlens = helper.x_cu_seqlens, helper.y_cu_seqlens
    x_maxlen = helper.x_maxlen
    B = len(x_cu_seqlens) - 1

    y_len = NSAHelper.y_len
    y = torch.empty(y_len, H, stride, D, device=x.device, dtype=x.dtype)

    if D<=128:
        kwargs = {'num_warps':2, 'num_stages': 2}
    else:
        kwargs = {'num_warps':8, 'num_stages': 3}
    grids = (B, H, triton.cdiv(x_maxlen, stride))
    _construct_block_fwd_kernel[grids](
        x,
        y,
        x_cu_seqlens,
        y_cu_seqlens,
        *x.stride(),
        *y.stride(),
        stride,
        kernel_size,
        D1,
        D2,
        **kwargs
    )
    return y

def construct_block_bwd(x, dy, dx=None, helper=NSAHelper):
    T, H, D = x.shape
    D1, D2 = NSAHelper.split_d(D)
    kernel_size, stride = NSAHelper.kernel_size, NSAHelper.stride
    x_cu_seqlens, y_cu_seqlens, x_maxlen = helper.x_cu_seqlens, helper.y_cu_seqlens, helper.x_maxlen
    B = len(x_cu_seqlens) - 1

    if dx is None:
        atomic = False
        dx = torch.empty_like(x)
    else:
        atomic = True

    if D <= 128:
        kwargs = {'num_warps':1, 'num_stages': 3}
    else:
        kwargs = {'num_warps':8, 'num_stages': 4}
    grids = (B, H, triton.cdiv(x_maxlen, stride))
    _construct_block_bwd_kernel[grids](                    
        dx,
        dy,
        x_cu_seqlens,
        y_cu_seqlens,
        *dx.stride(),
        *dy.stride(),
        stride,
        kernel_size,
        atomic,
        D1,
        D2,
        **kwargs
    )
    return dx

class _ConstructBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                )-> Tuple[torch.Tensor, torch.Tensor]:
        y = construct_block_fwd(x)
        ctx.save_for_backward(x,)
        ctx.bwd_helper = NSAHelper.get_bwd_helper()
        return y

    @staticmethod
    def backward(ctx, dy):
        x,  = ctx.saved_tensors
        dx = construct_block_bwd(x, dy, helper=ctx.bwd_helper)
        return dx

def construct_block(x:torch.Tensor) -> torch.Tensor:
    '''
    Transform x to non-causal block representation

    Args:
        x (torch.Tensor): [t, h, d]
    Returns:
        y (torch.Tensor): [total_num_blocks, h, block_size, d]
    '''
    return _ConstructBlock.apply(x)



@triton.jit
def _mean_pooling_fwd_kernel(
    X,
    Y,
    X_CU_SEQLENS,
    Y_CU_SEQLENS,
    sxn, sxh, sxd,
    sym, syh, syd,
    stride:tl.constexpr,
    kernel_size:tl.constexpr,
    D1:tl.constexpr,
    D2:tl.constexpr,
    BLOCK_K: tl.constexpr
):
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)
    off_m = tl.program_id(0)
    y_bos, y_eos = tl.load(Y_CU_SEQLENS+off_b), tl.load(Y_CU_SEQLENS+off_b+1)
    y_len = y_eos - y_bos
    if off_m >= y_len:
        return

    x_bos = tl.load(X_CU_SEQLENS+off_b)

    X += x_bos * sxn + off_h * sxh
    Y += y_bos * sym + off_h * syh

    center_pos = off_m * stride + kernel_size // 2
    start_pos = center_pos - kernel_size // 2
    
    off_n = start_pos + tl.arange(0, BLOCK_K)
    mask = (tl.arange(0, BLOCK_K) < kernel_size) & (off_n >= 0) & (off_n < (tl.load(X_CU_SEQLENS+off_b+1) - x_bos))
    x = tl.load(X + off_n[:, None] * sxn + tl.arange(0, D1)[None, :] * sxd, mask=mask[:, None]).to(tl.float32)

    y = tl.sum(x, 0) / kernel_size
    tl.store(Y + off_m * sym + tl.arange(0, D1), y)
    if D2 > 0:
        x2 = tl.load(X + off_n[:, None] * sxn + (tl.arange(0, D2)[None, :] + D1) * sxd, mask=mask[:, None]).to(tl.float32)
        y2 = tl.sum(x2, 0) / kernel_size
        tl.store(Y + off_m * sym + tl.arange(0, D2) + D1, y2)


@triton.jit
def _mean_pooling_bwd_kernel(
    DX,
    DY,
    X_CU_SEQLENS,
    Y_CU_SEQLENS,
    sxn, sxh, sxd,
    sym, syh, syd,
    stride: tl.constexpr,
    kernel_size,
    ATOMIC: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
):
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)
    start_n = tl.program_id(0) * stride
    off_n = start_n + tl.arange(0, stride)

    x_bos, x_eos = tl.load(X_CU_SEQLENS+off_b), tl.load(X_CU_SEQLENS+off_b+1)
    x_len = x_eos - x_bos

    if start_n >= x_len:
        return

    y_bos, y_eos = tl.load(Y_CU_SEQLENS+off_b), tl.load(Y_CU_SEQLENS+off_b+1)
    y_len = y_eos - y_bos

    DX += x_bos * sxn + off_h * sxh
    DY += y_bos * sym + off_h * syh

    half_kernel = kernel_size // 2
    
    block_start_idx = tl.maximum(0, (start_n - kernel_size + 1 + stride - 1) // stride)
    block_end_idx = tl.minimum(y_len - 1, (start_n + stride - 1) // stride)

    dx = tl.zeros((stride, D1), dtype=tl.float32)
    if D2 > 0:
        dx2 = tl.zeros((stride, D2), dtype=tl.float32)

    for block_idx in range(block_start_idx, block_end_idx + 1):
        if block_idx >= 0 and block_idx < y_len:
            block_center = block_idx * stride + half_kernel
            block_start = block_center - half_kernel
            block_end = block_start + kernel_size
            
            overlap_start = tl.maximum(block_start, start_n)
            overlap_end = tl.minimum(block_end, start_n + stride)
            
            if overlap_start < overlap_end:
                mask = block_idx < y_len
                dy = tl.load(DY + block_idx * sym + tl.arange(0, D1), mask)
                dx += dy[None, :]
                if D2 > 0:
                    dy2 = tl.load(DY + block_idx * sym + tl.arange(0, D2) + D1, mask)
                    dx2 += dy2[None, :]

    dx /= kernel_size
    if D2 > 0:
        dx2 /= kernel_size

    mask = off_n[:, None] < x_len
    if not ATOMIC:
        tl.store(DX + off_n[:, None] * sxn + tl.arange(0, D1)[None, :], dx, mask)
        if D2 > 0:
            tl.store(DX + off_n[:, None] * sxn + tl.arange(0, D2)[None, :] + D1, dx2, mask)
    else:
        tl.atomic_add(DX + off_n[:, None] * sxn + tl.arange(0, D1)[None, :], dx, mask)
        if D2 > 0:
            tl.atomic_add(DX + off_n[:, None] * sxn + tl.arange(0, D2)[None, :] + D1, dx2, mask)


def mean_pooling_fwd(x, helper=NSAHelper):
    T, H, D = x.shape
    D1, D2 = NSAHelper.split_d(D)
    kernel_size, stride = NSAHelper.kernel_size, NSAHelper.stride
    x_cu_seqlens, y_cu_seqlens = helper.x_cu_seqlens, helper.y_cu_seqlens
    y_maxlen = helper.y_maxlen
    B = len(x_cu_seqlens) - 1

    BLOCK_K = triton.next_power_of_2(kernel_size)

    y_len = NSAHelper.y_len
    y = torch.empty(y_len, H, D, device=x.device, dtype=x.dtype)
    if D<=128:
        kwargs = {'num_warps':1, 'num_stages': 1}
    else:
        kwargs = {'num_warps':1, 'num_stages': 1}
    grids = (y_maxlen, B, H)
    _mean_pooling_fwd_kernel[grids](
        x,
        y,
        x_cu_seqlens,
        y_cu_seqlens,
        *x.stride(),
        *y.stride(),
        stride,
        kernel_size,
        D1,
        D2,
        BLOCK_K,
        **kwargs
    )
    return y


def mean_pooling_bwd(x, dy, dx=None, helper=NSAHelper):
    T, H, D = x.shape
    D1, D2 = NSAHelper.split_d(D)
    kernel_size, stride = NSAHelper.kernel_size, NSAHelper.stride
    x_cu_seqlens, y_cu_seqlens, x_maxlen = helper.x_cu_seqlens, helper.y_cu_seqlens, helper.x_maxlen
    B = len(x_cu_seqlens) - 1
    if dx is None:
        dx = torch.empty_like(x)
        atomic = False
    else:
        atomic = True

    grids = (triton.cdiv(x_maxlen, stride), B, H)
    if D <= 128:
        kwargs = {'num_warps':4, 'num_stages': 3}
    else:
        kwargs = {'num_warps':4, 'num_stages': 1}
    _mean_pooling_bwd_kernel[grids](
        dx,
        dy,
        x_cu_seqlens,
        y_cu_seqlens,
        *dx.stride(),
        *dy.stride(),
        stride,
        kernel_size,
        atomic,
        D1,
        D2,
        **kwargs
    )
    return dx


class _MeanPooling(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                )-> Tuple[torch.Tensor, torch.Tensor]:
        y = mean_pooling_fwd(x)
        ctx.save_for_backward(x,)
        ctx.helper = NSAHelper.get_bwd_helper()
        return y

    @staticmethod
    def backward(ctx, dy):
        x,  = ctx.saved_tensors
        dx = mean_pooling_bwd(x, dy, helper=ctx.helper)
        return dx


def mean_pooling(x: torch.Tensor) -> torch.Tensor:
    '''
    Transform x to cmp_x by mean pooling (non-causal for ViT)

    Args:
        x (torch.Tensor): [t, h, d]
    Returns:
        y (torch.Tensor): [total_num_blocks, h, d]
    '''
    return _MeanPooling.apply(x)
