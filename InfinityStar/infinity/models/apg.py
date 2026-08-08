# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import math
import torch


def project(
    v0: torch.Tensor, # [B, seq_len, dim]
    v1: torch.Tensor, # [B, seq_len, dim]
):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1,-2])
    v0_parallel = (v0 * v1).sum(dim=[-1,-2], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def normalized_guidance(
    pred_cond: torch. Tensor, # [B, seq_len, dim]
    pred_uncond: torch.Tensor, # [B, seq_len, dim]
    guidance_scale: float,
    momentum_buffer: None,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
):
    B, seq_len, dim = pred_cond.shape
    diff = pred_cond - pred_uncond
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = 1/math.sqrt(seq_len*dim) * diff.norm(p=2, dim=[-1, -2], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    return pred_guided