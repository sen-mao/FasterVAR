# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""Training GANs with DiffAugment."""

import numpy as np
import torch
import torch.nn.functional as F


def DiffAugment(x: torch.Tensor, policy: str = '', channels_first: bool = True) -> torch.Tensor:
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x: torch.Tensor, ratio: float = 0.2) -> torch.Tensor:
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


def rand_resize(x: torch.Tensor, min_ratio: float = 0.8, max_ratio: float = 1.2) -> torch.Tensor:
    resize_ratio = np.random.rand()*(max_ratio-min_ratio) + min_ratio
    resized_img = F.interpolate(x, size=int(resize_ratio*x.shape[3]), mode='bilinear')
    org_size = x.shape[3]
    if int(resize_ratio*x.shape[3]) < x.shape[3]:
        left_pad = (x.shape[3]-int(resize_ratio*x.shape[3]))/2.
        left_pad = int(left_pad)
        right_pad = x.shape[3] - left_pad - resized_img.shape[3]
        x = F.pad(resized_img, (left_pad, right_pad, left_pad, right_pad), "constant", 0.)
    else:
        left = (int(resize_ratio*x.shape[3])-x.shape[3])/2.
        left = int(left)
        x = resized_img[:, :, left:(left+x.shape[3]), left:(left+x.shape[3])]
    assert x.shape[2] == org_size
    assert x.shape[3] == org_size
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'resize': [rand_resize],
    'cutout': [rand_cutout],
}