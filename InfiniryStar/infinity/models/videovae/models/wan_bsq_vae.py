# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

from typing import Dict, Optional, Tuple, Union
import math
import numpy as np
from einops import rearrange
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from infinity.models.videovae.modules import DiagonalGaussianDistribution
from infinity.models.videovae.utils.misc import ptdtype
from infinity.models.videovae.modules.quantizer import MultiScaleBSQTP_AP as MultiScaleBSQTP_AP
from infinity.models.videovae.modules.quantizer import MultiScaleFSQTP
from infinity.models.videovae.modules.conv_wan import DCDownBlock2d, DCUpBlock2d, DCDownBlock3d, DCUpBlock3d, CogVideoXCausalConv3d, CogVideoXSafeConv3d
from infinity.models.videovae.modules.normalization_wan import get_norm
from infinity.models.videovae.utils.context_parallel import ContextParallelUtils as cp
from infinity.models.videovae.utils.context_parallel import dist_decoder_gather_result, dist_encoder_gather_result
from infinity.models.videovae.utils.dynamic_resolution_two_pyramid import get_ratio2hws_video_v2


def patchify(item):
    assert item.ndim == 5
    # (B,c,t,H,W) -> (B,t,c,H,W) -> (B,t,4c,H/2,W/2) -> (B,4c,t,H/2,W/2)
    item = torch.nn.functional.pixel_unshuffle(item.permute(0,2,1,3,4), 2).permute(0,2,1,3,4)
    return item

def unpatchify(item):
    assert item.ndim == 5
    item = item.permute(0,2,1,3,4) # (B,4c,t,H/2,W/2) -> [B, t, 4c, H/2, W/2]
    item = torch.nn.functional.pixel_shuffle(item, 2) # [B, t, 4c, H/2, W/2] -> [B, t, c, H, W]
    item = item.permute(0,2,1,3,4) # [B, t, c, H, W] -> [B, c, t, H, W]
    return item

class CogVideoXDownsample3D(nn.Module):
    # Todo: Wait for paper relase.
    r"""
    A 3D Downsampling layer using in [CogVideoX]() by Tsinghua University & ZhipuAI

    Args:
        in_channels (`int`):
            Number of channels in the input image.
        out_channels (`int`):
            Number of channels produced by the convolution.
        kernel_size (`int`, defaults to `3`):
            Size of the convolving kernel.
        stride (`int`, defaults to `2`):
            Stride of the convolution.
        padding (`int`, defaults to `0`):
            Padding added to all four sides of the input.
        compress_time (`bool`, defaults to `False`):
            Whether or not to compress the time dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
        compress_time = None,
        down_layer = "conv",
        down_norm = False,
        pad_mode = "constant", 
        norm_type=None,
    ):
        super().__init__()

        self.pad_mode = pad_mode
        self.down_layer = down_layer
        if down_layer == "conv":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        elif down_layer == "dc":
            self.conv = DCDownBlock2d(in_channels, out_channels, downsample=True, shortcut=True, pad_mode=pad_mode, group_norm=down_norm)
        elif down_layer == "3d-dc":
            self.conv = DCDownBlock3d(in_channels, out_channels, group_norm=down_norm, compress_time=compress_time, pad_mode=pad_mode, norm_type=norm_type)
        self.compress_time = compress_time

    def forward(self, x: torch.Tensor, conv_cache: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        if self.down_layer == "3d-dc":
            x, new_conv_cache = self.conv(x, conv_cache=conv_cache)
        else:
            if self.compress_time == 2:
                batch_size, channels, frames, height, width = x.shape

                # (batch_size, channels, frames, height, width) -> (batch_size, height, width, channels, frames) -> (batch_size * height * width, channels, frames)
                x = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, channels, frames)

                if x.shape[-1] % 2 == 1:
                    x_first, x_rest = x[..., 0], x[..., 1:]
                    if x_rest.shape[-1] > 0:
                        # (batch_size * height * width, channels, frames - 1) -> (batch_size * height * width, channels, (frames - 1) // 2)
                        x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)

                    x = torch.cat([x_first[..., None], x_rest], dim=-1)
                    # (batch_size * height * width, channels, (frames // 2) + 1) -> (batch_size, height, width, channels, (frames // 2) + 1) -> (batch_size, channels, (frames // 2) + 1, height, width)
                    x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)
                else:
                    # (batch_size * height * width, channels, frames) -> (batch_size * height * width, channels, frames // 2)
                    x = F.avg_pool1d(x, kernel_size=2, stride=2)
                    # (batch_size * height * width, channels, frames // 2) -> (batch_size, height, width, channels, frames // 2) -> (batch_size, channels, frames // 2, height, width)
                    x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)
            elif self.compress_time == 3:
                batch_size, channels, frames, height, width = x.shape
                x = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, channels, frames)

                if x.shape[-1] % 2 == 1:
                    x_first, x_rest = x[..., 0], x[..., 1:]
                    if x_rest.shape[-1] > 0:
                        x_rest = F.avg_pool1d(x_rest, kernel_size=3, stride=3)

                    x = torch.cat([x_first[..., None], x_rest], dim=-1)
                    # (batch_size * height * width, channels, (frames // 2) + 1) -> (batch_size, height, width, channels, (frames // 2) + 1) -> (batch_size, channels, (frames // 2) + 1, height, width)
                    x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)
                else:
                    # (batch_size * height * width, channels, frames) -> (batch_size * height * width, channels, frames // 2)
                    x = F.avg_pool1d(x, kernel_size=3, stride=3)
                    # (batch_size * height * width, channels, frames // 2) -> (batch_size, height, width, channels, frames // 2) -> (batch_size, channels, frames // 2, height, width)
                    x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)

            # Pad the tensor
            if self.down_layer == "conv":
                pad = (0, 1, 0, 1)
                if self.pad_mode == "constant":
                    x = F.pad(x, pad, mode="constant", value=0)
                else:
                    _shape = x.shape
                    x = F.pad(x, pad, mode="replicate")
                    inputs = inputs.view(*_shape[:-2], *inputs.shape[-2:])
                
            batch_size, channels, frames, height, width = x.shape
            # (batch_size, channels, frames, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size * frames, channels, height, width)
            x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
            x = self.conv(x)
            # (batch_size * frames, channels, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size, channels, frames, height, width)
            x = x.reshape(batch_size, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        return x, new_conv_cache


class CogVideoXUpsample3D(nn.Module):
    r"""
    A 3D Upsample layer using in CogVideoX by Tsinghua University & ZhipuAI # Todo: Wait for paper relase.

    Args:
        in_channels (`int`):
            Number of channels in the input image.
        out_channels (`int`):
            Number of channels produced by the convolution.
        kernel_size (`int`, defaults to `3`):
            Size of the convolving kernel.
        stride (`int`, defaults to `1`):
            Stride of the convolution.
        padding (`int`, defaults to `1`):
            Padding added to all four sides of the input.
        compress_time (`bool`, defaults to `False`):
            Whether or not to compress the time dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        compress_time = None,
        up_layer = "conv",
        up_norm = False,
        norm_type = None,
        pad_mode = "constant",
    ) -> None:
        super().__init__()

        self.up_layer = up_layer
        if up_layer == "conv":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        elif up_layer == "dc":
            self.conv = DCUpBlock2d(in_channels, out_channels, interpolate=False, shortcut=True, group_norm=up_norm, norm_type=norm_type, pad_mode=pad_mode)
        elif up_layer == "3d-dc":
            self.conv = DCUpBlock3d(in_channels, out_channels, group_norm=up_norm, compress_time=compress_time, norm_type=norm_type, pad_mode=pad_mode)
        self.compress_time = compress_time

    def forward(self, inputs: torch.Tensor, conv_cache: Optional[Dict[str, torch.Tensor]] = None, split_first=False) -> torch.Tensor:
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        if self.up_layer == "3d-dc":
            inputs, new_conv_cache = self.conv(inputs, conv_cache=conv_cache, split_first=split_first)
        else:
            raise NotImplementedError
            if self.up_layer == "conv":
                spatial_scale = (2., 2.)
            elif self.up_layer == "dc":
                spatial_scale = (1., 1.)
            if self.compress_time:
                temporal_scale = (float(self.compress_time), *spatial_scale)
                if inputs.shape[2] > 1 and inputs.shape[2] % 2 == 1:
                    # split first frame
                    x_first, x_rest = inputs[:, :, 0], inputs[:, :, 1:]
                    x_first = F.interpolate(x_first, scale_factor=spatial_scale)
                    x_rest = F.interpolate(x_rest, scale_factor=temporal_scale)
                    x_first = x_first[:, :, None, :, :]
                    inputs = torch.cat([x_first, x_rest], dim=2)
                elif inputs.shape[2] > 1:
                    inputs = F.interpolate(inputs, scale_factor=temporal_scale)
                else:
                    inputs = inputs.squeeze(2)
                    inputs = F.interpolate(inputs, scale_factor=spatial_scale)
                    inputs = inputs[:, :, None, :, :]
            else:
                # only interpolate 2D
                b, c, t, h, w = inputs.shape
                inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                inputs = F.interpolate(inputs, scale_factor=spatial_scale)
                inputs = inputs.reshape(b, t, c, *inputs.shape[2:]).permute(0, 2, 1, 3, 4)

            b, c, t, h, w = inputs.shape
            inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            inputs = self.conv(inputs)
            inputs = inputs.reshape(b, t, *inputs.shape[1:]).permute(0, 2, 1, 3, 4)
        return inputs, new_conv_cache

class CogVideoXSpatialNorm3D(nn.Module):
    r"""
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002. This implementation is specific
    to 3D-video like data.

    CogVideoXSafeConv3d is used instead of nn.Conv3d to avoid OOM in CogVideoX Model.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
        groups (`int`):
            Number of groups to separate the channels into for group normalization.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        groups: int = 32,
        norm_type = None,
        pad_mode = "constant"
    ):
        super().__init__()
        norm_layer = get_norm(norm_type)
        self.norm_layer = norm_layer(num_channels=f_channels, num_groups=groups, eps=1e-6, affine=True)
        self.conv_y = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1, pad_mode=pad_mode)
        self.conv_b = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1, pad_mode=pad_mode)

    def forward(
        self, f: torch.Tensor, zq: torch.Tensor, conv_cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        if f.shape[2] > 1 and f.shape[2] % 2 == 1:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            z_first, z_rest = zq[:, :, :1], zq[:, :, 1:]
            z_first = F.interpolate(z_first, size=f_first_size)
            z_rest = F.interpolate(z_rest, size=f_rest_size)
            zq = torch.cat([z_first, z_rest], dim=2)
        else:
            zq = F.interpolate(zq, size=f.shape[-3:])

        conv_y, new_conv_cache["conv_y"] = self.conv_y(zq, conv_cache=conv_cache.get("conv_y"))
        conv_b, new_conv_cache["conv_b"] = self.conv_b(zq, conv_cache=conv_cache.get("conv_b"))

        norm_f = self.norm_layer(f)
        new_f = norm_f * conv_y + conv_b
        return new_f, new_conv_cache


class CogVideoXResnetBlock3D(nn.Module):
    r"""
    A 3D ResNet block used in the CogVideoX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        conv_shortcut (bool, defaults to `False`):
            Whether or not to use a convolution shortcut.
        spatial_norm_dim (`int`, *optional*):
            The dimension to use for spatial norm if it is to be used instead of group norm.
        pad_mode (str, defaults to `"constant"`):
            Padding mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        conv_shortcut: bool = False,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "constant",
        norm_type = None,
    ):
        super().__init__()
        norm_layer = get_norm(norm_type)
        out_channels = out_channels or in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = nn.SiLU()
        self.use_conv_shortcut = conv_shortcut
        self.spatial_norm_dim = spatial_norm_dim

        if spatial_norm_dim is None:
            self.norm1 = norm_layer(num_channels=in_channels, num_groups=groups, eps=eps)
            self.norm2 = norm_layer(num_channels=out_channels, num_groups=groups, eps=eps)
        else:
            self.norm1 = CogVideoXSpatialNorm3D(
                f_channels=in_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
                norm_type=norm_type,
                pad_mode=pad_mode,
            )
            self.norm2 = CogVideoXSpatialNorm3D(
                f_channels=out_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
                norm_type=norm_type,
                pad_mode=pad_mode,
            )

        self.conv1 = CogVideoXCausalConv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
        )

        if temb_channels > 0:
            self.temb_proj = nn.Linear(in_features=temb_channels, out_features=out_channels)

        self.dropout = nn.Dropout(dropout)
        self.conv2 = CogVideoXCausalConv3d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CogVideoXCausalConv3d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode
                )
            else:
                self.conv_shortcut = CogVideoXSafeConv3d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(
        self,
        inputs: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states = inputs

        if zq is not None:
            hidden_states, new_conv_cache["norm1"] = self.norm1(hidden_states, zq, conv_cache=conv_cache.get("norm1"))
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states, new_conv_cache["conv1"] = self.conv1(hidden_states, conv_cache=conv_cache.get("conv1"))

        if temb is not None:
            hidden_states = hidden_states + self.temb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if zq is not None:
            hidden_states, new_conv_cache["norm2"] = self.norm2(hidden_states, zq, conv_cache=conv_cache.get("norm2"))
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states, new_conv_cache["conv2"] = self.conv2(hidden_states, conv_cache=conv_cache.get("conv2"))

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                inputs, new_conv_cache["conv_shortcut"] = self.conv_shortcut(
                    inputs, conv_cache=conv_cache.get("conv_shortcut")
                )
            else:
                inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states, new_conv_cache


class CogVideoXDownBlock3D(nn.Module):
    r"""
    A downsampling block used in the CogVideoX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        add_downsample (`bool`, defaults to `True`):
            Whether or not to use a downsampling layer. If not used, output dimension would be same as input dimension.
        compress_time (`bool`, defaults to `False`):
            Whether or not to downsample across temporal dimension.
        pad_mode (str, defaults to `"constant"`):
            Padding mode.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 0,
        compress_time = None,
        compress_spatial = None,
        pad_mode: str = "constant",
        norm_type = None,
        down_layer = "conv",
        down_block_mode = "cogvideox",
        down_norm = False,
    ):
        super().__init__()

        if down_block_mode == "cogvideox":
            resnets = []
            for i in range(num_layers):
                in_channel = in_channels if i == 0 else out_channels
                resnets.append(
                    CogVideoXResnetBlock3D(
                        in_channels=in_channel,
                        out_channels=out_channels,
                        dropout=dropout,
                        temb_channels=temb_channels,
                        groups=resnet_groups,
                        eps=resnet_eps,
                        pad_mode=pad_mode,
                        norm_type=norm_type
                    )
                )
            self.resnets = nn.ModuleList(resnets)
            self.downsamplers = None
            if add_downsample:
                self.downsamplers = nn.ModuleList(
                    [
                        CogVideoXDownsample3D(
                            out_channels, out_channels, padding=downsample_padding, compress_time=compress_time, down_layer=down_layer, down_norm=down_norm, pad_mode=pad_mode, norm_type=norm_type,
                        )
                    ]
                )
        elif down_block_mode == "dc":
            resnets = []
            for i in range(num_layers):
                resnets.append(
                    CogVideoXResnetBlock3D(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        dropout=dropout,
                        temb_channels=temb_channels,
                        groups=resnet_groups,
                        eps=resnet_eps,
                        pad_mode=pad_mode,
                        norm_type=norm_type
                    )
                )
            self.resnets = nn.ModuleList(resnets)
            self.downsamplers = None
            if add_downsample:
                self.downsamplers = nn.ModuleList(
                    [
                        CogVideoXDownsample3D(
                            in_channels, out_channels, padding=downsample_padding, compress_time=compress_time, down_layer=down_layer,down_norm=down_norm, pad_mode=pad_mode, norm_type=norm_type,
                        )
                    ]
                )
        else:
            raise NotImplementedError(f"Invalid `down_block_mode` {down_block_mode} encountered. ")

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""Forward method of the `CogVideoXDownBlock3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states, new_conv_cache[conv_cache_key] = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    zq,
                    conv_cache.get(conv_cache_key),
                    use_reentrant=False
                )
            else:
                hidden_states, new_conv_cache[conv_cache_key] = resnet(
                    hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
                )

        if self.downsamplers is not None:
            for i, downsampler in enumerate(self.downsamplers):
                conv_cache_key = f"downsampler_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = downsampler(hidden_states, conv_cache=conv_cache.get(conv_cache_key))

        return hidden_states, new_conv_cache


class CogVideoXMidBlock3D(nn.Module):
    r"""
    A middle block used in the CogVideoX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        spatial_norm_dim (`int`, *optional*):
            The dimension to use for spatial norm if it is to be used instead of group norm.
        pad_mode (str, defaults to `"constant"`):
            Padding mode.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "constant",
        norm_type = None
    ):
        super().__init__()

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    spatial_norm_dim=spatial_norm_dim,
                    pad_mode=pad_mode,
                    norm_type=norm_type,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""Forward method of the `CogVideoXMidBlock3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states, new_conv_cache[conv_cache_key] = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb, zq, conv_cache.get(conv_cache_key), use_reentrant=False
                )
            else:
                hidden_states, new_conv_cache[conv_cache_key] = resnet(
                    hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
                )

        return hidden_states, new_conv_cache


class CogVideoXUpBlock3D(nn.Module):
    r"""
    An upsampling block used in the CogVideoX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        spatial_norm_dim (`int`, defaults to `16`):
            The dimension to use for spatial norm if it is to be used instead of group norm.
        add_upsample (`bool`, defaults to `True`):
            Whether or not to use a upsampling layer. If not used, output dimension would be same as input dimension.
        compress_time (`bool`, defaults to `False`):
            Whether or not to downsample across temporal dimension.
        pad_mode (str, defaults to `"constant"`):
            Padding mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        spatial_norm_dim: int = 16,
        add_upsample: bool = True,
        upsample_padding: int = 1,
        compress_time = None,
        compress_spatial = None,
        pad_mode: str = "constant",
        norm_type = None,
        up_layer = "conv",
        up_block_mode="cogvideox",
        up_norm = False,
    ):
        super().__init__()

        if up_block_mode == "cogvideox":
            resnets = []
            for i in range(num_layers):
                in_channel = in_channels if i == 0 else out_channels
                resnets.append(
                    CogVideoXResnetBlock3D(
                        in_channels=in_channel,
                        out_channels=out_channels,
                        dropout=dropout,
                        temb_channels=temb_channels,
                        groups=resnet_groups,
                        eps=resnet_eps,
                        spatial_norm_dim=spatial_norm_dim,
                        pad_mode=pad_mode,
                        norm_type=norm_type,
                    )
                )
            self.resnets = nn.ModuleList(resnets)
            self.upsamplers = None
            if add_upsample:
                self.upsamplers = nn.ModuleList(
                    [
                        CogVideoXUpsample3D(
                            out_channels, out_channels, padding=upsample_padding, compress_time=compress_time, up_layer=up_layer, up_norm=up_norm, norm_type=norm_type, pad_mode=pad_mode
                        )
                    ]
                )
        elif up_block_mode == "dc":
            resnets = []
            for i in range(num_layers):
                resnets.append(
                    CogVideoXResnetBlock3D(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        dropout=dropout,
                        temb_channels=temb_channels,
                        groups=resnet_groups,
                        eps=resnet_eps,
                        spatial_norm_dim=spatial_norm_dim,
                        pad_mode=pad_mode,
                        norm_type=norm_type,
                    )
                )
            self.resnets = nn.ModuleList(resnets)
            self.upsamplers = None
            if add_upsample:
                self.upsamplers = nn.ModuleList(
                    [
                        CogVideoXUpsample3D(
                            in_channels, out_channels, padding=upsample_padding, compress_time=compress_time, up_layer=up_layer, up_norm=up_norm, norm_type=norm_type, pad_mode=pad_mode
                        )
                    ]
                )
        else:
            raise NotImplementedError(f"Invalid `up_block_mode` {up_block_mode} encountered. ")

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
        split_first = False,
    ) -> torch.Tensor:
        r"""Forward method of the `CogVideoXUpBlock3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states, new_conv_cache[conv_cache_key] = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    zq,
                    conv_cache.get(conv_cache_key),
                    use_reentrant=False
                )
            else:
                hidden_states, new_conv_cache[conv_cache_key] = resnet(
                    hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
                )

        if self.upsamplers is not None:
            for i, upsampler in enumerate(self.upsamplers):
                conv_cache_key = f"upsampler_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = upsampler(hidden_states, conv_cache=conv_cache.get(conv_cache_key), split_first=split_first)

        return hidden_states, new_conv_cache


class CogVideoXEncoder3D(nn.Module):
    _supports_gradient_checkpointing = True
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        down_block_types: Tuple[str, ...] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "constant",
        temporal_compression_list: list = [],
        spatial_compression_list: list = [],
        norm_type=None,
        down_layer = "conv",
        down_block_mode = "cogvideox",
        down_norm=False,
    ):
        super().__init__()

        norm_layer = get_norm(norm_type)
        # log2 of temporal_compress_times
        # temporal_compress_level = int(np.log2(temporal_compression_ratio))

        self.conv_in = CogVideoXCausalConv3d(in_channels, block_out_channels[0], kernel_size=3, pad_mode=pad_mode)

        self.down_blocks = nn.ModuleList([])

        # down blocks
        for i, down_block_type in enumerate(down_block_types):
            input_channel = block_out_channels[i]
            output_channel = block_out_channels[i+1]
            compress_time = temporal_compression_list[i] if i < len(temporal_compression_list) else None
            compress_spatial = spatial_compression_list[i] if i < len(spatial_compression_list) else None

            if down_block_type == "CogVideoXDownBlock3D":
                down_block = CogVideoXDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    dropout=dropout,
                    num_layers=layers_per_block,
                    resnet_eps=norm_eps,
                    resnet_groups=norm_num_groups,
                    add_downsample=compress_time or compress_spatial,
                    compress_time=compress_time,
                    compress_spatial=compress_spatial,
                    pad_mode=pad_mode,
                    norm_type=norm_type,
                    down_layer=down_layer,
                    down_block_mode=down_block_mode,
                    down_norm=down_norm,
                )
            else:
                raise ValueError("Invalid `down_block_type` encountered. Must be `CogVideoXDownBlock3D`")

            self.down_blocks.append(down_block)

        # mid block
        self.mid_block = CogVideoXMidBlock3D(
            in_channels=block_out_channels[len(down_block_types)],
            temb_channels=0,
            dropout=dropout,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_groups=norm_num_groups,
            pad_mode=pad_mode,
            norm_type=norm_type,
        )

        self.norm_out = norm_layer(num_channels=block_out_channels[len(down_block_types)], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = CogVideoXCausalConv3d(
            block_out_channels[len(down_block_types)], 2 * out_channels, kernel_size=3, pad_mode=pad_mode
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        sample: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""The forward method of the `CogVideoXEncoder3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states, new_conv_cache["conv_in"] = self.conv_in(sample, conv_cache=conv_cache.get("conv_in"))

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # 1. Down
            for i, down_block in enumerate(self.down_blocks):
                conv_cache_key = f"down_block_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block),
                    hidden_states,
                    temb,
                    None,
                    conv_cache.get(conv_cache_key),
                    use_reentrant=False
                )

            # 2. Mid
            hidden_states, new_conv_cache["mid_block"] = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block),
                hidden_states,
                temb,
                None,
                conv_cache.get("mid_block"),
                use_reentrant=False
            )
        else:
            # 1. Down
            for i, down_block in enumerate(self.down_blocks):
                conv_cache_key = f"down_block_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = down_block(
                    hidden_states, temb, None, conv_cache=conv_cache.get(conv_cache_key)
                )

            # 2. Mid
            hidden_states, new_conv_cache["mid_block"] = self.mid_block(
                hidden_states, temb, None, conv_cache=conv_cache.get("mid_block")
            )

        # 3. Post-process
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)

        hidden_states, new_conv_cache["conv_out"] = self.conv_out(hidden_states, conv_cache=conv_cache.get("conv_out"))

        return hidden_states, new_conv_cache


class CogVideoXDecoder3D(nn.Module):
    r"""
    The `CogVideoXDecoder3D` layer of a variational autoencoder that decodes its latent representation into an output
    sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "constant",
        temporal_compression_list: list = [],
        spatial_compression_list: list = [],
        norm_type=None,
        up_layer="conv",
        up_block_mode="cogvideox",
        up_norm=False,
    ):
        super().__init__()

        reversed_block_out_channels = list(reversed(block_out_channels))

        self.conv_in = CogVideoXCausalConv3d(
            in_channels, reversed_block_out_channels[0], kernel_size=3, pad_mode=pad_mode
        )

        # mid block
        self.mid_block = CogVideoXMidBlock3D(
            in_channels=reversed_block_out_channels[0],
            temb_channels=0,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_groups=norm_num_groups,
            spatial_norm_dim=in_channels,
            pad_mode=pad_mode,
            norm_type=norm_type,
        )

        # up blocks
        self.up_blocks = nn.ModuleList([])

        # output_channel = reversed_block_out_channels[0]
        # temporal_compress_level = int(np.log2(temporal_compression_ratio))

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = reversed_block_out_channels[i]
            output_channel = reversed_block_out_channels[i+1]
            if up_block_mode == "cogvideox":
                raise NotImplementedError
                is_final_block = i == len(up_block_types) - 1
                compress_time = temporal_compression_list[i] if i < len(temporal_compression_list) else None
                compress_spatial = spatial_compression_list[i] if i < len(spatial_compression_list) else None
            elif up_block_mode == "dc":
                # is_final_block = i == 0
                idx_temporal = i - (len(up_block_types) - len(temporal_compression_list))
                compress_time = temporal_compression_list[-idx_temporal] if idx_temporal >= 0 else None
                idx_spatial = i - (len(up_block_types) - len(spatial_compression_list))
                compress_spatial = spatial_compression_list[-idx_spatial] if idx_spatial >= 0 else None
                # print(temporal_compression_list, idx_temporal, compress_time, spatial_compression_list, idx_spatial, compress_spatial, compress_time or compress_spatial)
            
            if up_block_type == "CogVideoXUpBlock3D":
                up_block = CogVideoXUpBlock3D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    dropout=dropout,
                    num_layers=layers_per_block + 1,
                    resnet_eps=norm_eps,
                    resnet_groups=norm_num_groups,
                    spatial_norm_dim=in_channels,
                    add_upsample=compress_time or compress_spatial,
                    compress_time=compress_time,
                    compress_spatial=compress_spatial,
                    pad_mode=pad_mode,
                    norm_type=norm_type,
                    up_layer=up_layer,
                    up_block_mode=up_block_mode,
                    up_norm=up_norm,
                )
                prev_output_channel = output_channel
            else:
                raise ValueError("Invalid `up_block_type` encountered. Must be `CogVideoXUpBlock3D`")

            self.up_blocks.append(up_block)

        self.norm_out = CogVideoXSpatialNorm3D(reversed_block_out_channels[len(up_block_types)], in_channels, groups=norm_num_groups, norm_type=norm_type, pad_mode=pad_mode)
        self.conv_act = nn.SiLU()
        self.conv_out = CogVideoXCausalConv3d(
            reversed_block_out_channels[len(up_block_types)], out_channels, kernel_size=3, pad_mode=pad_mode
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        sample: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
        split_first = False,
    ) -> torch.Tensor:
        r"""The forward method of the `CogVideoXDecoder3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states, new_conv_cache["conv_in"] = self.conv_in(sample, conv_cache=conv_cache.get("conv_in"))

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # 1. Mid
            hidden_states, new_conv_cache["mid_block"] = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block),
                hidden_states,
                temb,
                sample,
                conv_cache.get("mid_block"),
                use_reentrant=False
            )

            # 2. Up
            for i, up_block in enumerate(self.up_blocks):
                conv_cache_key = f"up_block_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(up_block),
                    hidden_states,
                    temb,
                    sample,
                    conv_cache.get(conv_cache_key),
                    split_first, 
                    use_reentrant=False
                )
        else:
            # 1. Mid
            hidden_states, new_conv_cache["mid_block"] = self.mid_block(
                hidden_states, temb, sample, conv_cache=conv_cache.get("mid_block")
            )

            # 2. Up
            for i, up_block in enumerate(self.up_blocks):
                conv_cache_key = f"up_block_{i}"
                hidden_states, new_conv_cache[conv_cache_key] = up_block(
                    hidden_states, temb, sample, conv_cache=conv_cache.get(conv_cache_key), split_first=split_first
                )

        # 3. Post-process
        hidden_states, new_conv_cache["norm_out"] = self.norm_out(
            hidden_states, sample, conv_cache=conv_cache.get("norm_out")
        )
        hidden_states = self.conv_act(hidden_states)
        hidden_states, new_conv_cache["conv_out"] = self.conv_out(hidden_states, conv_cache=conv_cache.get("conv_out"))

        return hidden_states, new_conv_cache


class AutoencoderKLCogVideoX(nn.Module):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["CogVideoXResnetBlock3D"]

    def __init__(
        self,
        args
    ):
        super().__init__()
        self.args = args
        self.embed_dim = args.latent_channels
        self.encoder_dtype = ptdtype[args.encoder_dtype]
        self.decoder_dtype = ptdtype[args.decoder_dtype]

        self.encoder = CogVideoXEncoder3D(
            in_channels=args.in_channels,
            out_channels=args.latent_channels,
            down_block_types=args.down_block_types,
            block_out_channels=args.block_out_channels,
            layers_per_block=args.layers_per_block,
            act_fn=args.act_fn,
            norm_eps=args.norm_eps,
            norm_num_groups=args.norm_num_groups,
            temporal_compression_list=args.temporal_compression_list,
            spatial_compression_list=args.spatial_compression_list,
            pad_mode=args.pad_mode,
            norm_type=args.norm_type,
            down_layer=args.down_layer,
            down_block_mode=args.down_block_mode,
            down_norm=args.down_norm,
        )
        self.decoder = CogVideoXDecoder3D(
            in_channels=args.latent_channels,
            out_channels=args.out_channels,
            up_block_types=args.up_block_types,
            block_out_channels=args.block_out_channels,
            layers_per_block=args.layers_per_block,
            act_fn=args.act_fn,
            norm_eps=args.norm_eps,
            norm_num_groups=args.norm_num_groups,
            temporal_compression_list=args.temporal_compression_list,
            spatial_compression_list=args.spatial_compression_list,
            pad_mode=args.pad_mode,
            norm_type=args.norm_type,
            up_layer=args.up_layer,
            up_block_mode=args.up_block_mode,
            up_norm=args.up_norm,
        )
        self.dropout_z_layer = nn.Dropout(p=args.dropout_z)
        if args.use_checkpoint:
            self._set_gradient_checkpointing(self.encoder, True)
            self._set_gradient_checkpointing(self.decoder, True)
        
        if args.fix_model != ["no"]:
            for _model in args.fix_model:
                if _model == "encoder":
                    self._set_no_grad(self.encoder)
                elif _model == "decoder":
                    self._set_no_grad(self.decoder)
                elif _model.startswith("down_blocks"):
                    fix_block_num = int(_model.split("_")[2])
                    self._set_no_grad(self.encoder.conv_in)
                    for idx in range(fix_block_num):
                        self._set_no_grad(self.encoder.down_blocks[idx])
                elif _model.startswith("up_blocks"):
                    fix_block_num = int(_model.split("_")[2])
                    self._set_no_grad(self.decoder.conv_out)
                    self._set_no_grad(self.decoder.norm_out)
                    for idx in range(fix_block_num):
                        total_num = len(self.decoder.up_blocks)
                        self._set_no_grad(self.decoder.up_blocks[total_num - idx - 1]) # reverse fix
                else:
                    raise NotImplementedError

            print("Learnable Parameters:")
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(name)
        
        # for down_block in self.encoder.down_blocks:
        #     if down_block.downsamplers is not None:
        #         print(f"downsample compress time {down_block.downsamplers[0].compress_time}")
        #     else:
        #         print(f"downsample None")
        # for up_block in self.decoder.up_blocks:
        #     if up_block.upsamplers is not None:
        #         print(f"upsample compress time {up_block.upsamplers[0].compress_time}")
        #     else:
        #         print("upsample None")

        self.quant_conv = CogVideoXSafeConv3d(2 * args.out_channels, 2 * args.out_channels, 1) if args.use_quant_conv else None
        self.post_quant_conv = CogVideoXSafeConv3d(args.out_channels, args.out_channels, 1) if args.use_post_quant_conv else None

        self.use_slicing = False
        self.use_tiling = False

        # Can be increased to decode more latent frames at once, but comes at a reasonable memory cost and it is not
        # recommended because the temporal parts of the VAE, here, are tricky to understand.
        # If you decode X latent frames together, the number of output frames is:
        #     (X + (2 conv cache) + (2 time upscale_1) + (4 time upscale_2) - (2 causal conv downscale)) => X + 6 frames
        #
        # Example with num_latent_frames_batch_size = 2:
        #     - 12 latent frames: (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11) are processed together
        #         => (12 // 2 frame slices) * ((2 num_latent_frames_batch_size) + (2 conv cache) + (2 time upscale_1) + (4 time upscale_2) - (2 causal conv downscale))
        #         => 6 * 8 = 48 frames
        #     - 13 latent frames: (0, 1, 2) (special case), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12) are processed together
        #         => (1 frame slice) * ((3 num_latent_frames_batch_size) + (2 conv cache) + (2 time upscale_1) + (4 time upscale_2) - (2 causal conv downscale)) +
        #            ((13 - 3) // 2) * ((2 num_latent_frames_batch_size) + (2 conv cache) + (2 time upscale_1) + (4 time upscale_2) - (2 causal conv downscale))
        #         => 1 * 9 + 5 * 8 = 49 frames
        # It has been implemented this way so as to not have "magic values" in the code base that would be hard to explain. Note that
        # setting it to anything other than 2 would give poor results because the VAE hasn't been trained to be adaptive with different
        # number of temporal frames.
        self.num_latent_frames_batch_size = 2
        self.num_sample_frames_batch_size = 2 * int(math.prod([float(a) for a in self.args.temporal_compression_list]))

        # We make the minimum height and width of sample for tiling half that of the generally supported
        self.tile_sample_min_height = args.sample_height // 2
        self.tile_sample_min_width = args.sample_width // 2
        self.tile_latent_min_height = int(
            self.tile_sample_min_height / 8
        )
        self.tile_latent_min_width = int(self.tile_sample_min_width / 8)

        # These are experimental overlap factors that were chosen based on experimentation and seem to work best for
        # 720x480 (WxH) resolution. The above resolution is the strongly recommended generation resolution in CogVideoX
        # and so the tiling implementation has only been tested on those specific resolutions.
        self.tile_overlap_factor_height = 1 / 6
        self.tile_overlap_factor_width = 1 / 5

        if cp.is_cp_initialized():
            self.cp_size = cp.get_cp_size()
            self.cp_rank = cp.get_cp_rank()

        self.lfq_weight = args.lfq_weight
        self.commitment_loss_weight = args.commitment_loss_weight
        self.compute_all_commitment = args.compute_all_commitment # compute commitment between input and rq-output
        if args.quantizer_type == 'MultiScaleBSQ':
            quantizer_class = MultiScaleBSQ
        elif args.quantizer_type == 'MultiScaleBSQTP':
            quantizer_class = MultiScaleBSQTP_AP
        elif args.quantizer_type == 'MultiScaleFSQ':
            quantizer_class = MultiScaleFSQ
        elif args.quantizer_type == 'MultiScaleFSQTP':
            quantizer_class = MultiScaleFSQTP
        elif args.quantizer_type == 'MultiScaleFSQSIM':
            quantizer_class = MultiScaleFSQSIM
        else:
            raise NotImplementedError
       
        ratio2hws_video_common_v2, total_pixels2scales = get_ratio2hws_video_v2()
        scales_256 = total_pixels2scales['0.06M']
        h_div_w2hw = {}
        for h_div_w in ratio2hws_video_common_v2:
            h_div_w2hw[h_div_w] = ratio2hws_video_common_v2[h_div_w][scales_256-1]
            h_div_w2hw[1/h_div_w] = (h_div_w2hw[h_div_w][1], h_div_w2hw[h_div_w][0])
        self.h_div_w2hw = h_div_w2hw
        self.h_div_w_templates = np.array(list(self.h_div_w2hw.keys()))
        self.scales_256 = scales_256
        args.h_div_w2hw = h_div_w2hw
        args.h_div_w_templates = self.h_div_w_templates
        args.scales_256 = scales_256
        dim = args.codebook_dim if args.codebook_dim_low < 0 else args.codebook_dim_low * 4
        self.quantizer = quantizer_class(
            dim = args.codebook_dim_low * 4, # this is the input feature dimension, defaults to log2(codebook_size) if not defined  
            entropy_loss_weight = args.entropy_loss_weight, # how much weight to place on entropy loss
            commitment_loss_weight=args.commitment_loss_weight, # loss weight of commitment loss
            use_stochastic_depth=args.use_stochastic_depth,
            drop_rate=args.drop_rate,
            schedule_mode=args.schedule_mode,
            keep_first_quant=args.keep_first_quant,
            keep_last_quant=args.keep_last_quant,
            remove_residual_detach=args.remove_residual_detach,
            use_out_phi=args.use_out_phi,
            use_out_phi_res=args.use_out_phi_res,
            random_flip = args.random_flip,
            flip_prob = args.flip_prob,
            flip_mode = args.flip_mode,
            max_flip_lvl = args.max_flip_lvl,
            random_flip_1lvl = args.random_flip_1lvl,
            flip_lvl_idx = args.flip_lvl_idx,
            drop_when_test = args.drop_when_test,
            drop_lvl_idx = args.drop_lvl_idx,
            drop_lvl_num = args.drop_lvl_num,
            random_short_schedule = args.random_short_schedule,
            short_schedule_prob = args.short_schedule_prob,
            use_bernoulli = args.use_bernoulli,
            use_rot_trick = args.use_rot_trick,
            disable_flip_prob = args.disable_flip_prob,
            casual_multi_scale = args.casual_multi_scale,
            temporal_slicing = args.temporal_slicing,
            last_scale_repeat_n = args.last_scale_repeat_n,
            num_lvl_fsq = args.num_lvl_fsq,
            other_args=args,
        )
        self.quantize = self.quantizer
        self.codebook_dim_continuous = args.codebook_dim
        assert args.codebook_dim_low > 0
        self.codebook_dim = args.codebook_dim_low * 4
        self.vocab_size = 2**self.codebook_dim

        if args.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if args.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
        
        self.origin_dim = 64
        assert args.use_feat_proj in [0, 1, 2], f'use_feat_proj must be 0, 1, 2'
        if args.use_feat_proj > 0:
            if args.use_feat_proj == 1:
                self.proj_down = nn.Linear(self.origin_dim*2, self.origin_dim*2)
                self.proj_down_two = nn.Linear(self.origin_dim*2, self.origin_dim*2)
            elif args.use_feat_proj == 2:
                self.proj_down = nn.Linear(self.origin_dim, self.origin_dim)
                self.proj_down_two = nn.Linear(self.origin_dim, self.origin_dim)
            self.proj_up = nn.Linear(self.origin_dim, self.origin_dim)
            self.proj_up_two = nn.Linear(self.origin_dim, self.origin_dim)
        else:
            self.proj_down, self.proj_up, self.proj_down_two, self.proj_up_two = nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity()
        self.other_args = args
        self.scale_learnable_parameters = nn.Parameter(torch.ones(4))

    def _set_gradient_checkpointing(self, module, value=False, subset=True):
        if isinstance(module, (CogVideoXEncoder3D, CogVideoXDecoder3D)):
            module.gradient_checkpointing = value

        for n, m in module.named_modules():
            if hasattr(m, 'gradient_checkpointing') and subset:
                m.gradient_checkpointing = value
    
    def _set_no_grad(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_overlap_factor_height: Optional[float] = None,
        tile_overlap_factor_width: Optional[float] = None,
    ) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_latent_min_height = int(
            self.tile_sample_min_height / 8
        )
        self.tile_latent_min_width = int(self.tile_sample_min_width / 8)
        self.tile_overlap_factor_height = tile_overlap_factor_height or self.tile_overlap_factor_height
        self.tile_overlap_factor_width = tile_overlap_factor_width or self.tile_overlap_factor_width

    def disable_tiling(self) -> None:
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False

    def enable_slicing(self) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = x.shape
        self.raw_height = height
        self.raw_width = width

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        frame_batch_size = self.num_sample_frames_batch_size
        # Note: We expect the number of frames to be either `1` or `frame_batch_size * k` or `frame_batch_size * k + 1` for some k.
        # As the extra single frame is handled inside the loop, it is not required to round up here.
        num_batches = max(num_frames // frame_batch_size, 1)
        if num_batches > 1:
            if cp.is_cp_initialized():
                frame_batch_size = num_frames // self.cp_size
                num_batches = self.cp_size
                cp.set_cp_on(True)
        else:
            cp.set_cp_on(False)


        conv_cache = None
        enc = []

        for i in range(num_batches):
            if cp.cp_on() and i != self.cp_rank:
                continue

            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            x_intermediate = x[:, :, start_frame:end_frame]

            
            torch._dynamo.mark_dynamic(x_intermediate, 0)
            torch._dynamo.mark_dynamic(x_intermediate, 2)
            if conv_cache is not None:
                for key, tensor in conv_cache.items():
                    if tensor is not None and isinstance(tensor, torch.Tensor):
                        torch._dynamo.mark_dynamic(tensor, 0)
            
            x_intermediate, conv_cache = self.encoder(x_intermediate, conv_cache=conv_cache)

            if self.quant_conv is not None:
                x_intermediate = self.quant_conv(x_intermediate)

            enc.append(x_intermediate)

        if cp.cp_on():
            enc = dist_encoder_gather_result(enc[0])

        enc = torch.cat(enc, dim=2)

        return enc

    def encode_for_raw_features(
        self, x: torch.Tensor, 
        scale_schedule,
        return_residual_norm_per_scale=False,
        slice=None,
    ):
        is_image = x.ndim == 4
        if not is_image:
            B, C, T, H, W = x.shape
        else:
            B, C, H, W = x.shape
            T = 1
            x = x.unsqueeze(2)

        with torch.amp.autocast("cuda", dtype=self.encoder_dtype):
            h = self.encode(x)
        # adjust latent dim
        h = patchify(h) # (B,c,t,H,W) -> (B,4c,t,H/2,W/2)

        posterior = DiagonalGaussianDistribution(h)
        z = posterior.sample()
        z = self.dropout_z_layer(z)
        if self.other_args.use_feat_proj == 2:
            z = self.proj_down(z.permute(0,2,3,4,1)).permute(0,4,1,2,3) # (B,24,t,H/2,W/2)
        z = z * self.scale_learnable_parameters[0]
        return z, None, None


    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ):
        h = None
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)

        if not return_dict:
            return (h,)
        return h

    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        batch_size, num_channels, num_frames, height, width = z.shape

        if self.use_tiling and (width > self.tile_latent_min_width or height > self.tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        frame_batch_size = self.num_latent_frames_batch_size

        num_batches = max(num_frames // frame_batch_size, 1)
        split_first = False
        if num_frames % frame_batch_size == 0 and num_batches:
            split_first = True
            num_batches -= 1
        if num_batches > 1:
            if cp.is_cp_initialized():
                frame_batch_size = num_frames // self.cp_size
                num_batches = self.cp_size
                cp.set_cp_on(True)
        else:
            cp.set_cp_on(False)

        conv_cache = None
        dec = []

        start_frame = 0
        remaining_frames = num_frames % frame_batch_size
        if split_first:
            remaining_frames += frame_batch_size
        for i in range(num_batches):
            if cp.cp_on() and i != self.cp_rank:
                continue

            end_frame = frame_batch_size * (i + 1) + remaining_frames
            z_intermediate = z[:, :, start_frame:end_frame]
            start_frame = end_frame
            if self.post_quant_conv is not None:
                z_intermediate = self.post_quant_conv(z_intermediate)


            torch._dynamo.mark_dynamic(z_intermediate, 0)
            torch._dynamo.mark_dynamic(z_intermediate, 2)
            torch._dynamo.mark_dynamic(z_intermediate, 3)
            torch._dynamo.mark_dynamic(z_intermediate, 4)
            if conv_cache is not None:
                for key, tensor in conv_cache.items():
                    if tensor is not None and isinstance(tensor, torch.Tensor):
                        torch._dynamo.mark_dynamic(tensor, 0)

            z_intermediate, conv_cache = self.decoder(z_intermediate, conv_cache=conv_cache, split_first=split_first)
            split_first = False

            dec.append(z_intermediate)

        if cp.cp_on():
            dec = dist_decoder_gather_result(dec[0])

        dec = torch.cat(dec, dim=2)

        if not return_dict:
            return (dec,)

        return dec

    def decode(self, z: torch.Tensor, return_dict: bool = True, **kwargs):

        z = z / self.scale_learnable_parameters[0]
        z = self.proj_up(z.permute(0,2,3,4,1)).permute(0,4,1,2,3)

        z = unpatchify(z) 
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z)

        if not return_dict:
            return (decoded,)
        return decoded

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.Tensor`): Input batch of videos.

        Returns:
            `torch.Tensor`:
                The latent representation of the encoded videos.
        """
        # For a rough memory estimate, take a look at the `tiled_decode` method.
        batch_size, num_channels, num_frames, height, width = x.shape

        overlap_height = int(self.tile_sample_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_sample_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_latent_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_latent_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_latent_min_height - blend_extent_height
        row_limit_width = self.tile_latent_min_width - blend_extent_width
        frame_batch_size = self.num_sample_frames_batch_size

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                # Note: We expect the number of frames to be either `1` or `frame_batch_size * k` or `frame_batch_size * k + 1` for some k.
                # As the extra single frame is handled inside the loop, it is not required to round up here.
                num_batches = max(num_frames // frame_batch_size, 1)
                conv_cache = None
                time = []

                for k in range(num_batches):
                    remaining_frames = num_frames % frame_batch_size
                    start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                    end_frame = frame_batch_size * (k + 1) + remaining_frames
                    tile = x[
                        :,
                        :,
                        start_frame:end_frame,
                        i : i + self.tile_sample_min_height,
                        j : j + self.tile_sample_min_width,
                    ]
                    tile, conv_cache = self.encoder(tile, conv_cache=conv_cache)
                    if self.quant_conv is not None:
                        tile = self.quant_conv(tile)
                    time.append(tile)

                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(torch.cat(result_row, dim=4))

        enc = torch.cat(result_rows, dim=3)
        return enc

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True):
        # Rough memory assessment:
        #   - In CogVideoX-2B, there are a total of 24 CausalConv3d layers.
        #   - The biggest intermediate dimensions are: [1, 128, 9, 480, 720].
        #   - Assume fp16 (2 bytes per value).
        # Memory required: 1 * 128 * 9 * 480 * 720 * 24 * 2 / 1024**3 = 17.8 GB
        #
        # Memory assessment when using tiling:
        #   - Assume everything as above but now HxW is 240x360 by tiling in half
        # Memory required: 1 * 128 * 9 * 240 * 360 * 24 * 2 / 1024**3 = 4.5 GB

        batch_size, num_channels, num_frames, height, width = z.shape

        overlap_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_sample_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_sample_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_sample_min_height - blend_extent_height
        row_limit_width = self.tile_sample_min_width - blend_extent_width
        frame_batch_size = self.num_latent_frames_batch_size

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                num_batches = max(num_frames // frame_batch_size, 1)
                conv_cache = None
                time = []

                for k in range(num_batches):
                    remaining_frames = num_frames % frame_batch_size
                    start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                    end_frame = frame_batch_size * (k + 1) + remaining_frames
                    tile = z[
                        :,
                        :,
                        start_frame:end_frame,
                        i : i + self.tile_latent_min_height,
                        j : j + self.tile_latent_min_width,
                    ]
                    if self.post_quant_conv is not None:
                        tile = self.post_quant_conv(tile)
                    tile, conv_cache = self.decoder(tile, conv_cache=conv_cache)
                    time.append(tile)

                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)

        if not return_dict:
            return (dec,)

        return dec

    ### original cogvideox forward
    # def forward(
    #     self,
    #     sample: torch.Tensor,
    #     sample_posterior: bool = False,
    #     return_dict: bool = True,
    #     generator: Optional[torch.Generator] = None,
    # ) -> Union[torch.Tensor, torch.Tensor]:
    #     x = sample
    #     posterior = self.encode(x).latent_dist
    #     if sample_posterior:
    #         z = posterior.sample(generator=generator)
    #     else:
    #         z = posterior.mode()
    #     dec = self.decode(z)
    #     if not return_dict:
    #         return (dec,)
    #     return dec

    def forward(self, x, disc_factor, image_disc=None, video_disc=None, image_perceptual_model=None, video_perceptual_model=None, is_train=True):
        device = x.device
        is_image = x.ndim == 4
        if not is_image:
            B, C, T, H, W = x.shape
        else:
            B, C, H, W = x.shape
            T = 1
            x = x.unsqueeze(2)
        
        semantic_enlarge_factor = torch.clamp(self.scale_learnable_parameters, min=0.01)[0] # for low resolution
        detail_enlarge_factor = torch.clamp(self.scale_learnable_parameters, min=0.01)[1] # for high resolution
        
        h_div_w = H / W
        h_div_w_template = self.h_div_w_templates[np.argmin(np.abs(self.h_div_w_templates - h_div_w))]
        hh, ww = self.h_div_w2hw[h_div_w_template]
        is_high_resolution = H*W > hh*ww*256
        x_list = []
        if self.other_args.use_multi_scale and is_high_resolution:
            x_list.append(F.interpolate(x, size=(T, hh*16, ww*16), mode=self.quantizer.z_interplote_down))
        x_list.append(x)
        assert len(x_list) <= 2
        z_list = []
        for i, x in enumerate(x_list):
            with torch.amp.autocast("cuda", dtype=self.encoder_dtype):
                h = self.encode(x)
            # adjust latent dim
            h = patchify(h) # (B,c,t,H,W) -> (B,4c,t,H/2,W/2)

            if self.other_args.use_feat_proj == 1:
                if i==0:
                    h = self.proj_down(h.permute(0,2,3,4,1)).permute(0,4,1,2,3) # (B,24,t,H/2,W/2)
                elif i==1:
                    h = self.proj_down_two(h.permute(0,2,3,4,1)).permute(0,4,1,2,3) # (B,24,t,H/2,W/2)

            posterior = DiagonalGaussianDistribution(h)
            z = posterior.sample()
            z = self.dropout_z_layer(z)

            if self.other_args.use_feat_proj == 2:
                if i==0:
                    z = self.proj_down(z.permute(0,2,3,4,1)).permute(0,4,1,2,3) # (B,24,t,H/2,W/2)
                elif i==1:
                    z = self.proj_down_two(z.permute(0,2,3,4,1)).permute(0,4,1,2,3) # (B,24,t,H/2,W/2)

            if i == 0:
                z_list.append(z.clone() * semantic_enlarge_factor)
            elif i==1:
                z_list.append(z.clone() * detail_enlarge_factor)

        # quantize
        # z_list_bk = z_list
        z_list, all_indices, all_loss = self.quantizer(z_list) # (B,24,t,H/2,W/2)
        # z_list = z_list_bk

        x_recon_list = []
        for i in range(len(z_list)):
            if i==0:
                z_list[i] = z_list[i] / semantic_enlarge_factor
                z_list[i] = self.proj_up(z_list[i].permute(0,2,3,4,1)).permute(0,4,1,2,3) # (B,64,t,H/2,W/2)
            elif i==1:
                z_list[i] = z_list[i] / detail_enlarge_factor
                z_list[i] = self.proj_up_two(z_list[i].permute(0,2,3,4,1)).permute(0,4,1,2,3) # (B,64,t,H/2,W/2)

            z_list[i] = unpatchify(z_list[i]) # (B,4c,t,H/2,W/2) -> (B,c,t,H,W)

            with torch.amp.autocast("cuda", dtype=self.decoder_dtype):
                x_recon = self.decode(z_list[i]).to(torch.float32)
            x_recon_list.append(x_recon)

        loss_dict, log_dict = {}, {}
        log_dict['semantic_enlarge_factor'] = torch.tensor(self.scale_learnable_parameters[0].item(), device=device)
        log_dict['detail_enlarge_factor'] = torch.tensor(self.scale_learnable_parameters[1].item(), device=device)

        if "FSQ" in self.args.quantizer_type:
            vq_output = {"encodings": all_indices}
        else:
            vq_output = {
                "commitment_loss": torch.mean(all_loss) * self.lfq_weight, # here commitment loss is sum of commitment loss and entropy penalty
                "encodings": all_indices, 
            }

        # assert x.shape == x_recon.shape, f"x.shape {x.shape}, x_recon.shape {x_recon.shape}"
        if is_train == False:
            if self.other_args.return_256_res:
                return x_list[0], x_recon_list[0]
            else:
                return x_list[-1], x_recon_list[-1]

        # if is_high_resolution_video:
        #     x_recon_list, x_list = x_recon_list[1:], x_list[1:]
        if "FSQ" not in self.args.quantizer_type:
            loss_dict["train/commitment_loss"] = vq_output['commitment_loss']
            # loss_dict["train/all_commitment_loss"] = vq_output['all_commitment_loss']
        for (x_recon, x) in zip(x_recon_list, x_list):
            if self.args.recon_loss_type == 'l1':
                recon_loss = F.l1_loss(x_recon, x) * self.args.l1_weight
            else:
                recon_loss = F.mse_loss(x_recon, x) * self.args.l1_weight
            if 'train/recon_loss' not in loss_dict:
                loss_dict['train/recon_loss'] = recon_loss
            else:
                loss_dict['train/recon_loss'] += recon_loss
            
            if is_image: # handle the cases with 4 dims
                flat_frames = x = x.squeeze(2)
                flat_frames_recon = x_recon = x_recon.squeeze(2)
            else:
                flat_frames = rearrange(x, "B C T H W -> (B T) C H W")
                flat_frames_recon = rearrange(x_recon, "B C T H W -> (B T) C H W")

            # Perceptual loss
            if is_image:
                image_perceptual_loss = image_perceptual_model(flat_frames, flat_frames_recon).mean() * self.args.perceptual_weight
                if "train/image_perceptual_loss" not in loss_dict:
                    loss_dict["train/image_perceptual_loss"] = image_perceptual_loss
                else:
                    loss_dict["train/image_perceptual_loss"] += image_perceptual_loss
            else:
                if self.args.lpips_model == "swin3d_t":
                    video_perceptual_loss = video_perceptual_model(x, x_recon).mean() * self.args.video_perceptual_weight
                else:
                    video_perceptual_loss = video_perceptual_model(flat_frames, flat_frames_recon).mean() * self.args.video_perceptual_weight
                if "train/video_perceptual_loss" not in loss_dict:
                    loss_dict["train/video_perceptual_loss"] = video_perceptual_loss
                else:
                    loss_dict["train/video_perceptual_loss"] += video_perceptual_loss

            ### GAN loss
            if self.args.image_gan_weight > 0 and (self.args.gan_image4video == "yes" or is_image):
                logits_image_fake = image_disc(flat_frames_recon)
                g_image_loss = -torch.mean(logits_image_fake) * self.args.image_gan_weight * disc_factor
                if 'train/g_image_loss' not in loss_dict:
                    loss_dict["train/g_image_loss"] = g_image_loss
                else:
                    loss_dict["train/g_image_loss"] += g_image_loss
            if T > 1 and self.args.video_gan_weight > 0:
                logits_video_fake = video_disc(x_recon)
                g_video_loss = -torch.mean(logits_video_fake) * self.args.video_gan_weight * disc_factor
                if 'train/g_video_loss' not in loss_dict:
                    loss_dict["train/g_video_loss"] = g_video_loss
                else:
                    loss_dict["train/g_video_loss"] += g_video_loss
        
        loss_dict['train/recon_loss'] /= len(x_list)
        if "train/image_perceptual_loss" in loss_dict:
            loss_dict["train/image_perceptual_loss"] /= len(x_list)
        if "train/video_perceptual_loss" in loss_dict:
            loss_dict["train/video_perceptual_loss"] /= len(x_list)

        x_recon1, flat_frames1, flat_frames_recon1 = x_recon.detach(), flat_frames.detach(), flat_frames_recon.detach()

        return (x, x_recon1, flat_frames1, flat_frames_recon1, loss_dict, log_dict)


    @staticmethod
    def add_model_specific_args(parent_parser):
        from infinity.models.videovae.utils import str2bool

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--in_channels", type=int, default=3)
        parser.add_argument("--out_channels", type=int, default=3)
        parser.add_argument("--down_block_types", type=str, nargs='+', default=[
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ])
        parser.add_argument("--down_block_mode", type=str, default="cogvideox", choices=["cogvideox", "dc"])
        parser.add_argument("--up_block_types", type=str, nargs='+', default=[
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ])
        parser.add_argument("--up_block_mode", type=str, default="cogvideox", choices=["cogvideox", "dc"])
        parser.add_argument("--block_out_channels", type=int, nargs='+', default=[128, 128, 256, 256, 512, 512])
        parser.add_argument("--layers_per_block", type=int, default=3)
        parser.add_argument("--latent_channels", type=int, default=16)
        parser.add_argument("--act_fn", type=str, default="silu")
        parser.add_argument("--norm_eps", type=float, default=1e-6)
        parser.add_argument("--norm_num_groups", type=int, default=32)
        # parser.add_argument("--temporal_compression_ratio", type=float, default=4) # deprecated
        parser.add_argument("--spatial_compression_list", type=int, nargs='+', default=[2, 2, 2], choices=[2])
        parser.add_argument("--temporal_compression_list", type=int, nargs='+', default=[2, 2], choices=[2, 3])
        parser.add_argument("--sample_height", type=int, default=480)
        parser.add_argument("--sample_width", type=int, default=720)
        parser.add_argument("--use_quant_conv", action="store_true")
        parser.add_argument("--use_post_quant_conv", action="store_true")
        parser.add_argument("--down_layer", type=str, default="conv", choices=["conv", "dc", "3d-dc"])
        parser.add_argument('--down_norm', type=str2bool, default=False)
        parser.add_argument("--up_layer", type=str, default="conv", choices=["conv", "dc", "3d-dc"])
        parser.add_argument('--up_norm', type=str2bool, default=False)
        parser.add_argument("--pad_mode", type=str, default="constant", choices=["constant", "replicate"])
        parser.add_argument("--dropout_z", type=float, default=0.0)
        return parser

if __name__ == '__main__':
    pass

