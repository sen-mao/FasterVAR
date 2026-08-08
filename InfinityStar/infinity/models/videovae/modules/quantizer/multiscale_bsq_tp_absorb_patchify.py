# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Binary Spherical Quantization
Proposed in https://arxiv.org/abs/2406.07548

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.
"""

import random
import copy
from math import log2, ceil
from functools import partial, cache
from collections import namedtuple
from contextlib import nullcontext

import torch.distributed as dist
from torch.distributed import nn as dist_nn

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module
from torch.amp import autocast
import numpy as np

from einops import rearrange, reduce, pack, unpack

# from einx import get_at

from infinity.models.videovae.utils.dynamic_resolution import predefined_HW_Scales_dynamic
from infinity.models.videovae.utils.dynamic_resolution_two_pyramid import dynamic_resolution_thw, total_pixels2scales

# print(f"{dynamic_resolution_thw=}")

# constants

Return = namedtuple('Return', ['quantized', 'indices', 'bit_indices', 'entropy_aux_loss'])

LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'batch_entropy', 'commitment'])

# distributed helpers

@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def maybe_distributed_mean(t):
    if not is_distributed():
        return t

    dist_nn.all_reduce(t)
    t = t / dist.get_world_size()
    return t

# helper functions

def exists(v):
    return v is not None

def identity(t):
    return t

def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    return F.normalize(t, dim = -1)

# entropy

def log(t, eps = 1e-5):
    return t.clamp(min = eps).log()

def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)

# cosine sim linear

class CosineSimLinear(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scale = 1.
    ):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(dim_in, dim_out))

    def forward(self, x):
        x = F.normalize(x, dim = -1)
        w = F.normalize(self.weight, dim = 0)
        return (x @ w) * self.scale

def repeat_schedule(scale_schedule, repeat_scales_num, times):
    new_scale_schedule = []
    for i in range(repeat_scales_num):
        new_scale_schedule.extend([scale_schedule[i] for _ in range(times)])
    new_scale_schedule.extend(scale_schedule[repeat_scales_num:])
    return new_scale_schedule

def get_latent2scale_schedule(T: int, H: int, W: int, mode="original", last_scale_repeat_n=0):
    assert mode in ["original", "dynamic", "dense", "same1", "same2", "same3", "half", "dense_f8", 'dense_f8_double', \
                    "infinity_video_two_pyramid", "infinity_video_two_pyramid_full_time", "infinity_video_two_pyramid_full_time_motion_boost_v2"]
    predefined_HW_Scales = {}
    if mode.startswith("infinity_video_two_pyramid"):
        if "motion_boost_v2" in mode:
            times = 6
            base_scale_schedule = copy.deepcopy(dynamic_resolution_thw[(H//2, W//2)]['scales'])
            image_scale_schedule = repeat_schedule(base_scale_schedule, 3, times)
            spatial_time_schedule = []
            spatial_time_schedule.extend(image_scale_schedule)
            firstframe_scalecnt = len(image_scale_schedule)
            if T > 1:
                scale_schedule = repeat_schedule(base_scale_schedule, 7, times)
                predefined_t = [T - 1 for _ in range(len(scale_schedule))]
                spatial_time_schedule.extend([(min(int(np.round(predefined_t[i])), T - 1), h, w) for i, (_, h, w) in enumerate(scale_schedule)])
            # double h and w
            spatial_time_schedule_double = [(t, 2*h, 2*w) for (t, h, w) in spatial_time_schedule]
            tower_split_index = firstframe_scalecnt
            return spatial_time_schedule_double, tower_split_index
        spatial_time_schedule = copy.deepcopy(dynamic_resolution_thw[(H//2, W//2)]['scales'])
        spatial_time_schedule.extend(spatial_time_schedule[-1:] * last_scale_repeat_n)
        tower_split_index = dynamic_resolution_thw[(H//2, W//2)]['tower_split_index'] + last_scale_repeat_n
        if T > 1:
            # predefined_t = np.linspace(1, compressed_frames - 1, len(scale_schedule))
            if mode == "infinity_video_two_pyramid_full_time":
                spatial_time_schedule.extend([(T - 1, h, w) for i, (_, h, w) in enumerate(spatial_time_schedule)])
            else:
                predefined_t = np.linspace(1, T - 1, total_pixels2scales['0.06M']-3).tolist() + [T - 1] * (len(spatial_time_schedule)-total_pixels2scales['0.06M']+3)
                spatial_time_schedule.extend([(min(int(np.round(predefined_t[i])), T - 1), h, w) for i, (_, h, w) in enumerate(spatial_time_schedule)])
            spatial_time_schedule.extend(spatial_time_schedule[-1:] * last_scale_repeat_n)
        # double h and w
        spatial_time_schedule_double = [(t, 2*h, 2*w) for (t, h, w) in spatial_time_schedule]
        return spatial_time_schedule_double, tower_split_index
    if mode == "original":
        predefined_HW_Scales = {
            # 256x256
            (16, 16): [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (10, 10), (13, 13), (16, 16)],
            (36, 64): [(1, 1), (2, 2), (3, 3), (4, 4), (6, 6), (9, 12), (13, 16), (18, 24), (24, 32), (32, 48), (36, 64)],
            (18, 32): [(1, 1), (2, 2), (3, 3), (4, 4), (6, 8), (8, 10), (10, 14), (12, 18), (14, 22), (16, 26), (18, 32)],
            (30, 53): [(1, 1), (2, 2), (3, 3), (4, 7), (6, 11), (8, 14), (12, 21), (16, 28), (20, 35), (22, 39), (24, 42), (26, 46), (28, 50), (30, 53)]
        }
        predefined_HW_Scales[(32, 32)] = predefined_HW_Scales[(16, 16)] + [(20, 20), (24, 24), (32, 32)]
        predefined_HW_Scales[(64, 64)] = predefined_HW_Scales[(32, 32)] + [(40, 40), (48, 48), (64, 64)]
    elif mode == "dynamic":
        predefined_HW_Scales.update(predefined_HW_Scales_dynamic)
    elif mode == "dense":
        predefined_HW_Scales[(16, 16)] = [(x, x) for x in range(1, 16+1)]
        predefined_HW_Scales[(32, 32)] = predefined_HW_Scales[(16, 16)] + [(20, 20), (24, 24), (28, 28), (32, 32)]
        predefined_HW_Scales[(64, 64)] = predefined_HW_Scales[(32, 32)] + [(40, 40), (48, 48), (56, 56), (64, 64)]
    elif mode == "dense_f8":
        # predefined_HW_Scales[(16, 16)] = [(x, x) for x in range(1, 16+1)]
        predefined_HW_Scales[(32, 32)] = [(x, x) for x in range(1, 16+1)] + [(20, 20), (24, 24), (28, 28), (32, 32)]
        predefined_HW_Scales[(64, 64)] = predefined_HW_Scales[(32, 32)] + [(40, 40), (48, 48), (56, 56), (64, 64)]
        predefined_HW_Scales[(128, 128)] = predefined_HW_Scales[(64, 64)] + [(80, 80), (96, 96), (112, 112), (128, 128)]
    elif mode == "dense_f8_double":
        # predefined_HW_Scales setting double from dense f16
        predefined_HW_Scales[(32, 32)] = [(x, x) for x in range(1, 16+1)]
        predefined_HW_Scales[(64, 64)] = predefined_HW_Scales[(32, 32)] + [(20, 20), (24, 24), (28, 28), (32, 32)]
        predefined_HW_Scales[(96, 96)] = predefined_HW_Scales[(64, 64)] + [(40, 40), (48, 48)]
        predefined_HW_Scales[(128, 128)] = predefined_HW_Scales[(64, 64)] + [(40, 40), (48, 48), (56, 56), (64, 64)]

        predefined_HW_Scales[(24, 42)] = [(1, 1), (2, 2), (3, 3), (3, 4), (3, 5), (4, 6), (4, 7), (5, 8), (6, 9), (6, 10), (6, 11), (7, 12), (7, 13), (8, 14), (9, 15), (9, 16), (12, 21)]       
        predefined_HW_Scales[(36, 64)] = predefined_HW_Scales[(24, 42)] + [(14, 26), (18, 32)]
        predefined_HW_Scales[(60, 108)] = predefined_HW_Scales[(36, 64)] + [(24, 42), (30, 54)]
        predefined_HW_Scales[(90, 160)] = predefined_HW_Scales[(60, 108)] + [(38, 66),(45, 80)]

        for k, v in predefined_HW_Scales.items():
            predefined_HW_Scales[k] = [(2*x, 2*y) for (x, y) in v]
    elif mode.startswith("same"):
        num_quant = int(mode[len("same"):])
        predefined_HW_Scales[(16, 16)] = [(16, 16) for _ in range(num_quant)]
        predefined_HW_Scales[(32, 32)] = [(32, 32) for _ in range(num_quant)]
        predefined_HW_Scales[(64, 64)] = [(64, 64) for _ in range(num_quant)]
    elif mode == "half":
        predefined_HW_Scales[(32, 32)] = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (10, 10), (13, 13), (16, 16)]
        predefined_HW_Scales[(64, 64)] = [(1,1),(2,2),(4,4),(6,6),(8,8),(12,12),(16,16)]
    else:
        raise NotImplementedError

    # predefined_T_Scales = [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 17, 17, 17, 17, 17, 17]
    # predefined_T_Scales = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
    predefined_T_Scales = [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
    # predefined_T_Scales = [1, 2, 3, 5, 6, 8, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    patch_THW_shape_per_scale = predefined_HW_Scales[(H, W)]
    if len(predefined_T_Scales) < len(patch_THW_shape_per_scale):
        # print("warning: the length of predefined_T_Scales is less than the length of patch_THW_shape_per_scale!")
        predefined_T_Scales += [predefined_T_Scales[-1]] * (len(patch_THW_shape_per_scale) - len(predefined_T_Scales))
    patch_THW_shape_per_scale = [(min(T, t), h, w ) for (h, w), t in zip(patch_THW_shape_per_scale, predefined_T_Scales[:len(patch_THW_shape_per_scale)])]
    return patch_THW_shape_per_scale

# TP: Two Pyramid
class MultiScaleBSQTP(Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(
        self,
        *,
        dim,
        soft_clamp_input_value = None,
        aux_loss = False, # intermediate auxiliary loss
        use_stochastic_depth=False,
        drop_rate=0.,
        schedule_mode="original", # ["original", "dynamic", "dense"]
        keep_first_quant=False,
        keep_last_quant=False,
        remove_residual_detach=False,
        random_flip = False,
        flip_prob = 0.5,
        flip_mode = "stochastic", # "stochastic", "deterministic"
        max_flip_lvl = 1,
        random_flip_1lvl = False, # random flip one level each time
        flip_lvl_idx = None,
        drop_when_test=False,
        drop_lvl_idx=None,
        drop_lvl_num=0,
        random_short_schedule = False, # randomly use short schedule (schedule for images of 256x256)
        short_schedule_prob = 0.5,
        disable_flip_prob = 0.0, # disable random flip in this image
        casual_multi_scale = False,  # causal multiscale
        temporal_slicing = False,
        last_scale_repeat_n = 0,
        num_lvl_fsq = None,
        other_args = None,
        **kwargs
    ):
        super().__init__()
        codebook_dim = dim
        self.use_stochastic_depth = use_stochastic_depth
        self.drop_rate = drop_rate
        self.remove_residual_detach = remove_residual_detach
        self.random_flip = random_flip
        self.flip_prob = flip_prob
        self.flip_mode = flip_mode
        self.max_flip_lvl = max_flip_lvl
        self.random_flip_1lvl = random_flip_1lvl
        self.flip_lvl_idx = flip_lvl_idx
        assert (random_flip and random_flip_1lvl) == False
        self.disable_flip_prob = disable_flip_prob
        self.casual_multi_scale = casual_multi_scale
        self.temporal_slicing = temporal_slicing
        self.last_scale_repeat_n = last_scale_repeat_n
        # print(f"{casual_multi_scale=}")

        self.drop_when_test = drop_when_test
        self.drop_lvl_idx = drop_lvl_idx
        self.drop_lvl_num = drop_lvl_num
        if self.drop_when_test:
            assert drop_lvl_idx is not None
            assert drop_lvl_num > 0
        self.random_short_schedule = random_short_schedule
        self.short_schedule_prob = short_schedule_prob
        self.z_interplote_up = 'trilinear'
        self.z_interplote_down = 'area'
        
        self.schedule_mode = schedule_mode
        self.keep_first_quant = keep_first_quant
        self.keep_last_quant = keep_last_quant
        if self.use_stochastic_depth and self.drop_rate > 0:
            assert self.keep_first_quant or self.keep_last_quant

        self.full2short = {7:7, 10:7, 13:7, 16:16, 20:16, 24:16}
        if self.schedule_mode == 'dense_f8':
            self.full2short_f8 = {20:20, 24:24, 28:24}
        elif self.schedule_mode == 'dense_f8_double':
            self.full2short_f8 = {16: 14, 17: 14, 19: 14, 20:14, 21:14, 22:14, 24:14}
        elif self.schedule_mode.startswith("infinity_video_two_pyramid"):
            self.full2short_f8 = {11: 11, 13: 11, 14: 11, 16: 11, 29: 26, 28: 26, 26: 26}

        self.other_args = other_args
        self.origin_C = 64
        self.detail_scale_dim, self.semantic_scale_dim = self.other_args.detail_scale_dim, self.other_args.semantic_scale_dim
        self.lfq_semantic = BSQ(
            dim = self.semantic_scale_dim,
            codebook_scale = 1,
            soft_clamp_input_value = soft_clamp_input_value,
            **kwargs,
        )
        self.lfq_detail = BSQ(
            dim = self.detail_scale_dim,
            codebook_scale = 1,
            soft_clamp_input_value = soft_clamp_input_value,
            **kwargs,
        )

        self.detail_scale_min_tokens = other_args.detail_scale_min_tokens # include
        
        if self.other_args.use_learnable_dim_proj:
            middle_hidden_dim=64
            self.semantic_proj_down = nn.Sequential(
                nn.Linear(self.origin_C, middle_hidden_dim),
                nn.SiLU(),
                nn.Linear(middle_hidden_dim, self.semantic_scale_dim),
            )
            self.semantic_proj_up = nn.Sequential(
                nn.Linear(self.semantic_scale_dim, middle_hidden_dim),
                nn.SiLU(),
                nn.Linear(middle_hidden_dim, self.origin_C),
            )

            assert self.detail_scale_dim >= self.origin_C
            if self.detail_scale_dim == self.origin_C:
                self.detail_proj_up, self.detail_proj_down = nn.Identity(), nn.Identity()
            else:
                self.detail_proj_up = nn.Sequential(
                    nn.Linear(self.origin_C, middle_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(middle_hidden_dim, self.detail_scale_dim),
                )
                self.detail_proj_down = nn.Sequential(
                    nn.Linear(self.detail_scale_dim, middle_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(middle_hidden_dim, self.origin_C),
                )

    @property
    def codebooks(self):
        return self.lfq.codebook

    def get_codes_from_indices(self, indices_list):
        all_codes = []
        for indices in indices_list:
            codes = self.lfq.indices_to_codes(indices)
            all_codes.append(codes)
        _, _, T, H, W = all_codes[-1].size()
        summed_codes = 0
        for code in all_codes:
            summed_codes += F.interpolate(code, size=(T, H, W), mode=self.z_interplote_up)
        return summed_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return codes_summed

    def flip_quant(self, x):
        # assert self.flip_mode in ['stochastic', 'stochastic_dynamic']
        if self.flip_mode == 'stochastic':
            flip_mask = torch.rand_like(x) < self.flip_prob
        elif self.flip_mode == 'stochastic_dynamic':
            flip_prob = random.uniform(0, self.flip_prob)
            flip_mask = torch.rand_like(x) < flip_prob
        else:
            raise NotImplementedError
        x = x.clone()
        x[flip_mask] = -x[flip_mask]
        return x

    def forward(
        self,
        x,
        mask = None,
        return_all_codes = False,
    ):
        if x.ndim == 4:
            x = x.unsqueeze(2)
        B, C, T, H, W = x.size()    

        if self.schedule_mode.startswith("same"):
            scale_num = int(self.schedule_mode[len("same"):])
            assert T == 1
            scale_schedule = [(1, H, W)] * scale_num
        elif self.schedule_mode.startswith("infinity_video_two_pyramid"):
            scale_schedule, tower_split_index = get_latent2scale_schedule(T, H, W, mode=self.schedule_mode, last_scale_repeat_n=self.last_scale_repeat_n)
            scale_num = len(scale_schedule)
        else:
            scale_schedule = get_latent2scale_schedule(T, H, W, mode=self.schedule_mode)
            scale_num = len(scale_schedule)
                        
        if self.training and self.random_short_schedule and random.random() < self.short_schedule_prob:
            if self.schedule_mode.startswith("infinity_video_two_pyramid"):
                if T == 1:
                    scale_num = self.full2short_f8[scale_num]
                    tower_split_index = scale_num
                else:
                    pass
            else:
                if self.schedule_mode.startswith("dense_f8"):
                    # print(B, C, T, H, W, scale_num, self.full2short_f8[scale_num], scale_schedule)
                    scale_num = self.full2short_f8[scale_num]
                    # print('after: \n', scale_schedule[:scale_num])
                else:
                    scale_num = self.full2short[scale_num]
            scale_schedule = scale_schedule[:scale_num]
        
        quantized_out = 0.
        residual = x
        quantized_out_firstframe = None

        all_losses = []
        all_indices = []
        all_bit_indices = []
        var_inputs = []
        residual_norm_per_scale = []
        
        # go through the layers
        # residual_list = []
        # interpolate_residual_list = []
        # quantized_list = []
        with autocast('cuda', enabled = False):
            for si, (pt, ph, pw) in enumerate(scale_schedule):

                if si < tower_split_index:
                    if (pt, ph, pw) != (1, H, W):
                        interpolate_residual = F.interpolate(residual[:, :, :1, :, :].clone(), size=(pt, ph, pw), mode=self.z_interplote_down)
                    else:
                        interpolate_residual = residual[:, :, :1, :, :]
                else:
                    if (pt, ph, pw) != (T-1, H, W):
                        if self.casual_multi_scale:
                            interpolate_residual = F.interpolate(residual[:, :, 1:pt+1, :, :], size=(pt, ph, pw), mode=self.z_interplote_down)
                        elif self.temporal_slicing:
                            temporal_indices = list(map(int, np.linspace(1, T-1, pt)))
                            assert len(temporal_indices) == pt
                            interpolate_residual = F.interpolate(residual[:, :, temporal_indices, :, :], size=(pt, ph, pw), mode=self.z_interplote_down)
                        else:
                            interpolate_residual = F.interpolate(residual[:, :, 1:, :, :].clone(), size=(pt, ph, pw), mode=self.z_interplote_down)
                    else:
                        interpolate_residual = residual[:, :, 1:, :, :]
                if si != 0 and si != tower_split_index and self.use_stochastic_depth and random.random() < self.drop_rate:
                    quantized = torch.zeros_like(interpolate_residual)
                else:
                    quantized, indices, bit_indices, loss = self.lfq(interpolate_residual)
                    all_indices.append(indices)
                    all_losses.append(loss)
                    all_bit_indices.append(bit_indices)

                # if (pt, ph, pw) != (T, H, W):
                if si < tower_split_index:
                    if (pt, ph, pw) != (1, H, W):
                        quantized = F.interpolate(quantized, size=(1, H, W), mode=self.z_interplote_up).contiguous()
                else:
                    if (pt, ph, pw) != (T-1, H, W):
                        quantized = F.interpolate(quantized, size=(T-1, H, W), mode=self.z_interplote_up).contiguous()
                if si < tower_split_index:
                    residual[:, :, :1, :, :] = residual[:, :, :1, :, :] - quantized
                else:
                    residual[:, :, 1:, :, :] = residual[:, :, 1:, :, :] - quantized

                if si < tower_split_index:
                    quantized_out = quantized_out + quantized
                    if si == tower_split_index - 1:
                        quantized_out_firstframe = quantized_out.clone()
                        quantized_out = 0
                else:
                    quantized_out = quantized_out + quantized

            if quantized_out_firstframe is not None:
                if len(scale_schedule) == tower_split_index:
                    quantized_out = quantized_out_firstframe
                else:
                    quantized_out = torch.cat([quantized_out_firstframe, quantized_out], dim=2)

        # print("residual_list:", residual_list)
        # print("interpolate_residual_list:", interpolate_residual_list)
        # print("quantized_list:", quantized_list)
        # import ipdb; ipdb.set_trace()
        # project out, if needed

        # stack all losses and indices

        all_losses = torch.stack(all_losses, dim = -1)

        ret = (quantized_out, all_indices, all_bit_indices, residual_norm_per_scale, all_losses, var_inputs)

        if not return_all_codes:
            return ret

        # whether to return all codes from all codebooks across layers
        all_codes = self.get_codes_from_indices(all_indices)

        # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)

        return (*ret, all_codes)


class BSQ(Module):
    def __init__(
        self,
        *,
        dim = None,
        entropy_loss_weight = 0.1,
        commitment_loss_weight = 0.25,
        num_codebooks = 1,
        keep_num_codebooks_dim = None,
        codebook_scale = 1.,                        # for residual LFQ, codebook scaled down by 2x at each layer
        frac_per_sample_entropy = 1.,               # make less than 1. to only use a random fraction of the probs for per sample entropy
        soft_clamp_input_value = None,
        channel_first = None,
        experimental_softplus_entropy_loss = False,
        entropy_loss_offset = 5.,                   # how much to shift the loss before softplus
        spherical = True,                          # from https://arxiv.org/abs/2406.07548
        force_quantization_f32 = True,               # will force the quantization step to be full precision
        inv_temperature = 100.0,
        gamma0=1.0, gamma=1.0, zeta=1.0,
        use_out_phi = False, # use output phi network
        use_out_phi_res = False, # residual out phi
        use_bernoulli = False,
        use_rot_trick = False,
    ):
        super().__init__()

        # some assert validations
        assert exists(dim) , 'dim must be specified for BSQ'

        codebook_dim = dim
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)
        self.codebook_dims = codebook_dims

        self.out_phi = nn.Linear(codebook_dims, codebook_dims) if use_out_phi else nn.Identity()
        self.use_out_phi_res = use_out_phi_res
        if self.use_out_phi_res:
            self.out_phi_scale = nn.Parameter(torch.zeros(codebook_dims), requires_grad=True) # init as zero

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # channel first
        self.channel_first = channel_first

        # For BSQ (binary spherical quantization)
        if not spherical:
            raise ValueError("For BSQ, spherical must be True.")
        self.persample_entropy_compute = 'analytical'
        self.inv_temperature = inv_temperature
        self.gamma0 = gamma0  # loss weight for entropy penalty
        self.gamma = gamma  # loss weight for entropy penalty
        self.zeta = zeta    # loss weight for entire entropy penalty
        self.use_bernoulli = use_bernoulli
        self.use_rot_trick = use_rot_trick

        # entropy aux loss related weights

        assert 0 < frac_per_sample_entropy <= 1.
        self.frac_per_sample_entropy = frac_per_sample_entropy

        self.entropy_loss_weight = entropy_loss_weight

        # codebook scale

        self.codebook_scale = codebook_scale

        # commitment loss

        self.commitment_loss_weight = commitment_loss_weight

        # whether to soft clamp the input value from -value to value

        self.soft_clamp_input_value = soft_clamp_input_value
        assert not exists(soft_clamp_input_value) or soft_clamp_input_value >= codebook_scale

        # whether to make the entropy loss positive through a softplus (experimental, please report if this worked or not in discussions)

        self.entropy_loss_offset = entropy_loss_offset
        self.experimental_softplus_entropy_loss = experimental_softplus_entropy_loss

        # for no auxiliary loss, during inference

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # whether to force quantization step to be f32

        self.force_quantization_f32 = force_quantization_f32

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    # @property
    # def dtype(self):
    #     return self.codebook.dtype

    def indices_to_codes(
        self,
        indices,
        label_type = 'int_label',
        project_out = True
    ):
        assert label_type in ['int_label', 'bit_label']
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        should_transpose = default(self.channel_first, is_img_or_video)

        if not self.keep_num_codebooks_dim:
            if label_type == 'int_label':
                indices = rearrange(indices, '... -> ... 1')
            else:
                indices = indices.unsqueeze(-2)

        # indices to codes, which are bits of either -1 or 1

        if label_type == 'int_label':
            assert indices[..., None].int().min() > 0
            bits = ((indices[..., None].int() & self.mask) != 0).float() # .to(self.dtype)
        else:
            bits = indices

        codes = self.bits_to_codes(bits).float()

        codes = l2norm(codes) # must normalize when using BSQ

        codes = rearrange(codes, '... c d -> ... (c d)')

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)

        # rearrange codes back to original shape

        if should_transpose:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def quantize(self, z):
        assert z.shape[-1] == self.codebook_dims, f"Expected {self.codebook_dims} dimensions, got {z.shape[-1]}"

        zhat = torch.where(z > 0, 
                           torch.tensor(1, dtype=z.dtype, device=z.device), 
                           torch.tensor(-1, dtype=z.dtype, device=z.device))

        q_scale = 1. / (self.codebook_dims ** 0.5)
        zhat = q_scale * zhat # on unit sphere

        return z + (zhat - z).detach()

    def quantize_new_bernoulli(self, z, prob_z):
        assert z.shape[-1] == self.codebook_dims, f"Expected {self.codebook_dims} dimensions, got {z.shape[-1]}"

        zhat = (torch.bernoulli(prob_z) - 0.5) * 2.0

        q_scale = 1. / (self.codebook_dims ** 0.5)
        zhat = q_scale * zhat # on unit sphere

        return z + (zhat - z).detach()

    def rot_quantize(self, z, inference=False):
        assert z.shape[-1] == self.codebook_dims, f"Expected {self.codebook_dims} dimensions, got {z.shape[-1]}"
        q_scale = 1. / (self.codebook_dims ** 0.5)
        zhat = torch.where(z > 0, 
                            torch.tensor(1, dtype=z.dtype, device=z.device), 
                            torch.tensor(-1, dtype=z.dtype, device=z.device)) * q_scale
        if inference:
            return zhat

        w = ((z + zhat) / torch.norm(z + zhat, dim=-1, keepdim=True)).detach()
        z = z.unsqueeze(1) - 2*torch.bmm(torch.bmm(z.unsqueeze(1), w.unsqueeze(-1)), w.unsqueeze(1)) + 2 * torch.bmm(
            torch.bmm(z.unsqueeze(1), z.unsqueeze(-1).detach()), zhat.unsqueeze(1).detach())
        return z.squeeze()

    def soft_entropy_loss(self, z):
        if self.persample_entropy_compute == 'analytical':
            # if self.l2_norm:
            p = torch.sigmoid(-4 * z / (self.codebook_dims ** 0.5) * self.inv_temperature)
            # else:
            #     p = torch.sigmoid(-4 * z * self.inv_temperature)
            prob = torch.stack([p, 1-p], dim=-1) # (b, h, w, 18, 2)
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean() # (b,h,w,18)->(b,h,w)->scalar
        else:
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()

        # macro average of the probability of each subgroup
        avg_prob = reduce(prob, '... g d ->g d', 'mean') # (18, 2)
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)

        # the approximation of the entropy is the sum of the entropy of each subgroup
        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        if normalize: # False
            probs = (count + eps) / (count + eps).sum(dim=dim, keepdim =True)
        else: # True
            probs = count
        H = -(probs * torch.log(probs + 1e-8)).sum(dim=dim)
        return H

    def forward(
        self,
        x,
        return_loss_breakdown = False,
        mask = None,
        entropy_weight=0.1
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = x.ndim >= 4
        should_transpose = default(self.channel_first, is_img_or_video)

        # standardize image or video into (batch, seq, dimension)

        if should_transpose:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d') # x.shape [b, hwt, c]

        assert x.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {x.shape[-1]}'

        # split out number of codebooks

        x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)

        if self.use_bernoulli:
            prob_x = torch.sigmoid(x)
        
        x = l2norm(x)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32

        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        with quantization_context():

            if force_f32:
                orig_dtype = x.dtype
                x = x.float()
            
            # use straight-through gradients
            if self.use_rot_trick:
                x_f = x.flatten(end_dim=-2) # (b, hwt, 1, d) -> (bhwt, d)
                q_f = self.rot_quantize(x_f, inference= not self.training)
                quantized = q_f.reshape(x.shape)
            elif self.use_bernoulli:
                quantized = self.quantize_new_bernoulli(x, prob_x)
            else:
                quantized = self.quantize(x)

            # calculate indices
            indices = reduce((quantized > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')
            bit_indices = (quantized > 0).int()

            # entropy aux loss
            if self.training:
                persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(x) # compute entropy
                entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
            else:
                # if not training, just return dummy 0
                entropy_penalty = persample_entropy = cb_entropy = self.zero

            # commit loss

            if self.training and self.commitment_loss_weight > 0.:

                commit_loss = F.mse_loss(x, quantized.detach(), reduction = 'none')

                if exists(mask):
                    commit_loss = commit_loss[mask]

                commit_loss = commit_loss.mean()
            else:
                commit_loss = self.zero

            # input back to original dtype if needed

            if force_f32:
                x = x.type(orig_dtype)

        # merge back codebook dim
        x = quantized # rename quantized to x for output
        
        if self.use_out_phi_res:
            x = x + self.out_phi_scale * self.out_phi(x) # apply out_phi on quant output as residual
        else:
            x = self.out_phi(x) # apply out_phi on quant output
        
        x = rearrange(x, 'b n c d -> b n (c d)')

        # reconstitute image or video dimensions

        if should_transpose:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')

            bit_indices = unpack_one(bit_indices, ps, 'b * c d')

        # whether to remove single codebook dim

        if not self.keep_num_codebooks_dim:
            bit_indices = rearrange(bit_indices, '... 1 d -> ... d')

        # complete aux loss

        aux_loss = commit_loss * self.commitment_loss_weight + (self.zeta * entropy_penalty / self.inv_temperature)*entropy_weight
        # returns

        ret = Return(x, indices, bit_indices, aux_loss)

        if not return_loss_breakdown:
            return ret

        return ret, LossBreakdown(persample_entropy, cb_entropy, commit_loss)

