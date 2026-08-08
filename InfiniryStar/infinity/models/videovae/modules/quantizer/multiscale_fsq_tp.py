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
from infinity.models.videovae.modules.quantizer.finite_scalar_quantization import FSQ
# print(f"{dynamic_resolution_thw=}")

# constants

Return = namedtuple('Return', ['quantized', 'indices', 'entropy_aux_loss'])

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
    predefined_HW_Scales = {}
    if mode.startswith("infinity_video_two_pyramid"):
        if 'elegant' in mode:
            base_scale_schedule = copy.deepcopy(dynamic_resolution_thw[(H, W)]['scales'])
            image_scale_repetition = [5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2] + [1] * 10
            video_scale_repetition = [5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2] + [1] * 10
            base_scale_schedule = copy.deepcopy(dynamic_resolution_thw[(H, W)]['scales'])
            def repeat_scales(base_scale_schedule, scale_repetition):
                scale_schedule = []
                for i in range(len(base_scale_schedule)):
                    scale_schedule.extend([base_scale_schedule[i] for _ in range(scale_repetition[i])])
                return scale_schedule
            image_scale_schedule = repeat_scales(base_scale_schedule, image_scale_repetition)
            spatial_time_schedule = []
            spatial_time_schedule.extend(image_scale_schedule)
            firstframe_scalecnt = len(image_scale_schedule)
            if T > 1:
                scale_schedule = repeat_scales(base_scale_schedule, video_scale_repetition)
                spatial_time_schedule.extend([(T-1, h, w) for i, (_, h, w) in enumerate(scale_schedule)])
            # double h and w
            tower_split_index = firstframe_scalecnt
            # print(f'{spatial_time_schedule=}')
            return spatial_time_schedule, tower_split_index
        if "motion_boost_v2" in mode:
            times = 6
            base_scale_schedule = copy.deepcopy(dynamic_resolution_thw[(H, W)]['scales'])
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
        spatial_time_schedule = copy.deepcopy(dynamic_resolution_thw[(H, W)]['scales'])
        spatial_time_schedule.extend(spatial_time_schedule[-1:] * last_scale_repeat_n)
        tower_split_index = dynamic_resolution_thw[(H, W)]['tower_split_index'] + last_scale_repeat_n
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
class MultiScaleFSQTP(Module):
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
            self.full2short_f8 = {11: 11, 13: 11, 14: 11, 16: 11}

        self.other_args = other_args
        print(f'{self.other_args=}')
        self.origin_C = 64
        self.detail_scale_dim, self.semantic_scale_dim = self.other_args.detail_scale_dim, self.other_args.semantic_scale_dim
        self.lfq_semantic = FSQ(
            dim = self.semantic_scale_dim,
            num_lvl = num_lvl_fsq,
        )
        self.lfq_detail = FSQ(
            dim = self.detail_scale_dim,
            num_lvl = num_lvl_fsq,
        )

        self.detail_scale_min_tokens = 80 # include
        middle_hidden_dim=64
        if self.other_args.use_learnable_dim_proj:
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
            # assert self.detail_scale_dim >= self.origin_C
            if self.detail_scale_dim == self.origin_C:
                self.detail_proj_up, self.detail_proj_down = nn.Identity(), nn.Identity()
            elif self.detail_scale_dim > self.origin_C:
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
            else:
                self.detail_proj_down = nn.Sequential(
                    nn.Linear(self.origin_C, middle_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(middle_hidden_dim, self.detail_scale_dim),
                )
                self.detail_proj_up = nn.Sequential(
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
        double = False
    ):
        if x.ndim == 4:
            x = x.unsqueeze(2)
        B, C, T, H, W = x.size()    
        if self.schedule_mode.startswith("same"):
            scale_num = int(self.schedule_mode[len("same"):])
            assert T == 1
            scale_schedule = [(1, H, W)] * scale_num
        elif self.schedule_mode.startswith("infinity_video_two_pyramid") or self.schedule_mode == "last_only_two_pyramid":
            if double:
                scale_schedule, tower_split_index = get_latent2scale_schedule(T, H*2, W*2, mode=self.schedule_mode, last_scale_repeat_n=self.last_scale_repeat_n)
                scale_schedule = [(t, h//2, w//2) for (t, h, w) in scale_schedule]
                scale_num = len(scale_schedule)
            else:
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
        
        # go through the layers
        # residual_list = []
        # interpolate_residual_list = []
        # quantized_list = []
        with autocast('cuda', enabled = False):
            for si, (pt, ph, pw) in enumerate(scale_schedule):
                if si < tower_split_index:
                    tgt_shape = (self.origin_C, 1, H, W)
                    ss, ee = 0, 1
                else:
                    tgt_shape = (self.origin_C, T-1, H, W)
                    ss, ee = 1, T
                is_semantic_scale = True
                if ph * pw >= self.detail_scale_min_tokens:
                    is_semantic_scale = False
                    C1 = self.detail_scale_dim
                    lfq = self.lfq_detail
                else:
                    C1 = self.semantic_scale_dim
                    lfq = self.lfq_semantic

                def interpolate(tensor, size, mode, quantizer, is_semantic_scale):
                    """
                    arguments:
                        tensor: (B,C,T,H,W)
                        size: (C1,T,H1,W1)
                        mode: str
                        quantizer: quantizer
                        is_semantic_scale: bool
                    return:
                        tensor: (B,*size)
                    """
                    B, C, T, H, W = tensor.shape
                    C1, T, H1, W1 = size
                    if quantizer.other_args.use_learnable_dim_proj:
                        if is_semantic_scale:
                            if C > C1:
                                proj = self.semantic_proj_down
                            elif C < C1:
                                proj = self.semantic_proj_up
                        else:
                            if C > C1:
                                proj = self.detail_proj_down
                            elif C < C1:
                                proj = self.detail_proj_up
                        if C != C1:
                            tensor = tensor.permute(0,2,3,4,1) #  (B,C,T,H,W) -> (B,T,H,W,C)
                            tensor = proj(tensor) # (B,T,H,W,C1)
                            tensor = tensor.permute(0,4,1,2,3) # (B,T,H,W,C1) -> (B,C1,T,H,W)
                        tensor = F.interpolate(tensor, size=(T, H1, W1), mode=mode) # (B,C1,T,H,W) -> (B,C1,T,H1,W1)
                        return tensor
                    else:
                        tensor = tensor.permute(0,2,1,3,4) # (B,C,T,H,W) -> (B,T,C,H,W)
                        tensor = F.interpolate(tensor, size=(C1, H1, W1), mode=mode)
                        tensor = tensor.permute(0,2,1,3,4) # (B,T,C1,H1,W1) -> (B,C1,T,H1,W1)
                    return tensor

                if ph * pw < 16*16: # 192p drop
                    skip_detail_scales = False
                else:
                    if random.random() < self.other_args.skip_detail_scales_prob:
                        skip_detail_scales = True
                    
                if (not skip_detail_scales):
                    interpolate_residual = interpolate(residual[:, :, ss:ee, :, :].clone(), size=(C1, pt, ph, pw), mode=self.z_interplote_down, quantizer=self, is_semantic_scale=is_semantic_scale)
                    quantized, indices = lfq(interpolate_residual)
                    quantized = interpolate(quantized, size=tgt_shape, mode=self.z_interplote_up, quantizer=self, is_semantic_scale=is_semantic_scale)
                    all_indices.append(indices)
                    # all_losses.append(loss)
                    residual[:, :, ss:ee, :, :] = residual[:, :, ss:ee, :, :] - quantized
                    quantized_out = quantized_out + quantized
                if si == tower_split_index - 1:
                    quantized_out_firstframe = quantized_out.clone()
                    quantized_out = 0

            if quantized_out_firstframe is not None:
                if len(scale_schedule) == tower_split_index:
                    quantized_out = quantized_out_firstframe
                else:
                    quantized_out = torch.cat([quantized_out_firstframe, quantized_out], dim=2)

        # stack all losses and indices

        all_losses = None

        ret = (quantized_out, all_indices, all_losses)

        if not return_all_codes:
            return ret

        # whether to return all codes from all codebooks across layers
        all_codes = self.get_codes_from_indices(all_indices)

        # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)

        return (*ret, all_codes)
