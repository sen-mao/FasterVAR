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

