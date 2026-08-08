# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Definitions of blocks of VAR transformer model.
"""

import math
import os
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from infinity.models.rope import apply_rotary_emb
from infinity.utils.sequence_parallel import sp_all_to_all, SequenceParallelManager as sp_manager

# Import flash_attn's fused ops
try:
    from flash_attn.ops.rms_norm import rms_norm as rms_norm_impl
    from flash_attn.ops.fused_dense import fused_mlp_func
    flash_fused_op_installed = True
except ImportError:
    fused_mlp_func = None
    flash_fused_op_installed = False
    
    def rms_norm_impl(x, weight, epsilon):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(epsilon))) * weight


class FastRMSNorm(nn.Module):
    def __init__(self, C, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.C = C
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(C))
        else:
            self.register_buffer('weight', torch.ones(C))
    
    def forward(self, x):
        src_type = x.dtype
        return rms_norm_impl(x.float(), self.weight, epsilon=self.eps).to(src_type)
    
    def extra_repr(self) -> str:
        return f'C={self.C}, eps={self.eps:g}, elementwise_affine={self.elementwise_affine}'


def get_dropout_layer(p):
    return nn.Dropout(p, inplace=True) if p > 0 else nn.Identity()


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_mlp=False):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_mlp else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = get_dropout_layer(drop)
        self.heuristic = -1
    
    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x,
                weight1=self.fc1.weight,
                weight2=self.fc2.weight,
                bias1=self.fc1.bias,
                bias2=self.fc2.bias,
                activation='gelu_approx',
                save_pre_act=self.training,
                return_residual=False,
                checkpoint_lvl=0,
                heuristic=self.heuristic,
                process_group=None,
            ))
        else:
            return self.drop(self.fc2(self.act(self.fc1(x))))
    
    def extra_repr(self) -> str:
        return f'fused_mlp={self.fused_mlp_func is not None}'

class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class FFNSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, drop=0., fused_mlp=False):
        super().__init__()
        self.fused_mlp_func = None
        hidden_features = round(2 * hidden_features / 3 / 256) * 256
        
        out_features = out_features or in_features
        self.fcg = nn.Linear(in_features, hidden_features, bias=False)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = get_dropout_layer(drop)
    
    def forward(self, x):
        return self.drop(self.fc2( F.silu(self.fcg(x), inplace=True).mul_(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp={self.fused_mlp_func is not None}'

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class SelfAttention(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12, num_key_value_heads=-1,
        use_flex_attn=False, 
        pad_to_multiplier=1, rope2d_normalized_by_hw=0,
        mask_type='var', context_frames=1000000, steps_per_frame=4,
        arch='var',
        qwen_qkvo_bias=False,
    ):
        """
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        assert num_key_value_heads == -1 or num_heads % num_key_value_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads > 0 else num_heads
        self.arch = arch
        if self.arch == 'qwen':
            self.q_proj = nn.Linear(embed_dim, self.num_heads*self.head_dim, bias=qwen_qkvo_bias)
            self.k_proj = nn.Linear(embed_dim, self.num_key_value_heads*self.head_dim, bias=qwen_qkvo_bias)
            self.v_proj = nn.Linear(embed_dim, self.num_key_value_heads*self.head_dim, bias=qwen_qkvo_bias)
            self.o_proj = nn.Linear(self.num_heads*self.head_dim, embed_dim, bias=qwen_qkvo_bias)
            self.q_norm = FastRMSNorm(self.head_dim)
            self.k_norm = FastRMSNorm(self.head_dim)
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        else:
            raise ValueError(f'arch {self.arch} not supported')
        
        self.caching = False    # kv caching: only used during inference
        self.cached_k = {}    # kv caching: only used during inference
        self.cached_v = {}    # kv caching: only used during inference

        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier

        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        self.mask_type = mask_type
        self.context_frames = context_frames
        self.steps_per_frame = steps_per_frame
    
    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = {}
        self.cached_v = {}

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], attn_fn=None, rope2d_freqs_grid=[], scale_schedule=[], scale_ind=0, context_info=None, last_repetition_step=True, ref_text_scale_inds=[]):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be sharded (L = raw_seq_len//sp_size)
        :param (fp32) attn_bias_or_two_vector:
                if not using_flash:
                    a block-wise, lower-triangle matrix, like:
                    [[[[0, -, -, -, -, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
                    where 0 means visible and - means invisible (-inf)
                else:
                    a tuple of two 1-dim int vector (VAR_visible_kvlen, VAR_invisible_qlen)
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be sharded
        """
        # x: fp32
        B, L, C = x.shape

        if self.arch == 'qwen':
            hidden_states = x
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2) # batch, num_key_value_heads, slen, head_dim
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) # batch, num_key_value_heads, slen, head_dim

            if sp_manager.sp_on():
                # Headnum need to be sharded and L needs to be gathered
                # [B, H, raw_L/sp, C] --> [B, H/sp, raw_L, C]
                sdim = 1
                gdim = 2
                L = L * sp_manager.get_sp_size()
                C = C // sp_manager.get_sp_size()
                query_states = sp_all_to_all(query_states, sdim, gdim)
                key_states = sp_all_to_all(key_states, sdim, gdim)
                value_states = sp_all_to_all(value_states, sdim, gdim)

            query_states, key_states = apply_rotary_emb(query_states, key_states, rope2d_freqs_grid)
            if self.caching:    # kv caching: only used during inference
                if last_repetition_step:
                    self.cached_k[scale_ind] = key_states
                    self.cached_v[scale_ind] = value_states
                if isinstance(scale_ind, int):
                    ref_scale_inds = context_info[scale_ind]['ref_sids'] + ref_text_scale_inds
                    key_states = torch.cat([self.cached_k[ind] for ind in ref_scale_inds] + [key_states], dim=2)
                    value_states = torch.cat([self.cached_v[ind] for ind in ref_scale_inds] + [value_states], dim=2)
                
                    ref_scale_2_last_use_scale = [-1 for _ in range(len(context_info))]
                    for si in range(len(context_info)):
                        for ref_si in context_info[si]['ref_sids']:
                            ref_scale_2_last_use_scale[ref_si] = si
                    for ref_si in range(scale_ind):
                        if (ref_scale_2_last_use_scale[ref_si] < scale_ind) and (self.cached_k[ref_si] is not None):
                            tmpk, tmpv = self.cached_k[ref_si], self.cached_v[ref_si]
                            self.cached_k[ref_si], self.cached_v[ref_si] = None, None
                            del tmpk, tmpv

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            scale = self.head_dim**-0.5
            if self.use_flex_attn and attn_fn is not None:
                attn_output = attn_fn(query_states.to(value_states.dtype), key_states.to(value_states.dtype), value_states, scale=scale).transpose(1, 2).reshape(B, L, C)
            else:
                # fa2, flash_attn_func input/output should be (batch_size, seqlen, nheads, headdim)
                from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
                attn_output = flash_attn_func(query_states.permute([0,2,1,3]).to(torch.bfloat16), key_states.permute([0,2,1,3]).to(torch.bfloat16), value_states.permute([0,2,1,3]).to(torch.bfloat16), softmax_scale=scale)
                attn_output = attn_output.reshape(B, L, C)

                # fa3, flash_attn_func input/output should be (batch_size, seqlen, nheads, headdim)
                # from flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_func
                # attn_output = flash_attn_func(query_states.permute([0,2,1,3]).to(torch.bfloat16), key_states.permute([0,2,1,3]).to(torch.bfloat16), value_states.permute([0,2,1,3]).to(torch.bfloat16), softmax_scale=scale)
                # attn_output = attn_output[0].reshape(B, L, C)
                
                # slow attn
                # attn_output = slow_attn(query=query_states, key=key_states, value=value_states, scale=scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, L, C)
            if sp_manager.sp_on():
                # [B, raw_L, C/sp] --> [B, raw_L/sp, C]
                sdim = 1
                gdim = 2
                attn_output = sp_all_to_all(attn_output, sdim, gdim)

            attn_output = self.o_proj(attn_output)

            return attn_output
        
        # qkv: amp, bf16
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)  # BL3Hc
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); L_dim = 2   # q or k or v: all are shaped in (B:batch_size, H:heads, L:seq_len, c:head_dim), this way
        
        scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
        q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()   # fp32
        k = F.normalize(k, dim=-1, eps=1e-12).contiguous()                  # fp32
        v = v.contiguous()                                                  # bf16

        if sp_manager.sp_on():
            # Headnum need to be sharded and L needs to be gathered
            # [B, H, raw_L/sp, C] --> [B, H/sp, raw_L, C]
            sdim = 1
            gdim = 2

            L = L * sp_manager.get_sp_size()
            C = C // sp_manager.get_sp_size()

            q = sp_all_to_all(q, sdim, gdim)
            k = sp_all_to_all(k, sdim, gdim)
            v = sp_all_to_all(v, sdim, gdim)


        q, k = apply_rotary_emb(q, k, rope2d_freqs_grid) #, freqs_cis=freqs_cis)
        if self.caching:    # kv caching: only used during inference
            if last_repetition_step:
                self.cached_k.append(k)
                self.cached_v.append(v)
            if scale_ind >= 0:
                ref_scale_inds = context_info[scale_ind]['ref_sids']
                k = torch.cat([self.cached_k[0]] + [self.cached_k[ind+1] for ind in ref_scale_inds] + [k], dim=L_dim)
                v = torch.cat([self.cached_v[0]] + [self.cached_v[ind+1] for ind in ref_scale_inds] + [v], dim=L_dim)

            ref_scale_2_last_use_scale = [-1 for _ in range(len(context_info))]
            for si in range(len(context_info)):
                for ref_si in context_info[si]['ref_sids']:
                    ref_scale_2_last_use_scale[ref_si] = si
            for ref_si in range(scale_ind):
                if (ref_scale_2_last_use_scale[ref_si] < scale_ind) and (self.cached_k[ref_si+1] is not None):
                    tmpk, tmpv = self.cached_k[ref_si+1], self.cached_v[ref_si+1]
                    self.cached_k[ref_si+1], self.cached_v[ref_si+1] = None, None
                    del tmpk, tmpv
        
        # if self.cos_attn: q, k are in fp32; v is in bf16
        # else: q, k, v are in bf16
        if self.use_flex_attn and attn_fn is not None:
            oup = attn_fn(q.to(v.dtype), k.to(v.dtype), v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
        else:
            # oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, L, C)
            # fa2, flash_attn_func input/output should be (batch_size, seqlen, nheads, headdim)
            from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
            oup = flash_attn_func(q.permute([0,2,1,3]).to(torch.bfloat16), k.permute([0,2,1,3]).to(torch.bfloat16), v.permute([0,2,1,3]).to(torch.bfloat16), softmax_scale=self.scale)
            oup = oup.reshape(B, L, C)
        # oup: bf16

        if sp_manager.sp_on():
            # [B, raw_L, C/sp] --> [B, raw_L/sp, C]
            sdim = 1
            gdim = 2
            oup = sp_all_to_all(oup, sdim, gdim)

        return self.proj_drop(self.proj(oup))
    
class SelfAttnBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        cond_dim,
        num_heads,
        num_key_value_heads,
        mlp_ratio=4.0,
        use_flex_attn=False,
        pad_to_multiplier=1,
        rope2d_normalized_by_hw=False,
        mask_type="",
        context_frames=-1,
        steps_per_frame=-1,
        arch="var",
        qwen_qkvo_bias=False,
        inject_sync=False,
    ):
        super(SelfAttnBlock, self).__init__()
        self.C, self.D = embed_dim, cond_dim
        self.arch=arch
        self.attn = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, num_key_value_heads=num_key_value_heads,
            use_flex_attn=use_flex_attn, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            mask_type=mask_type, context_frames=context_frames, steps_per_frame=steps_per_frame, arch=arch, qwen_qkvo_bias=qwen_qkvo_bias,
        )
        if self.arch == 'qwen':
            self.mlp = Qwen3MLP(hidden_size=embed_dim, intermediate_size=round(embed_dim * mlp_ratio / 256) * 256)
            self.input_layernorm = FastRMSNorm(embed_dim)
            self.post_attention_layernorm = FastRMSNorm(embed_dim)
            self.inject_sync = inject_sync
        else:
            raise ValueError(f'arch {self.arch} not supported')
        
    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, rope2d_freqs_grid=[], scale_schedule=[], scale_ind=0, context_info=None, last_repetition_step=True, ref_text_scale_inds=[]):
        residual = x
        hidden_states = x
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attn(hidden_states, attn_bias_or_two_vector, attn_fn, rope2d_freqs_grid, scale_schedule, scale_ind, context_info, last_repetition_step, ref_text_scale_inds)
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    

if __name__ == '__main__':
    pass
