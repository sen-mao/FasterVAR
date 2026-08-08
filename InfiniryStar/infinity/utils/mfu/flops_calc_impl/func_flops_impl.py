# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import torch
from typing import List, Optional

Tensor = torch.Tensor

def _prod(dims):
    p = 1
    for v in dims:
        p *= v
    return p


def linear_flops_compute(input, weight, bias=None):
    out_features = weight.shape[0]
    macs = input.numel() * out_features
    return 2 * macs, macs


def relu_flops_compute(input, inplace=False):
    return input.numel(), 0


def prelu_flops_compute(input: Tensor, weight: Tensor):
    return input.numel(), 0


def elu_flops_compute(input: Tensor, alpha: float = 1.0, inplace: bool = False):
    return input.numel(), 0


def leaky_relu_flops_compute(input: Tensor, negative_slope: float = 0.01, inplace: bool = False):
    return input.numel(), 0


def relu6_flops_compute(input: Tensor, inplace: bool = False):
    return input.numel(), 0


def silu_flops_compute(input: Tensor, inplace: bool = False):
    return input.numel(), 0


def gelu_flops_compute(input, **kwargs):
    return input.numel(), 0


def pool_flops_compute(input,
                        kernel_size,
                        stride=None,
                        padding=0,
                        dilation=None,
                        ceil_mode=False,
                        count_include_pad=True,
                        divisor_override=None,
                        return_indices=None):
    return input.numel(), 0


def conv_flops_compute(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    assert weight.shape[1] * groups == input.shape[1]

    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)

    strides = stride if type(stride) is tuple else (stride, ) * length
    dilations = dilation if type(dilation) is tuple else (dilation, ) * length
    if isinstance(padding, str):
        if padding == 'valid':
            paddings = (0, ) * length
        elif padding == 'same':
            paddings = ()
            for d, k in zip(dilations, kernel_dims):
                total_padding = d * (k - 1)
                paddings += (total_padding // 2, )
    elif isinstance(padding, tuple):
        paddings = padding
    else:
        paddings = (padding, ) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):
        output_dim = (input_dim + 2 * paddings[idx] - (dilations[idx] *
                                                       (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
        output_dims.append(output_dim)

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(_prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(output_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * active_elements_count

    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def conv_trans_flops_compute(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[1]
    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)

    paddings = padding if type(padding) is tuple else (padding, ) * length
    strides = stride if type(stride) is tuple else (stride, ) * length
    dilations = dilation if type(dilation) is tuple else (dilation, ) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):

        output_dim = (input_dim + 2 * paddings[idx] - (dilations[idx] *
                                                       (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
        output_dims.append(output_dim)

    paddings = padding if type(padding) is tuple else (padding, padding)
    strides = stride if type(stride) is tuple else (stride, stride)
    dilations = dilation if type(dilation) is tuple else (dilation, dilation)

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(_prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(input_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * batch_size * int(_prod(output_dims))

    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def batch_norm_flops_compute(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    has_affine = weight is not None
    if training:
        # estimation
        return input.numel() * (5 if has_affine else 4), 0
    flops = input.numel() * (2 if has_affine else 1)
    return flops, 0


def layer_norm_flops_compute(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4), 0


def group_norm_flops_compute(input: Tensor,
                              num_groups: int,
                              weight: Optional[Tensor] = None,
                              bias: Optional[Tensor] = None,
                              eps: float = 1e-5):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4), 0


def instance_norm_flops_compute(
    input: Tensor,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4), 0


def upsample_flops_compute(*args, **kwargs):
    input = args[0]
    size = kwargs.get('size', None)
    if size is None and len(args) > 1:
        size = args[1]

    if size is not None:
        if isinstance(size, tuple) or isinstance(size, list):
            return int(_prod(size)), 0
        else:
            return int(size), 0

    scale_factor = kwargs.get('scale_factor', None)
    if scale_factor is None and len(args) > 2:
        scale_factor = args[2]
    assert scale_factor is not None, "either size or scale_factor should be defined"

    flops = input.numel()
    if isinstance(scale_factor, tuple) and len(scale_factor) == len(input):
        flops *= int(_prod(scale_factor))
    else:
        flops *= scale_factor**len(input)
    return flops, 0


def softmax_flops_compute(input, dim=None, _stacklevel=3, dtype=None):
    return input.numel(), 0


def embedding_flops_compute(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    return 0, 0

def attn_flops_compute(query, key, value, *args, **kwargs):
    """
    Count flops for the scaled_dot_product_attention operation.
    """
    macs = _prod(q.shape) * k.shape[-2]
    macs += _prod(q.shape[:-1]) * k.shape[-2] * v.shape[-1]
    return 2 * macs, macs
