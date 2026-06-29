import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image
import os
from sklearn.decomposition import PCA


def global_feature_entropy(feature_map, eps=1e-12):
    """
    计算整张特征图的熵（全局）
    输入:
        feature_map: Tensor, shape (B, C, H, W)
    返回:
        Tensor of shape (B,)
    """
    B = feature_map.shape[0]
    x = feature_map.contiguous().view(B, -1)  # (B, C*H*W)
    p = torch.softmax(x, dim=-1)
    p_clamped = torch.clamp(p, eps, 1.0)
    entropy = -torch.sum(p_clamped * torch.log(p_clamped), dim=-1)
    return entropy


def feature_frobenius_norm(feature_map):
    """
    计算每张feature map的Frobenius norm
    输入:
        feature_map: shape (B, C, H, W)
    返回:
        shape (B,) 的张量，表示每个样本的F-norm
    """
    B = feature_map.shape[0]
    return torch.norm(feature_map.contiguous().view(B, -1), p='fro', dim=1)


def compute_global_mean_var(feature_map):
    """
    计算每张特征图的全局均值和方差（跨 C, H, W 维度）

    参数:
        feature_map: Tensor, shape (B, C, H, W)

    返回:
        mean: Tensor, shape (B,)
        var:  Tensor, shape (B,)
    """
    B = feature_map.shape[0]
    mean = feature_map.contiguous().view(B, -1).mean(dim=1)  # 每张图一个均值
    var = feature_map.contiguous().view(B, -1).var(dim=1, unbiased=False)  # 每张图一个方差（无偏或有偏都可）
    return mean, var


def compute_high_freq_ratio(feature_map, ratio_threshold=0.5):
    """
    计算特征图中高频能量占比，衡量“混乱程度”
    输入:
        feature_map: Tensor, shape (B, C, H, W)
        ratio_threshold: 高频 vs 低频 的分界半径（0~1，越小低频越小）
    返回:
        high_freq_ratio: shape (B,)
    """
    b, c, s, h, w = feature_map.shape
    feature_map = feature_map.reshape(b, c * s, h, w)

    B, C, H, W = feature_map.shape
    x = feature_map.clone()

    # 对每个通道做FFT，计算频谱幅值
    fft = torch.fft.fft2(x, norm='ortho')  # (B, C, H, W), complex
    fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))  # 中心化频谱

    magnitude = torch.abs(fft_shift)  # 幅值代表能量

    # 创建高频掩码
    y = torch.linspace(-1, 1, H, device=feature_map.device).view(1, H, 1)
    x = torch.linspace(-1, 1, W, device=feature_map.device).view(1, 1, W)
    dist = torch.sqrt(x**2 + y**2)  # 距离频谱中心的距离

    high_mask = (dist >= ratio_threshold).float()  # 高频区域掩码
    low_mask = 1.0 - high_mask

    # 展开用于广播
    high_mask = high_mask.unsqueeze(0)  # (1, 1, H, W)
    low_mask = low_mask.unsqueeze(0)

    high_energy = (magnitude * high_mask).pow(2).sum(dim=(2, 3))  # (B, C)
    total_energy = magnitude.pow(2).sum(dim=(2, 3)) + 1e-8        # (B, C)
    high_ratio = high_energy / total_energy                       # (B, C)

    return high_ratio.mean(dim=1)  # 平均得到每张图的高频比例 (B,)
