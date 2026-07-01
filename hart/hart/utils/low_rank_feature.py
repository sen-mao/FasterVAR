import torch
import math

def get_rank_for_energy(S, sv_percent):
    """
    S: torch.Tensor of singular values, shape (n,) or (B, n)
    k_percent: float in (0, 1), desired retained energy ratio (e.g., 0.95)
    Returns:
        r: int or Tensor, number of singular values needed to retain k% energy
    """
    S_squared = S**2
    total_energy = S_squared.sum(dim=-1, keepdim=True)
    cumulative_energy = torch.cumsum(S_squared, dim=-1)
    energy_ratio = cumulative_energy / total_energy
    reached = energy_ratio >= float(sv_percent)
    q = S.shape[-1]
    r = torch.where(
        reached.any(dim=-1),
        reached.to(torch.int64).argmax(dim=-1) + 1,
        torch.full(S.shape[:-1], q, device=S.device, dtype=torch.int64),
    )
    return r.item() if S.dim() == 1 else r

def low_rank_approximation_percent(x, alpha):
    """
    对三维矩阵 X 做逐样本低秩近似（基于 torch.pca_lowrank）

    输入:
      X: torch.Tensor, shape=(B, m, n)
      alpha: float, threshold

    输出:
      x_approx: torch.Tensor, shape=(B, m, n)
    """
    assert x.dim() == 3, "输入必须是三维 (B, m, n)"

    B, m, n = x.shape
    q = min(m, n)
    last_stage = x.to(torch.float32)

    with torch.cuda.amp.autocast(enabled=False):
        last_stage_mean = last_stage.mean(dim=1, keepdim=True)
        U, S, V = torch.pca_lowrank(last_stage, q=q, center=True)
        rank_needed = get_rank_for_energy(S, alpha)

        rank_mask = torch.arange(q, device=last_stage.device).unsqueeze(0) < rank_needed.unsqueeze(1)
        S = S * rank_mask.to(S.dtype)
        x_approx = torch.bmm(U * S.unsqueeze(1), V.transpose(1, 2)) + last_stage_mean
        low_rank_ratio = (rank_needed.to(torch.float32) / q).tolist()

    return x_approx, low_rank_ratio

def random_projection_for_low_rank_feature(x: torch.Tensor, r: int = 128):
    """Project the token dimension with a random matrix for low-rank features.

    Args:
        last_stage: Tensor with shape (B, H, W), where H is token count and W is feature dimension.
        r: Target token rank.

    Returns:
        Tensor with shape (B, r, W).
    """
    b, H, W = x.shape
    token_type = x.dtype

    with torch.cuda.amp.autocast(enabled=False):
        x = x.to(torch.float32)
        Q = torch.randn(b, H, r, device=x.device) * (1.0 / math.sqrt(r))
        projected = torch.bmm(Q.transpose(1, 2), x)  # (B, r, W)

    return projected.to(token_type)

def representative_token_indices(x: torch.Tensor, p: int, c: float = 0.1, replace: bool = False):
    """按论文方案对 A 做行采样 (支持 batch 维度，torch 实现)

    Args:
        x      : (B, m, n) 的张量
        p      : 每个 batch 采样的行数
        c      : 常数 c<=1，控制概率下限
        replace: 是否放回采样

    Returns:
        idx : (B, p)，每个 batch 的采样索引
    """

    x = x.to(torch.float32)

    with torch.amp.autocast('cuda', enabled=False):
        # (B, m) 每行平方 L2 范数
        row_norm2 = torch.norm(x, dim=2) ** 2

        # (B, 1) 每个 batch 的 Frobenius 范数平方
        total_norm2 = row_norm2.sum(dim=1, keepdim=True)

        # (B, m) 概率分布
        prob = c * row_norm2 / total_norm2
        prob = prob / prob.sum(dim=1, keepdim=True)

        idx = torch.multinomial(prob, num_samples=p, replacement=replace)  # (B, p)

    return idx