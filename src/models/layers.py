# -*- coding: utf-8 -*-
"""
底层算子模块 — ChannelMixer, TemporalSEBlock, ConvGRUCell, LearnedPhysicsLayer
================================================================================
DeepSeismic_v3_STCR 网络的四大核心组件。

设计遵循《技术需求文档 V3.0》第三部分的微观约束。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ===========================================================================
# 1. ChannelMixer — GNSS 共模误差 (CME) 过滤器
# ===========================================================================
class ChannelMixer(nn.Module):
    """
    轻量级通道混合器，计算 GNSS 台站间差分特征以滤除共模误差 (CME)。

    架构: Linear(9→32) → BatchNorm1d → ReLU

    Parameters
    ----------
    in_features : int
        输入维度，默认 9 (3 台站 × 3 分量)。
    out_features : int
        输出维度，默认 32。
    """

    def __init__(self, in_features: int = 9, out_features: int = 32) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [B, 9]
            当前时刻 GNSS 观测。

        Returns
        -------
        Tensor, shape [B, 32]
            过滤 CME 后的差分特征。
        """
        out = self.linear(x)       # [B, 32]
        out = self.bn(out)         # [B, 32]
        out = self.relu(out)       # [B, 32]
        return out


# ===========================================================================
# 2. TemporalSEBlock — 时间注意力机制 (Squeeze-and-Excitation)
# ===========================================================================
class TemporalSEBlock(nn.Module):
    """
    时间轴 Squeeze-and-Excitation 注意力模块。

    在时间维度上计算注意力权重，帮助模型聚焦包含瞬态慢滑移信号的
    关键时间窗口，抑制高频噪声时段。

    机制:
      1. Squeeze: 沿特征维度 (C) 全局平均池化 → [B, 1, C]
      2. Excitation: FC(C→C//r)→ReLU→FC(C//r→C)→Sigmoid → [B, 1, C]
      3. 沿时间轴广播→逐元素乘，对每个特征通道独立加权

    实际实现中，为了给每个 **时间步** 独立的权重，我们采用如下策略：
      - Squeeze: 沿 C 维度均值池化 → [B, T, 1]
      - Excitation: FC(1→mid)→ReLU→FC(mid→1)→Sigmoid → [B, T, 1]
      - Scale: 广播乘以原始 [B, T, C]

    Parameters
    ----------
    n_channels : int
        输入特征通道数（仅用于计算 reduction 中间维度的下界）。
    reduction : int
        缩减比例，默认 4。
    """

    def __init__(self, n_channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid_dim = max(n_channels // reduction, 1)
        # 输入: [B, T, 1] → 输出: [B, T, 1]
        self.squeeze_excite = nn.Sequential(
            nn.Linear(1, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, 1),
            nn.Sigmoid(),
        )

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算时间注意力权重（不乘以输入）。

        Parameters
        ----------
        x : Tensor, shape [B, T, C]

        Returns
        -------
        Tensor, shape [B, T, 1]
            注意力权重，值域 [0, 1]。
        """
        # Squeeze: 沿特征维度均值池化 → [B, T, 1]
        squeeze = x.mean(dim=2, keepdim=True)        # [B, T, 1]
        # Excitation: FC(1→mid)→ReLU→FC(mid→1)→Sigmoid → [B, T, 1]
        weights = self.squeeze_excite(squeeze)         # [B, T, 1]
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [B, T, C]
            时间序列特征。

        Returns
        -------
        Tensor, shape [B, T, C]
            注意力加权后的特征（形状不变）。
        """
        weights = self.get_attention_weights(x)        # [B, T, 1]
        return x * weights                             # [B, T, C]


# ===========================================================================
# 3. ConvGRUCell — 时空演化引擎的基本单元
# ===========================================================================
class ConvGRUCell(nn.Module):
    """
    卷积门控循环单元 (ConvGRU) 的单步 Cell。

    使用 2D 卷积代替全连接，保持空间局部归纳偏置 (Inductive Bias)。
    kernel_size=3, padding=1 确保空间维度 [H, W] 在前向传播中严格不变。

    GRU 门控方程:
      z_t = σ(Conv([x_t, h_{t-1}]))         — 更新门
      r_t = σ(Conv([x_t, h_{t-1}]))         — 重置门
      h̃_t = tanh(Conv([x_t, r_t ⊙ h_{t-1}])) — 候选隐状态
      h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  — 最终隐状态

    Parameters
    ----------
    input_channels : int
        输入通道数（默认 2: 拼接 Encoded_GNSS + 上一时刻 Slip）。
    hidden_dim : int
        隐状态通道数（默认 64）。
    kernel_size : int
        卷积核大小（默认 3）。
    padding : int
        填充（默认 1，保持空间维度不变）。
    """

    def __init__(
        self,
        input_channels: int = 2,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim

        # 更新门 z 和重置门 r：输入为 [x_t, h_{t-1}] 的通道拼接
        self.conv_gates = nn.Conv2d(
            in_channels=input_channels + hidden_dim,
            out_channels=2 * hidden_dim,   # z 和 r 各一份
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        # 候选隐状态 h̃：输入为 [x_t, r_t ⊙ h_{t-1}]
        self.conv_candidate = nn.Conv2d(
            in_channels=input_channels + hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        ConvGRU 单步前向传播。

        Parameters
        ----------
        x : Tensor, shape [B, input_channels, H, W]
            当前时刻输入。
        h_prev : Tensor or None, shape [B, hidden_dim, H, W]
            上一时刻隐状态。None 时自动初始化为全零。

        Returns
        -------
        h_next : Tensor, shape [B, hidden_dim, H, W]
            当前时刻隐状态。
        """
        # 自动初始化隐状态
        if h_prev is None:
            h_prev = torch.zeros(
                x.size(0), self.hidden_dim, x.size(2), x.size(3),
                dtype=x.dtype, device=x.device,
            )

        # 拼接输入与隐状态 → [B, input_channels + hidden_dim, H, W]
        combined = torch.cat([x, h_prev], dim=1)

        # 更新门 z + 重置门 r
        gates = self.conv_gates(combined)                     # [B, 2*hidden_dim, H, W]
        z, r = gates.chunk(2, dim=1)                          # 各 [B, hidden_dim, H, W]
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        # 候选隐状态
        combined_r = torch.cat([x, r * h_prev], dim=1)       # [B, input_ch + hidden, H, W]
        h_candidate = torch.tanh(self.conv_candidate(combined_r))  # [B, hidden_dim, H, W]

        # 最终隐状态
        h_next = (1 - z) * h_prev + z * h_candidate          # [B, hidden_dim, H, W]

        return h_next


# ===========================================================================
# 4. LearnedPhysicsLayer — 物理一致性算子层 (Pseudo-Okada)
# ===========================================================================
class LearnedPhysicsLayer(nn.Module):
    """
    可学习格林函数 / 伪 Okada 算子。

    使用无偏置线性层 nn.Linear(3030, 9, bias=False) 学习地下滑移到
    地表 GNSS 位移的物理映射矩阵 G。

    正向传播: Pred_GNSS = G @ Flatten(Pred_Slip)

    该层仅用于计算物理一致性 Loss（不参与反演梯度的回传），
    配合 L1 正则化实现远场衰减稀疏性。

    Parameters
    ----------
    in_features : int
        输入维度，默认 3030 (滑移向量维度)。
    out_features : int
        输出维度，默认 9 (GNSS 维度)。
    """

    def __init__(self, in_features: int = 3030, out_features: int = 9) -> None:
        super().__init__()
        self.greens_function = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [B, 3030] or [B, T, 3030]
            展平的滑移向量。

        Returns
        -------
        Tensor, shape [B, 9] or [B, T, 9]
            预测 GNSS 位移。
        """
        return self.greens_function(x)

    def l1_regularization(self) -> torch.Tensor:
        """
        计算权重矩阵的 L1 正则化项。

        Returns
        -------
        Tensor (标量)
            权重矩阵元素绝对值之和。
        """
        return self.greens_function.weight.abs().sum()

    def __repr__(self) -> str:
        w = self.greens_function.weight
        return (
            f"LearnedPhysicsLayer(\n"
            f"  greens_function: Linear({w.shape[1]} → {w.shape[0]}, bias=False),\n"
            f"  weight_sparsity: {(w.abs() < 1e-4).float().mean():.2%}\n"
            f")"
        )
