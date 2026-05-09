# -*- coding: utf-8 -*-
"""
定制损失函数 — WeightedMSELoss, LaplacianSmoothnessLoss, PhysicsOperatorLoss
==============================================================================
实现《技术需求文档 V3.0》第四部分的损失函数体系:
  L_total = L_WMSE + λ1·L_Phy_GNSS + λ2·L_Laplacian + λ3·L_Operator + λ4·L1_Sparsity

所有 Loss 保持可导 (autograd compatible)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 1. WeightedMSELoss — 加权 MSE（对抗零陷阱）
# ===========================================================================
class WeightedMSELoss(nn.Module):
    """
    加权均方误差损失，对活跃滑移区域施加额外惩罚。

    当 True_Slip > threshold (默认 1e-4) 时，该区域的 MSE 权重
    被放大 weight_multiplier 倍（默认 10 倍），迫使模型不能通过
    "全部输出 0" 来骗取低 Loss（零陷阱 / Zero Trap）。

    Parameters
    ----------
    threshold : float
        活跃滑移阈值。
    weight_multiplier : float
        活跃区域权重倍数。
    """

    def __init__(
        self,
        threshold: float = 1e-4,
        weight_multiplier: float = 1000.0,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.weight_multiplier = weight_multiplier

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : Tensor
            预测滑移，任意形状。
        true : Tensor
            真实滑移，与 pred 形状相同。

        Returns
        -------
        Tensor (标量)
            加权 MSE Loss。
        """
        # 逐元素 MSE
        mse = (pred - true) ** 2

        # 构建权重矩阵：活跃区域权重 = weight_multiplier，非活跃 = 1
        weights = torch.ones_like(true)
        active_mask = true.abs() > self.threshold
        weights[active_mask] = self.weight_multiplier

        # 加权求均值
        weighted_mse = (mse * weights).mean()

        return weighted_mse


# ===========================================================================
# 2. LaplacianSmoothnessLoss — 空间平滑正则化（防棋盘格伪影）
# ===========================================================================
class LaplacianSmoothnessLoss(nn.Module):
    """
    拉普拉斯平滑正则化损失。

    使用固定的 3x3 拉普拉斯卷积核提取滑移网格的二阶空间导数，
    计算其 L1 范数，防止出现棋盘格状不连续物理伪影。

    Laplacian 核:
        [[ 0,  1,  0],
         [ 1, -4,  1],
         [ 0,  1,  0]]
    """

    def __init__(self) -> None:
        super().__init__()
        # 固定 3x3 拉普拉斯核（不参与梯度更新）
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3)
        # 注册为 buffer（不是 parameter，不会被优化器更新）
        self.register_buffer("laplacian_kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [B, 1, H, W]
            滑移场。

        Returns
        -------
        Tensor (标量)
            拉普拉斯二阶导数的 L1 范数均值。
        """
        # 确保输入为 4D
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # 卷积提取二阶导数
        laplacian = F.conv2d(
            x,
            self.laplacian_kernel.to(x.device),
            padding=1,
        )

        # L1 范数均值
        return laplacian.abs().mean()


# ===========================================================================
# 3. PhysicsOperatorLoss — 系统辨识损失
# ===========================================================================
class PhysicsOperatorLoss(nn.Module):
    """
    物理算子系统辨识损失。

    计算 MSE(G @ True_Slip, True_GNSS)，
    迫使 LearnedPhysicsLayer 的无偏置线性层逼近真实的格林函数 G。

    用于 Stage 1 物理层预热训练。
    """

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        physics_layer: nn.Module,
        true_slip: torch.Tensor,
        true_gnss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        physics_layer : LearnedPhysicsLayer
            可学习格林函数层。
        true_slip : Tensor, shape [B, 3030] or [B, T, 3030]
            真实滑移向量。
        true_gnss : Tensor, shape [B, 9] or [B, T, 9]
            真实 GNSS 观测。

        Returns
        -------
        Tensor (标量)
            MSE(G(True_Slip), True_GNSS)。
        """
        pred_gnss = physics_layer(true_slip)
        return self.mse(pred_gnss, true_gnss)
