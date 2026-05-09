# -*- coding: utf-8 -*-
"""
地学评估指标 — RMSE, R²_Score, TotalSeismicMomentError
========================================================
提供模型性能评估的标准地球物理指标。
"""

import torch


def rmse(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    均方根误差 (Root Mean Squared Error)。

    Parameters
    ----------
    pred, true : Tensor
        形状相同的预测值和真实值。

    Returns
    -------
    Tensor (标量)
        RMSE 值。
    """
    return torch.sqrt(((pred - true) ** 2).mean())


def r2_score(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    决定系数 (R² Score / Coefficient of Determination)。

    R² = 1 - SS_res / SS_tot
    - R² = 1.0: 完美预测
    - R² = 0.0: 等效于均值预测
    - R² < 0:   比均值预测更差

    Parameters
    ----------
    pred, true : Tensor
        形状相同的预测值和真实值。

    Returns
    -------
    Tensor (标量)
        R² 值。
    """
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()

    # 防止除零
    if ss_tot < 1e-10:
        return torch.tensor(1.0) if ss_res < 1e-10 else torch.tensor(0.0)

    return 1.0 - ss_res / ss_tot


def total_seismic_moment_error(
    pred_slip: torch.Tensor,
    true_slip: torch.Tensor,
    shear_modulus: float = 1.0,
    cell_area: float = 1.0,
) -> torch.Tensor:
    """
    总地震矩误差 (Total Seismic Moment Error, ΔM₀)。

    M₀ = μ · A · Σ(Slip)

    衡量模型是否因"零陷阱"而系统性低估 SSE 的物理能量释放。

    Parameters
    ----------
    pred_slip : Tensor
        预测滑移场。
    true_slip : Tensor
        真实滑移场。
    shear_modulus : float
        剪切模量 μ（归一化时设为 1.0）。
    cell_area : float
        网格单元面积 A（归一化时设为 1.0）。

    Returns
    -------
    Tensor (标量)
        ΔM₀ = M₀_pred - M₀_true（正值 = 过估，负值 = 欠估）。
    """
    m0_pred = shear_modulus * cell_area * pred_slip.sum()
    m0_true = shear_modulus * cell_area * true_slip.sum()
    return m0_pred - m0_true
