# -*- coding: utf-8 -*-
"""
TDD 单元测试 — 损失函数与物理约束测试
========================================
验证 WeightedMSELoss 的零陷阱惩罚放大机制、
LaplacianSmoothnessLoss 的可导性、以及评估指标。
"""

import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.loss_functions import (  # noqa: E402
    WeightedMSELoss,
    LaplacianSmoothnessLoss,
    PhysicsOperatorLoss,
)
from src.utils.metrics import (  # noqa: E402
    rmse,
    r2_score,
    total_seismic_moment_error,
)
from src.models.layers import LearnedPhysicsLayer  # noqa: E402

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
B = 2
T = 5
H, W = 15, 202
THRESHOLD = 1e-4
WEIGHT_MULTIPLIER = 10.0


# ===========================================================================
# WeightedMSELoss 测试
# ===========================================================================
class TestWeightedMSELoss:
    """WeightedMSELoss：活跃区域 10 倍惩罚对抗零陷阱。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.loss_fn = WeightedMSELoss(
            threshold=THRESHOLD,
            weight_multiplier=WEIGHT_MULTIPLIER,
        )

    def test_active_region_gradient_amplification(self):
        """
        核心断言：活跃区域 (slip > 1e-4) 的梯度惩罚被放大了约 10 倍。
        使用统一非零预测偏差，确保活跃和非活跃区域都有梯度。
        """
        # 构建统一偏差预测（leaf tensor）
        pred = torch.full((B, T, H, W), 0.005, requires_grad=True)

        # 构建含活跃/非活跃区域的真实值
        true = torch.zeros(B, T, H, W)
        true[:, :, :, :101] = 0.01    # > 1e-4, 活跃区域
        true[:, :, :, 101:] = 1e-5    # < 1e-4, 非活跃区域（非零以产生梯度）

        loss = self.loss_fn(pred, true)
        loss.backward()

        grad = pred.grad

        # 活跃/非活跃区域梯度幅值
        active_mask = true > THRESHOLD
        inactive_mask = ~active_mask

        active_grad_mean = grad[active_mask].abs().mean()
        inactive_grad_mean = grad[inactive_mask].abs().mean()

        # 活跃区域梯度应约为非活跃区域的 10 倍
        ratio = active_grad_mean / inactive_grad_mean
        assert ratio > 8.0 and ratio < 12.0, (
            f"Gradient amplification ratio: expected ~10.0, got {ratio:.2f}"
        )

    def test_zero_trap_penalty(self):
        """全 0 预测 vs 有活跃值的真实值时，Loss > 0。"""
        pred = torch.zeros(B, T, H, W)
        true = torch.ones(B, T, H, W) * 0.01
        loss = self.loss_fn(pred, true)
        assert loss.item() > 0, "WeightedMSELoss should penalize zero predictions"

    def test_perfect_prediction(self):
        """预测等于真实时 Loss 应为 0。"""
        data = torch.randn(B, T, H, W)
        loss = self.loss_fn(data, data)
        assert loss.item() < 1e-7, f"Loss should be ~0 for perfect prediction: {loss.item()}"

    def test_differentiable(self):
        """Loss 必须可导。"""
        pred = torch.randn(B, T, H, W, requires_grad=True)
        true = torch.randn(B, T, H, W).abs()
        loss = self.loss_fn(pred, true)
        loss.backward()
        assert pred.grad is not None, "WeightedMSELoss is not differentiable"

    def test_returns_scalar(self):
        """Loss 输出必须是标量。"""
        pred = torch.randn(B, T, H, W)
        true = torch.randn(B, T, H, W)
        loss = self.loss_fn(pred, true)
        assert loss.dim() == 0


# ===========================================================================
# LaplacianSmoothnessLoss 测试
# ===========================================================================
class TestLaplacianSmoothnessLoss:
    """LaplacianSmoothnessLoss：3x3 拉普拉斯卷积核的 L1 范数。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.loss_fn = LaplacianSmoothnessLoss()

    def test_smooth_input_low_loss(self):
        """平滑输入（常数场）的 Laplacian Loss 应远低于噪声输入。
        注意: padding=1 在边界会引入少量非零值，因此允许小幅容差。"""
        smooth = torch.ones(B, 1, H, W) * 5.0
        noisy = torch.randn(B, 1, H, W) * 10.0
        loss_smooth = self.loss_fn(smooth)
        loss_noisy = self.loss_fn(noisy)
        # 平滑场的 Loss 应远小于噪声场
        assert loss_smooth < loss_noisy * 0.1, (
            f"Smooth loss ({loss_smooth:.4f}) should be << noisy loss ({loss_noisy:.4f})"
        )

    def test_noisy_input_higher_loss(self):
        """噪声输入的 Loss 应显著高于平滑输入。"""
        smooth = torch.ones(B, 1, H, W) * 5.0
        noisy = torch.randn(B, 1, H, W) * 10.0
        loss_smooth = self.loss_fn(smooth)
        loss_noisy = self.loss_fn(noisy)
        assert loss_noisy > loss_smooth * 10, (
            f"Noisy loss ({loss_noisy:.4f}) should be >> smooth loss ({loss_smooth:.4f})"
        )

    def test_checkerboard_detection(self):
        """棋盘格模式应产生高 Loss（这正是该 Loss 要防止的）。"""
        checkerboard = torch.zeros(B, 1, H, W)
        checkerboard[:, :, ::2, ::2] = 1.0
        checkerboard[:, :, 1::2, 1::2] = 1.0
        loss = self.loss_fn(checkerboard)
        assert loss.item() > 0.1, f"Checkerboard loss too low: {loss.item()}"

    def test_differentiable(self):
        """Loss 必须可导。"""
        x = torch.randn(B, 1, H, W, requires_grad=True)
        loss = self.loss_fn(x)
        loss.backward()
        assert x.grad is not None

    def test_returns_scalar(self):
        """输出必须是标量。"""
        x = torch.randn(B, 1, H, W)
        loss = self.loss_fn(x)
        assert loss.dim() == 0

    def test_accepts_4d_input(self):
        """接受 [B, 1, H, W] 输入。"""
        x = torch.randn(B, 1, H, W)
        loss = self.loss_fn(x)
        assert isinstance(loss, torch.Tensor)


# ===========================================================================
# PhysicsOperatorLoss 测试
# ===========================================================================
class TestPhysicsOperatorLoss:
    """PhysicsOperatorLoss: MSE(G @ True_Slip, True_GNSS)"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.physics_layer = LearnedPhysicsLayer(in_features=3030, out_features=9)
        self.loss_fn = PhysicsOperatorLoss()

    def test_output_scalar(self):
        """输出必须是标量。"""
        true_slip = torch.randn(B, 3030)
        true_gnss = torch.randn(B, 9)
        loss = self.loss_fn(self.physics_layer, true_slip, true_gnss)
        assert loss.dim() == 0

    def test_differentiable_wrt_physics_layer(self):
        """Loss 必须对 physics_layer 的权重可导。"""
        true_slip = torch.randn(B, 3030)
        true_gnss = torch.randn(B, 9)
        loss = self.loss_fn(self.physics_layer, true_slip, true_gnss)
        loss.backward()
        assert self.physics_layer.greens_function.weight.grad is not None

    def test_perfect_reconstruction_low_loss(self):
        """当 G 完美映射时 Loss 应接近 0。"""
        true_slip = torch.randn(1, 3030)
        with torch.no_grad():
            true_gnss = self.physics_layer(true_slip)
        loss = self.loss_fn(self.physics_layer, true_slip, true_gnss)
        assert loss.item() < 1e-10, f"Perfect reconstruction loss: {loss.item()}"


# ===========================================================================
# Metrics 测试
# ===========================================================================
class TestMetrics:
    """RMSE, R2_Score, TotalSeismicMomentError 测试。"""

    def test_rmse_zero(self):
        """相同输入 RMSE 应为 0。"""
        x = torch.randn(10, 15, 202)
        assert rmse(x, x).item() < 1e-7

    def test_rmse_positive(self):
        """不同输入 RMSE > 0。"""
        pred = torch.randn(10, 15, 202)
        true = torch.randn(10, 15, 202)
        assert rmse(pred, true).item() > 0

    def test_r2_perfect(self):
        """完美预测 R² = 1.0。"""
        x = torch.randn(10, 15, 202)
        r2 = r2_score(x, x)
        assert abs(r2.item() - 1.0) < 1e-5

    def test_r2_mean_prediction(self):
        """用均值预测时 R² ≈ 0。"""
        true = torch.randn(100)
        pred = torch.full_like(true, true.mean())
        r2 = r2_score(pred, true)
        assert abs(r2.item()) < 0.1

    def test_r2_bad_prediction(self):
        """糟糕预测 R² < 0。"""
        true = torch.randn(100)
        pred = -true * 10  # 完全反向
        r2 = r2_score(pred, true)
        assert r2.item() < 0

    def test_seismic_moment_error_zero(self):
        """相同输入地震矩误差为 0。"""
        x = torch.randn(10, 15, 202).abs()
        err = total_seismic_moment_error(x, x)
        assert abs(err.item()) < 1e-5

    def test_seismic_moment_error_sign(self):
        """过估的预测 → 正误差，欠估 → 负误差。"""
        true = torch.ones(5, 15, 202)
        over_pred = torch.ones(5, 15, 202) * 2.0
        under_pred = torch.ones(5, 15, 202) * 0.5
        err_over = total_seismic_moment_error(over_pred, true)
        err_under = total_seismic_moment_error(under_pred, true)
        assert err_over.item() > 0, "Over-prediction should give positive M0 error"
        assert err_under.item() < 0, "Under-prediction should give negative M0 error"
