# -*- coding: utf-8 -*-
"""
TDD 单元测试 — 底层算子维度与属性断言
==========================================
针对 ChannelMixer, TemporalSEBlock, ConvGRUCell, LearnedPhysicsLayer
验证输入输出维度、参数属性、激活约束等。
"""

import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.layers import (  # noqa: E402
    ChannelMixer,
    TemporalSEBlock,
    ConvGRUCell,
    LearnedPhysicsLayer,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
B = 4             # Batch size
T = 10            # 时间步（测试用短序列）
N_GNSS = 9        # GNSS 维度
HIDDEN_DIM = 32   # ChannelMixer 隐藏维度
H, W = 15, 202    # 空间网格
CONV_HIDDEN = 64  # ConvGRU hidden dim
N_SLIP = 3030     # 滑移向量维度


# ===========================================================================
# ChannelMixer 测试
# ===========================================================================
class TestChannelMixer:
    """ChannelMixer: [B, 9] → [B, 32]"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.mixer = ChannelMixer(in_features=N_GNSS, out_features=HIDDEN_DIM)
        self.mixer.eval()

    def test_output_shape(self):
        """输出维度必须为 [B, 32]。"""
        x = torch.randn(B, N_GNSS)
        out = self.mixer(x)
        assert out.shape == (B, HIDDEN_DIM), (
            f"ChannelMixer output shape: expected ({B}, {HIDDEN_DIM}), got {out.shape}"
        )

    def test_output_nonnegative_after_relu(self):
        """经过 ReLU 后输出应 ≥ 0。"""
        x = torch.randn(B, N_GNSS)
        out = self.mixer(x)
        assert (out >= 0).all(), "ChannelMixer output contains negative values after ReLU"

    def test_single_sample(self):
        """单样本输入也能正常工作。"""
        x = torch.randn(1, N_GNSS)
        out = self.mixer(x)
        assert out.shape == (1, HIDDEN_DIM)


# ===========================================================================
# TemporalSEBlock 测试
# ===========================================================================
class TestTemporalSEBlock:
    """TemporalSEBlock: [B, T, C] → [B, T, C]（形状不变，权重在时间轴）"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.se = TemporalSEBlock(n_channels=HIDDEN_DIM, reduction=4)
        self.se.eval()

    def test_output_shape(self):
        """输出形状与输入相同。"""
        x = torch.randn(B, T, HIDDEN_DIM)
        out = self.se(x)
        assert out.shape == (B, T, HIDDEN_DIM), (
            f"SE output shape: expected ({B}, {T}, {HIDDEN_DIM}), got {out.shape}"
        )

    def test_attention_weights_range(self):
        """注意力权重应在 [0, 1] 范围内（Sigmoid 输出）。"""
        x = torch.randn(B, T, HIDDEN_DIM)
        # 前向传播获取权重
        weights = self.se.get_attention_weights(x)
        assert (weights >= 0).all() and (weights <= 1).all(), (
            "SE attention weights out of [0, 1] range"
        )

    def test_different_time_lengths(self):
        """不同时间步长度都应该能处理。"""
        for t in [1, 50, 273]:
            x = torch.randn(B, t, HIDDEN_DIM)
            out = self.se(x)
            assert out.shape == (B, t, HIDDEN_DIM)


# ===========================================================================
# ConvGRUCell 测试
# ===========================================================================
class TestConvGRUCell:
    """ConvGRUCell: 单步更新，空间维度 [15, 202] 不变。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.cell = ConvGRUCell(
            input_channels=2,
            hidden_dim=CONV_HIDDEN,
            kernel_size=3,
            padding=1,
        )
        self.cell.eval()

    def test_output_shape(self):
        """隐状态 H_t 维度: [B, 64, 15, 202]。"""
        x = torch.randn(B, 2, H, W)
        h_prev = torch.zeros(B, CONV_HIDDEN, H, W)
        h_next = self.cell(x, h_prev)
        assert h_next.shape == (B, CONV_HIDDEN, H, W), (
            f"ConvGRUCell output: expected ({B}, {CONV_HIDDEN}, {H}, {W}), "
            f"got {h_next.shape}"
        )

    def test_spatial_dims_preserved(self):
        """空间维度 (15, 202) 必须在前向传播后保持不变。"""
        x = torch.randn(B, 2, H, W)
        h_prev = torch.zeros(B, CONV_HIDDEN, H, W)
        h_next = self.cell(x, h_prev)
        assert h_next.shape[2] == H and h_next.shape[3] == W

    def test_none_hidden_state(self):
        """首次调用时 h_prev=None 应自动初始化为全零。"""
        x = torch.randn(B, 2, H, W)
        h_next = self.cell(x, None)
        assert h_next.shape == (B, CONV_HIDDEN, H, W)

    def test_gradient_flow(self):
        """梯度必须能正常回传。"""
        x = torch.randn(B, 2, H, W, requires_grad=True)
        h_prev = torch.zeros(B, CONV_HIDDEN, H, W)
        h_next = self.cell(x, h_prev)
        loss = h_next.sum()
        loss.backward()
        assert x.grad is not None, "Gradient did not flow through ConvGRUCell"


# ===========================================================================
# LearnedPhysicsLayer 测试
# ===========================================================================
class TestLearnedPhysicsLayer:
    """LearnedPhysicsLayer: nn.Linear(3030, 9, bias=False)"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.layer = LearnedPhysicsLayer(
            in_features=N_SLIP,
            out_features=N_GNSS,
        )

    def test_weight_shape(self):
        """权重矩阵形状必须为 [9, 3030]。"""
        w = self.layer.greens_function.weight
        assert w.shape == (N_GNSS, N_SLIP), (
            f"Weight shape: expected ({N_GNSS}, {N_SLIP}), got {w.shape}"
        )

    def test_no_bias(self):
        """必须无偏置项 (bias=False)。"""
        assert self.layer.greens_function.bias is None, (
            "LearnedPhysicsLayer must have bias=False"
        )

    def test_output_shape(self):
        """输入 [B, 3030] → 输出 [B, 9]。"""
        x = torch.randn(B, N_SLIP)
        out = self.layer(x)
        assert out.shape == (B, N_GNSS), (
            f"Output shape: expected ({B}, {N_GNSS}), got {out.shape}"
        )

    def test_output_shape_with_time(self):
        """输入 [B, T, 3030] → 输出 [B, T, 9]（支持时间维度）。"""
        x = torch.randn(B, T, N_SLIP)
        out = self.layer(x)
        assert out.shape == (B, T, N_GNSS), (
            f"Output shape: expected ({B}, {T}, {N_GNSS}), got {out.shape}"
        )

    def test_l1_regularization(self):
        """L1 正则化方法应返回标量 > 0。"""
        reg = self.layer.l1_regularization()
        assert reg.dim() == 0, "L1 regularization should return a scalar"
        assert reg.item() >= 0, "L1 regularization should be non-negative"

    def test_linear_mapping(self):
        """
        验证线性映射正确性：
        手动计算 W @ x 应等于 layer(x)。
        """
        x = torch.randn(1, N_SLIP)
        with torch.no_grad():
            out_layer = self.layer(x)
            out_manual = x @ self.layer.greens_function.weight.T
        assert torch.allclose(out_layer, out_manual, atol=1e-6), (
            "LearnedPhysicsLayer forward != manual W @ x"
        )
