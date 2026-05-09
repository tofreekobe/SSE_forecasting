# -*- coding: utf-8 -*-
"""
TDD 单元测试 — STCR_Net 主模型维度、因果性与物理约束测试
==========================================================
验证 GNSSEncoder, ConvGRUProcessor, PhysicalDecoder, STCR_Net
的输入输出维度和物理约束（Softplus 非负性）。
"""

import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.stcr_net import (  # noqa: E402
    GNSSEncoder,
    ConvGRUProcessor,
    PhysicalDecoder,
    STCR_Net,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
B = 2              # Batch size（省内存）
T = 10             # 时间步
N_GNSS = 9         # GNSS 维度
H, W = 15, 202     # 空间网格
HIDDEN_DIM = 64    # ConvGRU hidden
ENCODER_HIDDEN = 32
N_SLIP = 3030
NUM_GRU_LAYERS = 3


# ===========================================================================
# GNSSEncoder 测试
# ===========================================================================
class TestGNSSEncoder:
    """GNSSEncoder: [B, T, 9] → [B, T, 1, 15, 202]"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.encoder = GNSSEncoder(
            n_gnss=N_GNSS,
            hidden_dim=ENCODER_HIDDEN,
            grid_h=H,
            grid_w=W,
        )
        self.encoder.eval()

    def test_output_shape(self):
        """编码器输出形状必须为 [B, T, 1, 15, 202]。"""
        x = torch.randn(B, T, N_GNSS)
        with torch.no_grad():
            out = self.encoder(x)
        assert out.shape == (B, T, 1, H, W), (
            f"GNSSEncoder output: expected ({B}, {T}, 1, {H}, {W}), got {out.shape}"
        )

    def test_single_timestep(self):
        """单时间步输入。"""
        x = torch.randn(B, 1, N_GNSS)
        with torch.no_grad():
            out = self.encoder(x)
        assert out.shape == (B, 1, 1, H, W)


# ===========================================================================
# ConvGRUProcessor 测试
# ===========================================================================
class TestConvGRUProcessor:
    """ConvGRU 3 层堆叠：隐状态维度 [B, 64, 15, 202]"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.processor = ConvGRUProcessor(
            input_channels=2,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_GRU_LAYERS,
            kernel_size=3,
            padding=1,
        )
        self.processor.eval()

    def test_output_and_hidden_shape(self):
        """输出隐状态形状: 每层 [B, 64, 15, 202]。"""
        x = torch.randn(B, 2, H, W)
        with torch.no_grad():
            h_out, hidden_states = self.processor(x, None)
        assert h_out.shape == (B, HIDDEN_DIM, H, W)
        assert len(hidden_states) == NUM_GRU_LAYERS
        for i, h in enumerate(hidden_states):
            assert h.shape == (B, HIDDEN_DIM, H, W), (
                f"Layer {i} hidden: expected ({B}, {HIDDEN_DIM}, {H}, {W}), got {h.shape}"
            )

    def test_hidden_state_continuity(self):
        """隐状态可以跨步传递。"""
        x1 = torch.randn(B, 2, H, W)
        x2 = torch.randn(B, 2, H, W)
        with torch.no_grad():
            _, h1 = self.processor(x1, None)
            h_out2, h2 = self.processor(x2, h1)
        assert h_out2.shape == (B, HIDDEN_DIM, H, W)


# ===========================================================================
# PhysicalDecoder 测试
# ===========================================================================
class TestPhysicalDecoder:
    """PhysicalDecoder: [B, 64, 15, 202] → [B, 1, 15, 202]，输出 ≥ 0"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.decoder = PhysicalDecoder(in_channels=HIDDEN_DIM, out_channels=1)
        self.decoder.eval()

    def test_output_shape(self):
        """解码器输出形状: [B, 1, 15, 202]。"""
        x = torch.randn(B, HIDDEN_DIM, H, W)
        with torch.no_grad():
            out = self.decoder(x)
        assert out.shape == (B, 1, H, W)

    def test_output_nonnegative(self):
        """所有输出必须 ≥ 0（Softplus 约束）。"""
        # 使用极端负值输入，确保输出依然非负
        x = torch.randn(B, HIDDEN_DIM, H, W) * 100
        with torch.no_grad():
            out = self.decoder(x)
        assert (out >= 0).all(), (
            f"PhysicalDecoder output contains negative values! "
            f"min={out.min().item():.6f}"
        )

    def test_output_nonnegative_large_negative_input(self):
        """再次用全负输入验证 Softplus 约束。"""
        x = torch.full((B, HIDDEN_DIM, H, W), -50.0)
        with torch.no_grad():
            out = self.decoder(x)
        assert (out >= 0).all()


# ===========================================================================
# STCR_Net 完整模型测试
# ===========================================================================
class TestSTCRNet:
    """端到端模型测试。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = STCR_Net(
            n_gnss=N_GNSS,
            encoder_hidden=ENCODER_HIDDEN,
            grid_h=H,
            grid_w=W,
            gru_hidden=HIDDEN_DIM,
            gru_layers=NUM_GRU_LAYERS,
            n_slip=N_SLIP,
        )
        self.model.eval()

    def test_forward_inversion_slip_shape(self):
        """forward_inversion 输出 Slip 形状: [B, T, 15, 202]。"""
        gnss = torch.randn(B, T, N_GNSS)
        with torch.no_grad():
            result = self.model.forward_inversion(gnss)
        slip = result["slip"]
        assert slip.shape == (B, T, H, W), (
            f"Inversion slip: expected ({B}, {T}, {H}, {W}), got {slip.shape}"
        )

    def test_forward_inversion_slip_nonnegative(self):
        """forward_inversion 输出的所有 Slip 值必须 ≥ 0。"""
        gnss = torch.randn(B, T, N_GNSS)
        with torch.no_grad():
            result = self.model.forward_inversion(gnss)
        slip = result["slip"]
        assert (slip >= 0).all(), (
            f"Inversion slip contains negative values! min={slip.min().item():.6f}"
        )

    def test_forward_inversion_returns_hidden(self):
        """forward_inversion 必须返回隐状态供预测任务使用。"""
        gnss = torch.randn(B, T, N_GNSS)
        with torch.no_grad():
            result = self.model.forward_inversion(gnss)
        assert "hidden_states" in result
        assert len(result["hidden_states"]) == NUM_GRU_LAYERS

    def test_forward_forecasting_shape(self):
        """forward_forecasting 输出形状: [B, steps, 15, 202]。"""
        gnss = torch.randn(B, T, N_GNSS)
        forecast_steps = 5
        with torch.no_grad():
            inv_result = self.model.forward_inversion(gnss)
            fore_result = self.model.forward_forecasting(
                hidden_states=inv_result["hidden_states"],
                last_slip=inv_result["last_slip"],
                steps=forecast_steps,
            )
        pred_slip = fore_result["slip"]
        assert pred_slip.shape == (B, forecast_steps, H, W), (
            f"Forecasting slip: expected ({B}, {forecast_steps}, {H}, {W}), "
            f"got {pred_slip.shape}"
        )

    def test_forward_forecasting_nonnegative(self):
        """forward_forecasting 输出 Slip 也必须 ≥ 0。"""
        gnss = torch.randn(B, T, N_GNSS)
        with torch.no_grad():
            inv_result = self.model.forward_inversion(gnss)
            fore_result = self.model.forward_forecasting(
                hidden_states=inv_result["hidden_states"],
                last_slip=inv_result["last_slip"],
                steps=3,
            )
        assert (fore_result["slip"] >= 0).all()

    def test_hidden_state_shared(self):
        """反演结束时的隐状态可以直接传入预测方法（共享）。"""
        gnss = torch.randn(B, T, N_GNSS)
        with torch.no_grad():
            inv_result = self.model.forward_inversion(gnss)
            h_inv = inv_result["hidden_states"]
            # 应该可以直接用于 forecasting，不报错
            fore_result = self.model.forward_forecasting(
                hidden_states=h_inv,
                last_slip=inv_result["last_slip"],
                steps=2,
            )
        # 验证 forecasting 结果有效
        assert fore_result["slip"].shape == (B, 2, H, W)

    def test_physics_layer_integration(self):
        """LearnedPhysicsLayer 集成在模型中，可计算重构 GNSS。"""
        gnss = torch.randn(B, T, N_GNSS)
        with torch.no_grad():
            result = self.model.forward_inversion(gnss)
        slip = result["slip"]  # [B, T, 15, 202]
        # 手动通过物理层计算
        flat_slip = slip.reshape(B * T, N_SLIP)
        with torch.no_grad():
            pred_gnss = self.model.physics_layer(flat_slip)
        assert pred_gnss.shape == (B * T, N_GNSS)

    def test_gradient_flow_end_to_end(self):
        """梯度必须能端到端回传。"""
        self.model.train()
        gnss = torch.randn(B, T, N_GNSS, requires_grad=True)
        result = self.model.forward_inversion(gnss)
        loss = result["slip"].sum()
        loss.backward()
        assert gnss.grad is not None, "Gradient did not flow through STCR_Net"
