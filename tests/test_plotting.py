# -*- coding: utf-8 -*-
"""
TDD 单元测试 — 绘图脚本鲁棒性测试
====================================
使用 Dummy 模拟数据验证 plot_paper.py 中所有 5 个绘图函数
在各种输入条件下不会崩溃，且正确生成 PNG + PDF 文件。
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.plot_paper import (  # noqa: E402
    plot_data_characteristics,
    plot_training_dynamics,
    plot_inversion_results,
    plot_forecasting_results,
    plot_ablation_and_moment,
    _to_numpy,
)


# ---------------------------------------------------------------------------
# Fixture: 临时输出目录
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def output_dir():
    tmpdir = tempfile.mkdtemp(prefix="plot_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Fixture: 模拟数据
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dummy_data():
    rng = np.random.RandomState(42)
    T, H, W = 50, 15, 202
    gnss = rng.randn(T, 9) * 0.001
    slip = rng.rand(T, H, W) * 0.0001
    # 注入活跃区域
    slip[20:30, 5:10, 80:120] = rng.rand(10, 5, 40) * 0.01
    return {
        "gnss": gnss,
        "slip": slip,
        "T": T, "H": H, "W": W,
    }


# ===========================================================================
# 测试用例
# ===========================================================================
class TestPlotDataCharacteristics:
    """图 A: 数据特征与网格拼接可视化。"""

    def test_generates_files(self, dummy_data, output_dir):
        """必须生成 PNG 和 PDF。"""
        path = plot_data_characteristics(
            gnss_data=dummy_data["gnss"],
            slip_data=dummy_data["slip"],
            output_dir=output_dir,
        )
        assert os.path.exists(path), f"PNG not created: {path}"
        assert os.path.exists(path.replace(".png", ".pdf")), "PDF not created"

    def test_file_nonzero_size(self, dummy_data, output_dir):
        """生成的文件大小 > 0。"""
        path = plot_data_characteristics(
            gnss_data=dummy_data["gnss"],
            slip_data=dummy_data["slip"],
            output_dir=output_dir,
        )
        assert os.path.getsize(path) > 1000, "PNG file too small"


class TestPlotTrainingDynamics:
    """图 B: 训练 Loss 曲线与 W 矩阵。"""

    def test_without_csv(self, output_dir):
        """即使 CSV 不存在也不崩溃。"""
        W = np.random.randn(9, 3030) * 0.01
        path = plot_training_dynamics(
            log_csv_path="nonexistent.csv",
            physics_weight_matrix=W,
            output_dir=output_dir,
        )
        assert os.path.exists(path)

    def test_without_weight(self, output_dir):
        """即使 W 矩阵为 None 也不崩溃。"""
        path = plot_training_dynamics(
            log_csv_path="nonexistent.csv",
            physics_weight_matrix=None,
            output_dir=output_dir,
        )
        assert os.path.exists(path)


class TestPlotInversionResults:
    """图 C: 反演结果空间对比。"""

    def test_generates_files(self, dummy_data, output_dir):
        """正常输入生成文件。"""
        T, H, W = dummy_data["T"], dummy_data["H"], dummy_data["W"]
        true_s = dummy_data["slip"]
        pred_s = true_s + np.random.randn(T, H, W) * 0.001
        path = plot_inversion_results(
            true_slip=true_s,
            pred_slip=pred_s,
            output_dir=output_dir,
        )
        assert os.path.exists(path)

    def test_with_gnss(self, dummy_data, output_dir):
        """带 GNSS 重构时也能正常工作。"""
        T = dummy_data["T"]
        true_s = dummy_data["slip"]
        pred_s = true_s * 1.05
        true_gnss = dummy_data["gnss"]
        pred_gnss = true_gnss * 0.95
        path = plot_inversion_results(
            true_slip=true_s,
            pred_slip=pred_s,
            true_gnss=true_gnss,
            pred_gnss=pred_gnss,
            output_dir=output_dir,
        )
        assert os.path.exists(path)


class TestPlotForecastingResults:
    """图 D: 预测演化与误差曲线。"""

    def test_generates_files(self, dummy_data, output_dir):
        """正常预测数据生成文件。"""
        H, W = dummy_data["H"], dummy_data["W"]
        n_steps = 20
        true_f = np.random.rand(n_steps, H, W) * 0.005
        pred_f = true_f + np.random.randn(n_steps, H, W) * 0.001
        path = plot_forecasting_results(
            history_slip=dummy_data["slip"],
            true_future=true_f,
            pred_future=pred_f,
            output_dir=output_dir,
        )
        assert os.path.exists(path)

    def test_short_forecast(self, dummy_data, output_dir):
        """仅 2 步预测也不崩溃。"""
        H, W = dummy_data["H"], dummy_data["W"]
        true_f = np.random.rand(2, H, W) * 0.005
        pred_f = true_f * 1.1
        path = plot_forecasting_results(
            history_slip=None,
            true_future=true_f,
            pred_future=pred_f,
            output_dir=output_dir,
        )
        assert os.path.exists(path)


class TestPlotAblationAndMoment:
    """图 E: M₀ 散点图与消融实验。"""

    def test_with_slip_data(self, dummy_data, output_dir):
        """带滑移数据时生成 M₀ 散点图。"""
        rng = np.random.RandomState(0)
        true_s = rng.rand(8, 20, 15, 202) * 0.01
        pred_s = true_s * 1.1
        path = plot_ablation_and_moment(
            true_slip=true_s,
            pred_slip=pred_s,
            output_dir=output_dir,
        )
        assert os.path.exists(path)

    def test_with_ablation_data(self, output_dir):
        """带消融结果时生成柱状图。"""
        ablation = {
            "Data-only": {"inversion_R2": 0.65, "forecasting_R2": 0.40},
            "No Grid Stitch": {"inversion_R2": 0.72, "forecasting_R2": 0.48},
            "No Pseudo-Okada": {"inversion_R2": 0.78, "forecasting_R2": 0.55},
            "Ours (Full)": {"inversion_R2": 0.92, "forecasting_R2": 0.85},
        }
        path = plot_ablation_and_moment(
            ablation_results=ablation,
            output_dir=output_dir,
        )
        assert os.path.exists(path)

    def test_template_mode(self, output_dir):
        """无任何数据时使用模板占位（不崩溃）。"""
        path = plot_ablation_and_moment(output_dir=output_dir)
        assert os.path.exists(path)


class TestToNumpy:
    """_to_numpy 健壮性测试。"""

    def test_tensor_input(self):
        import torch
        t = torch.tensor([1.0, 2.0, 3.0])
        result = _to_numpy(t)
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_numpy_input(self):
        arr = np.array([4.0, 5.0])
        result = _to_numpy(arr)
        assert np.allclose(result, [4.0, 5.0])

    def test_none_input(self):
        result = _to_numpy(None)
        assert result.size == 0

    def test_list_input(self):
        result = _to_numpy([1, 2, 3])
        assert np.allclose(result, [1, 2, 3])

    def test_cuda_tensor_no_gpu(self):
        """CPU tensor 也能正常转换。"""
        import torch
        t = torch.randn(3, 4, requires_grad=True)
        result = _to_numpy(t)
        assert result.shape == (3, 4)
