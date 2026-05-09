# -*- coding: utf-8 -*-
"""
TDD 单元测试 — SSEDataset 数据流拓扑测试
=============================================
使用模拟数据验证 SSEDataset 的核心物理拼接逻辑和输出维度。

核心断言:
  - Y_slip 形状: [Batch, Time, 15, 202]
  - X_gnss 形状: [Batch, Time, 9]
  - 无 NaN 值
  - 段 1/段 2 拼接值正确
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# 将项目根目录加入 sys.path，确保可以 import src
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.data_engine import SSEDataset  # noqa: E402

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
T = 273           # 时间步数
RAW_COLS = 3040   # 每行总列数
N_DEPTH = 15
N_WIDTH_SEG1 = 166
N_WIDTH_SEG2 = 36
N_WIDTH_TOTAL = 202   # 166 + 36
N_GNSS = 9
N_SLIP = 3030


# ---------------------------------------------------------------------------
# Fixtures：创建临时数据目录与假事件文件
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fake_data_dir():
    """
    创建一个临时目录，写入若干模拟事件 .txt 文件。
    每个文件: 273 行 × 3040 列 (float)。
    """
    tmpdir = tempfile.mkdtemp(prefix="sse_test_")
    n_fake_events = 4  # 生成 4 个假事件用于批次测试

    rng = np.random.RandomState(42)
    for i in range(n_fake_events):
        # 构造 [T, RAW_COLS] 数据
        data = rng.rand(T, RAW_COLS).astype(np.float64)
        filepath = os.path.join(tmpdir, f"event_{i:04d}.txt")
        np.savetxt(filepath, data, fmt="%.6f")

    yield tmpdir
    # 清理
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def dataset(fake_data_dir):
    """创建 SSEDataset 实例。"""
    config = {
        "data": {
            "slip_col_start": 1,
            "slip_col_end": 3031,
            "gnss_col_start": 3031,
            "gnss_col_end": 3040,
        },
        "grid": {
            "n_depth": N_DEPTH,
            "n_width_seg1": N_WIDTH_SEG1,
            "n_width_seg2": N_WIDTH_SEG2,
            "n_width_total": N_WIDTH_TOTAL,
            "seg1_slip_start": 0,
            "seg1_slip_end": 2490,
            "seg2_slip_start": 2490,
            "seg2_slip_end": 3030,
        },
    }
    return SSEDataset(data_dir=fake_data_dir, config=config)


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------
class TestSSEDatasetShape:
    """维度与形状断言测试组。"""

    def test_y_slip_shape(self, dataset):
        """Y_slip 张量形状必须为 [Time, 15, 202]。"""
        x_gnss, y_slip = dataset[0]
        assert y_slip.shape == (T, N_DEPTH, N_WIDTH_TOTAL), (
            f"Y_slip shape mismatch: expected ({T}, {N_DEPTH}, {N_WIDTH_TOTAL}), "
            f"got {y_slip.shape}"
        )

    def test_x_gnss_shape(self, dataset):
        """X_gnss 张量形状必须为 [Time, 9]。"""
        x_gnss, y_slip = dataset[0]
        assert x_gnss.shape == (T, N_GNSS), (
            f"X_gnss shape mismatch: expected ({T}, {N_GNSS}), got {x_gnss.shape}"
        )

    def test_batch_collation(self, dataset):
        """
        DataLoader batch_size=4 时：
          - X_gnss: [4, 273, 9]
          - Y_slip: [4, 273, 15, 202]
        """
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch_x, batch_y = next(iter(loader))

        assert batch_x.shape == (4, T, N_GNSS), (
            f"Batch X_gnss shape mismatch: expected (4, {T}, {N_GNSS}), "
            f"got {batch_x.shape}"
        )
        assert batch_y.shape == (4, T, N_DEPTH, N_WIDTH_TOTAL), (
            f"Batch Y_slip shape mismatch: expected (4, {T}, {N_DEPTH}, {N_WIDTH_TOTAL}), "
            f"got {batch_y.shape}"
        )


class TestSSEDatasetIntegrity:
    """数据完整性与拼接正确性测试组。"""

    def test_no_nan(self, dataset):
        """输出中不得包含任何 NaN 值。"""
        x_gnss, y_slip = dataset[0]
        assert not torch.isnan(x_gnss).any(), "X_gnss contains NaN values!"
        assert not torch.isnan(y_slip).any(), "Y_slip contains NaN values!"

    def test_stitching_values(self, fake_data_dir):
        """
        验证物理拼接正确性：
        给定已知输入值，检查 Y_slip 中 seg1 和 seg2 区域的值是否正确放置。
        """
        # 构造一个确定性事件：seg1 全为 1.0，seg2 全为 2.0，GNSS 全为 3.0
        data = np.zeros((T, RAW_COLS), dtype=np.float64)
        data[:, 0] = 2024.0                    # Year（会被舍弃）
        data[:, 1:2491] = 1.0                   # seg1: cols 1..2490 → 2490 values
        data[:, 2491:3031] = 2.0                # seg2: cols 2491..3030 → 540 values
        data[:, 3031:3040] = 3.0                # GNSS

        # 写入临时文件
        filepath = os.path.join(fake_data_dir, "event_deterministic.txt")
        np.savetxt(filepath, data, fmt="%.6f")

        config = {
            "data": {
                "slip_col_start": 1,
                "slip_col_end": 3031,
                "gnss_col_start": 3031,
                "gnss_col_end": 3040,
            },
            "grid": {
                "n_depth": N_DEPTH,
                "n_width_seg1": N_WIDTH_SEG1,
                "n_width_seg2": N_WIDTH_SEG2,
                "n_width_total": N_WIDTH_TOTAL,
                "seg1_slip_start": 0,
                "seg1_slip_end": 2490,
                "seg2_slip_start": 2490,
                "seg2_slip_end": 3030,
            },
        }
        ds = SSEDataset(data_dir=fake_data_dir, config=config, normalize=False)

        # 找到确定性事件的索引（最后一个文件）
        det_idx = None
        for idx, fpath in enumerate(ds.file_list):
            if "event_deterministic" in fpath:
                det_idx = idx
                break
        assert det_idx is not None, "Deterministic event file not found in dataset"

        x_gnss, y_slip = ds[det_idx]

        # 检查 seg1 区域（宽度 0:166）全为 1.0
        seg1_region = y_slip[:, :, :N_WIDTH_SEG1]   # [T, 15, 166]
        assert torch.allclose(seg1_region, torch.tensor(1.0, dtype=torch.float32)), (
            f"Segment 1 values incorrect: expected all 1.0, "
            f"got min={seg1_region.min():.4f}, max={seg1_region.max():.4f}"
        )

        # 检查 seg2 区域（宽度 166:202）全为 2.0
        seg2_region = y_slip[:, :, N_WIDTH_SEG1:]   # [T, 15, 36]
        assert torch.allclose(seg2_region, torch.tensor(2.0, dtype=torch.float32)), (
            f"Segment 2 values incorrect: expected all 2.0, "
            f"got min={seg2_region.min():.4f}, max={seg2_region.max():.4f}"
        )

        # 检查 GNSS 全为 3.0
        assert torch.allclose(x_gnss, torch.tensor(3.0, dtype=torch.float32)), (
            f"GNSS values incorrect: expected all 3.0, "
            f"got min={x_gnss.min():.4f}, max={x_gnss.max():.4f}"
        )

    def test_dtype_float32(self, dataset):
        """输出张量必须为 float32。"""
        x_gnss, y_slip = dataset[0]
        assert x_gnss.dtype == torch.float32, f"X_gnss dtype: {x_gnss.dtype}"
        assert y_slip.dtype == torch.float32, f"Y_slip dtype: {y_slip.dtype}"

    def test_dataset_length(self, dataset):
        """数据集长度应等于事件文件数。"""
        assert len(dataset) >= 4, f"Dataset length too small: {len(dataset)}"
