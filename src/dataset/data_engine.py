# -*- coding: utf-8 -*-
"""
SSEDataset — 慢滑移事件数据加载与物理拓扑重构引擎
====================================================
核心职责:
  1. 扫描事件 .txt 文件
  2. 按列切片提取 Slip_Vector 和 GNSS
  3. 将一维 Slip_Vector (3030-D) 重构为二维断层网格 [15, 202]
     - 段 1: slip[0:2490] → [15, 166]
     - 段 2: slip[2490:3030] → [15, 36]
     - torch.cat → [15, 202]
  4. 可选 z-score 归一化 (用预计算的 normalization_stats.json)
  5. 返回 (X_gnss, Y_slip) 作为 float32 张量
"""

import os
import json
import glob
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SSEDataset(Dataset):
    """
    慢滑移事件 (SSE) 数据集。

    每个样本是一个事件 .txt 文件，包含 T 个时间步、每步 3040 列的原始观测。
    本类负责：
      - Col 0 (Year): 舍弃
      - Col 1:3031 (Slip_Vector, 3030-D): 物理拼接为 [T, 15, 202] → Y_slip
      - Col 3031:3040 (GNSS, 9-D): 直接提取为 [T, 9] → X_gnss

    Parameters
    ----------
    data_dir : str
        事件 .txt 文件所在目录（支持递归搜索子目录）。
    config : dict
        配置字典，需包含 'data' 和 'grid' 两个子字典。
    transform : callable, optional
        可选的额外数据变换函数。
    """

    def __init__(
        self,
        data_dir: str,
        config: Dict[str, Any],
        transform: Optional[Any] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        self.data_dir = data_dir
        self.config = config
        self.transform = transform
        self.normalize = normalize

        # ----- 解析配置 -----
        data_cfg = config["data"]
        grid_cfg = config["grid"]

        # 列索引
        self.slip_col_start: int = data_cfg["slip_col_start"]
        self.slip_col_end: int = data_cfg["slip_col_end"]
        self.gnss_col_start: int = data_cfg["gnss_col_start"]
        self.gnss_col_end: int = data_cfg["gnss_col_end"]

        # 网格拓扑
        self.n_depth: int = grid_cfg["n_depth"]
        self.n_width_seg1: int = grid_cfg["n_width_seg1"]
        self.n_width_seg2: int = grid_cfg["n_width_seg2"]
        self.n_width_total: int = grid_cfg["n_width_total"]
        self.seg1_start: int = grid_cfg["seg1_slip_start"]
        self.seg1_end: int = grid_cfg["seg1_slip_end"]
        self.seg2_start: int = grid_cfg["seg2_slip_start"]
        self.seg2_end: int = grid_cfg["seg2_slip_end"]

        # ----- 验证网格参数一致性 -----
        assert self.n_width_seg1 + self.n_width_seg2 == self.n_width_total, (
            f"网格宽度不一致: seg1({self.n_width_seg1}) + seg2({self.n_width_seg2}) "
            f"!= total({self.n_width_total})"
        )
        assert (self.seg1_end - self.seg1_start) == self.n_depth * self.n_width_seg1, (
            f"段 1 元素数不匹配: {self.seg1_end - self.seg1_start} "
            f"!= {self.n_depth} * {self.n_width_seg1}"
        )
        assert (self.seg2_end - self.seg2_start) == self.n_depth * self.n_width_seg2, (
            f"段 2 元素数不匹配: {self.seg2_end - self.seg2_start} "
            f"!= {self.n_depth} * {self.n_width_seg2}"
        )

        # ----- 加载归一化统计量 -----
        self.gnss_mean = None
        self.gnss_std = None
        self.slip_mean = None
        self.slip_std = None

        if self.normalize:
            stats_path = os.path.join(data_dir, "normalization_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                self.gnss_mean = np.array(stats["gnss_mean"], dtype=np.float32)  # [9]
                self.gnss_std = np.array(stats["gnss_std"], dtype=np.float32)    # [9]
                self.slip_mean = np.array(stats["slip_mean"], dtype=np.float32)  # [3030]
                self.slip_std = np.array(stats["slip_std"], dtype=np.float32)    # [3030]
                # 防止除零
                self.gnss_std = np.maximum(self.gnss_std, 1e-8)
                self.slip_std = np.maximum(self.slip_std, 1e-8)

        # ----- 扫描事件文件 -----
        self.file_list: List[str] = self._scan_event_files(data_dir)
        if len(self.file_list) == 0:
            raise RuntimeError(f"在 {data_dir} 中未找到任何 .txt 事件文件")

    def _scan_event_files(self, data_dir: str) -> List[str]:
        """
        递归扫描目录中所有 .txt 文件并按文件名排序。

        Parameters
        ----------
        data_dir : str
            根数据目录。

        Returns
        -------
        List[str]
            排序后的文件绝对路径列表。
        """
        pattern = os.path.join(data_dir, "**", "*.txt")
        files = sorted(glob.glob(pattern, recursive=True))
        return files

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        加载第 idx 个事件并执行物理拓扑重构。

        Returns
        -------
        x_gnss : torch.Tensor, shape [T, 9]
            GNSS 位移观测。
        y_slip : torch.Tensor, shape [T, 15, 202]
            断层滑移网格（段 1 + 段 2 拼接）。
        """
        filepath = self.file_list[idx]

        # 1. 加载原始数据 → [T, 3040]
        try:
            raw = np.loadtxt(filepath, dtype=np.float64)
        except Exception as e:
            raise IOError(f"无法加载事件文件 {filepath}: {e}")

        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        # 2. 基本维度校验
        n_cols = raw.shape[1]
        expected_min_cols = max(self.slip_col_end, self.gnss_col_end)
        if n_cols < expected_min_cols:
            raise ValueError(
                f"事件文件 {filepath} 列数不足: "
                f"实际 {n_cols}, 需要至少 {expected_min_cols}"
            )

        # 3. 提取 GNSS → [T, 9]
        gnss_np = raw[:, self.gnss_col_start : self.gnss_col_end].copy()

        # 4. 提取 Slip_Vector → [T, 3030]
        slip_vector = raw[:, self.slip_col_start : self.slip_col_end].copy()

        # 5. 归一化处理 (Normalization)
        if self.normalize:
            # 5.1 对目标滑移场保持全局 Z-score，以防止丢失绝对物理幅度
            if self.slip_mean is not None:
                slip_vector = (slip_vector - self.slip_mean) / self.slip_std  # [T, 3030]

            # 5.2 对 GNSS 输入引入单序列 RobustScaler (TimeSFM 思想)，抗击高频离群噪声
            q75, q25 = np.percentile(gnss_np, [75, 25], axis=0)
            iqr = q75 - q25
            iqr[iqr < 1e-8] = 1.0
            median = np.median(gnss_np, axis=0)
            gnss_np = (gnss_np - median) / iqr

        T = raw.shape[0]

        # 6. 物理拓扑重构 — 严格按文档执行
        # 段 1: slip_vector[:, 0:2490] → reshape [T, 15, 166]
        seg1 = slip_vector[:, self.seg1_start : self.seg1_end].reshape(
            T, self.n_depth, self.n_width_seg1
        )

        # 段 2: slip_vector[:, 2490:3030] → reshape [T, 15, 36]
        seg2 = slip_vector[:, self.seg2_start : self.seg2_end].reshape(
            T, self.n_depth, self.n_width_seg2
        )

        # 7. 在 width 维度上拼接 → [T, 15, 202]
        seg1_tensor = torch.from_numpy(seg1).float()
        seg2_tensor = torch.from_numpy(seg2).float()
        y_slip = torch.cat([seg1_tensor, seg2_tensor], dim=2)  # dim=2 ↔ width

        # 8. 转换 GNSS → [T, 9]
        x_gnss = torch.from_numpy(gnss_np).float()

        # 9. NaN 安全检查
        if torch.isnan(x_gnss).any() or torch.isnan(y_slip).any():
            raise ValueError(f"事件文件 {filepath} 存在 NaN 值")

        # 10. 可选变换
        if self.transform is not None:
            x_gnss, y_slip = self.transform(x_gnss, y_slip)

        return x_gnss, y_slip

    def __repr__(self) -> str:
        return (
            f"SSEDataset(\n"
            f"  data_dir='{self.data_dir}',\n"
            f"  n_events={len(self)},\n"
            f"  grid=[{self.n_depth}, {self.n_width_total}],\n"
            f"  seg1=[{self.n_depth}, {self.n_width_seg1}], "
            f"seg2=[{self.n_depth}, {self.n_width_seg2}]\n"
            f")"
        )
