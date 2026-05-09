# -*- coding: utf-8 -*-
"""
ST-CRNet 主模型 — 时空卷积循环网络
====================================
整合 GNSSEncoder, ConvGRUProcessor, PhysicalDecoder, LearnedPhysicsLayer
实现双模态前向传播: forward_inversion (反演) 和 forward_forecasting (预测)。

设计遵循《技术需求文档 V3.0》第三部分。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any

from src.models.layers import (
    ChannelMixer,
    TemporalSEBlock,
    ConvGRUCell,
    LearnedPhysicsLayer,
)


# ===========================================================================
# 1. GNSSEncoder — 观测特征编码器
# ===========================================================================
class GNSSEncoder(nn.Module):
    """
    GNSS 观测编码器：将低维地表位移信号编码为高维地下特征网格。

    流水线:
      1. ChannelMixer: [B, T, 9] → [B, T, 32]  (台站差分, 滤除 CME)
      2. TemporalSEBlock: [B, T, 32] → [B, T, 32]  (时间注意力)
      3. SpatialProjection: [B, T, 32] → [B, T, 15*202] → [B, T, 1, 15, 202]

    Parameters
    ----------
    n_gnss : int
        GNSS 输入维度（默认 9）。
    hidden_dim : int
        通道混合器隐藏维度（默认 32）。
    grid_h : int
        网格高度（默认 15）。
    grid_w : int
        网格宽度（默认 202）。
    se_reduction : int
        SE 注意力缩减比例（默认 4）。
    """

    def __init__(
        self,
        n_gnss: int = 9,
        hidden_dim: int = 32,
        grid_h: int = 15,
        grid_w: int = 202,
        se_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

        # 处理 1: Channel-Mixer (过滤 CME)
        self.channel_mixer = ChannelMixer(in_features=n_gnss, out_features=hidden_dim)

        # 处理 2: Temporal SE-Block (时间注意力)
        self.temporal_se = TemporalSEBlock(n_channels=hidden_dim, reduction=se_reduction)

        # 处理 3: Spatial Projection (投影到网格空间)
        self.spatial_proj = nn.Linear(hidden_dim, grid_h * grid_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [B, T, 9]
            GNSS 时间序列。

        Returns
        -------
        Tensor, shape [B, T, 1, 15, 202]
            编码后的空间特征网格。
        """
        B, T, _ = x.shape

        # Step 1: Channel-Mixer 逐时间步处理
        # 展平 [B*T, 9] → ChannelMixer → [B*T, 32] → 恢复 [B, T, 32]
        x_flat = x.reshape(B * T, -1)                       # [B*T, 9]
        mixed = self.channel_mixer(x_flat)                   # [B*T, 32]
        mixed = mixed.reshape(B, T, -1)                      # [B, T, 32]

        # Step 2: Temporal SE-Block (需要完整时间轴)
        attended = self.temporal_se(mixed)                    # [B, T, 32]

        # Step 3: Spatial Projection
        projected = self.spatial_proj(attended)               # [B, T, H*W]
        encoded = projected.reshape(B, T, 1, self.grid_h, self.grid_w)  # [B, T, 1, H, W]

        return encoded


# ===========================================================================
# 2. ConvGRUProcessor — 多层堆叠 ConvGRU 时空演化引擎
# ===========================================================================
class ConvGRUProcessor(nn.Module):
    """
    多层堆叠 ConvGRU 处理器。

    利用卷积核的空间局部归纳偏置，在小数据集上（6000 样本）比 Transformer
    更不容易过拟合，且天然适合模拟应力在断层面的局部连续扩散过程。

    Parameters
    ----------
    input_channels : int
        首层输入通道数（默认 2: Encoded_GNSS + Prev_Slip）。
    hidden_dim : int
        隐状态通道数（默认 64）。
    num_layers : int
        堆叠层数（默认 3）。
    kernel_size : int
        卷积核大小（默认 3）。
    padding : int
        卷积填充（默认 1，保持空间维度）。
    """

    def __init__(
        self,
        input_channels: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # 构建多层 ConvGRU
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_dim
            self.cells.append(
                ConvGRUCell(
                    input_channels=in_ch,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None,
    ) -> tuple:
        """
        单步前向传播（通过所有层）。

        Parameters
        ----------
        x : Tensor, shape [B, input_channels, H, W]
            当前时刻输入。
        hidden_states : list of Tensor or None
            每层的上一时刻隐状态。None 时自动初始化。

        Returns
        -------
        h_out : Tensor, shape [B, hidden_dim, H, W]
            最后一层的隐状态输出。
        new_hidden : list of Tensor
            更新后的各层隐状态。
        """
        if hidden_states is None:
            hidden_states = [None] * self.num_layers

        new_hidden = []
        current_input = x

        for i, cell in enumerate(self.cells):
            h = cell(current_input, hidden_states[i])
            new_hidden.append(h)
            current_input = h  # 下一层的输入是当前层的隐状态

        return current_input, new_hidden


# ===========================================================================
# 3. PhysicalDecoder — 滑移解码器 (Softplus 非负约束)
# ===========================================================================
class PhysicalDecoder(nn.Module):
    """
    物理滑移解码器。

    使用 1x1 卷积将 ConvGRU 隐状态映射为滑移分布，
    并强制使用 F.softplus 激活函数确保输出 ≥ 0。

    > 物理滑移量必须 ≥ 0，严禁使用无约束 Linear 或输出负值。

    Parameters
    ----------
    in_channels : int
        输入通道数（默认 64，即 ConvGRU hidden_dim）。
    out_channels : int
        输出通道数（默认 1）。
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [B, in_channels, H, W]

        Returns
        -------
        Tensor, shape [B, 1, H, W]
            非负滑移分布（经 Softplus 约束）。
        """
        out = self.conv(x)                    # [B, 1, H, W]
        out = F.softplus(out)                 # 强制非负
        return out


# ===========================================================================
# 4. STCR_Net — 主模型（双模态前向传播）
# ===========================================================================
class STCR_Net(nn.Module):
    """
    Spatio-Temporal Convolutional Recurrent Network (ST-CRNet)。

    整合四大组件：
      1. GNSSEncoder: 编码地表 GNSS → 地下特征网格
      2. ConvGRUProcessor: 多层 ConvGRU 时空演化
      3. PhysicalDecoder: 解码隐状态 → 非负滑移
      4. LearnedPhysicsLayer: 可学习格林函数（仅用于物理 Loss）

    提供两种前向传播模式:
      - forward_inversion: 已知 GNSS 序列 → 反演滑移历史
      - forward_forecasting: 已知隐状态 → 自回归预测未来滑移

    Parameters
    ----------
    n_gnss : int
        GNSS 维度（默认 9）。
    encoder_hidden : int
        编码器隐藏维度（默认 32）。
    grid_h, grid_w : int
        空间网格尺寸（默认 15, 202）。
    gru_hidden : int
        ConvGRU 隐状态维度（默认 64）。
    gru_layers : int
        ConvGRU 层数（默认 3）。
    n_slip : int
        滑移向量展平维度（默认 3030）。
    """

    def __init__(
        self,
        n_gnss: int = 9,
        encoder_hidden: int = 32,
        grid_h: int = 15,
        grid_w: int = 202,
        gru_hidden: int = 64,
        gru_layers: int = 3,
        n_slip: int = 3030,
    ) -> None:
        super().__init__()

        self.grid_h = grid_h
        self.grid_w = grid_w
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers

        # 四大组件
        self.encoder = GNSSEncoder(
            n_gnss=n_gnss,
            hidden_dim=encoder_hidden,
            grid_h=grid_h,
            grid_w=grid_w,
        )

        self.conv_gru = ConvGRUProcessor(
            input_channels=2,  # Encoded_GNSS (1 ch) + Prev_Slip (1 ch)
            hidden_dim=gru_hidden,
            num_layers=gru_layers,
            kernel_size=3,
            padding=1,
        )

        self.decoder = PhysicalDecoder(
            in_channels=gru_hidden,
            out_channels=1,
        )

        self.physics_layer = LearnedPhysicsLayer(
            in_features=n_slip,
            out_features=n_gnss,
        )

    def forward_inversion(
        self,
        gnss_seq: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        反演模式：已知 GNSS 时间序列 → 逐步反演地下滑移分布。

        对每个时间步 t:
          1. 编码 GNSS_t → 特征网格 [B, 1, H, W]
          2. 拼接 [Encoded_GNSS, Prev_Slip] → [B, 2, H, W]
          3. ConvGRU 更新隐状态
          4. 解码 → 非负 Slip_t [B, 1, H, W]

        Parameters
        ----------
        gnss_seq : Tensor, shape [B, T, 9]
            GNSS 观测序列。
        hidden_states : list or None
            初始隐状态（跨批次延续用）。

        Returns
        -------
        dict with:
          - "slip": Tensor [B, T, H, W]  反演滑移序列
          - "hidden_states": list  最终隐状态（供 forecasting 使用）
          - "last_slip": Tensor [B, 1, H, W]  最后一步滑移
        """
        B, T, _ = gnss_seq.shape
        device = gnss_seq.device

        # 一次性编码整个 GNSS 序列（利用 SE 时间注意力）
        encoded_seq = self.encoder(gnss_seq)  # [B, T, 1, H, W]

        # 初始化
        if hidden_states is None:
            hidden_states = [None] * self.gru_layers
        prev_slip = torch.zeros(B, 1, self.grid_h, self.grid_w, device=device)

        # 逐时间步循环
        slip_outputs = []
        for t in range(T):
            encoded_t = encoded_seq[:, t, :, :, :]     # [B, 1, H, W]

            # 拼接 Encoded_GNSS + Prev_Slip → [B, 2, H, W]
            gru_input = torch.cat([encoded_t, prev_slip], dim=1)

            # ConvGRU 更新
            h_out, hidden_states = self.conv_gru(gru_input, hidden_states)

            # 解码 → 非负 Slip
            slip_t = self.decoder(h_out)               # [B, 1, H, W]
            slip_outputs.append(slip_t.squeeze(1))     # [B, H, W]

            # 更新 prev_slip（Teacher Forcing 时可替换为真实值）
            prev_slip = slip_t.detach() if not self.training else slip_t

        # 堆叠 → [B, T, H, W]
        slip_seq = torch.stack(slip_outputs, dim=1)

        return {
            "slip": slip_seq,
            "hidden_states": hidden_states,
            "last_slip": slip_outputs[-1].unsqueeze(1),  # [B, 1, H, W]
        }

    def forward_forecasting(
        self,
        hidden_states: List[torch.Tensor],
        last_slip: torch.Tensor,
        steps: int,
    ) -> Dict[str, Any]:
        """
        预测模式：仅依靠隐状态，自回归滚动预测未来滑移。

        不再有 GNSS 输入，用零向量代替编码特征。
        模型完全依赖 ConvGRU 隐状态中积累的时空动力学记忆。

        Parameters
        ----------
        hidden_states : list of Tensor
            反演阶段结束时的隐状态。
        last_slip : Tensor, shape [B, 1, H, W]
            反演阶段最后一步的滑移。
        steps : int
            预测步数。

        Returns
        -------
        dict with:
          - "slip": Tensor [B, steps, H, W]
          - "hidden_states": list  最终隐状态
        """
        B = last_slip.size(0)
        device = last_slip.device
        prev_slip = last_slip

        slip_outputs = []
        for _ in range(steps):
            # 无 GNSS → 零编码特征
            zero_encoded = torch.zeros(B, 1, self.grid_h, self.grid_w, device=device)

            # 拼接 [Zero_Encoded, Prev_Slip] → [B, 2, H, W]
            gru_input = torch.cat([zero_encoded, prev_slip], dim=1)

            # ConvGRU 更新
            h_out, hidden_states = self.conv_gru(gru_input, hidden_states)

            # 解码 → 非负 Slip
            slip_t = self.decoder(h_out)               # [B, 1, H, W]
            slip_outputs.append(slip_t.squeeze(1))     # [B, H, W]

            prev_slip = slip_t

        slip_seq = torch.stack(slip_outputs, dim=1)    # [B, steps, H, W]

        return {
            "slip": slip_seq,
            "hidden_states": hidden_states,
        }
