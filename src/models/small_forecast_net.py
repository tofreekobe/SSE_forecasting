# -*- coding: utf-8 -*-
"""Small forecasting-only network for overfit checks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset.forecast_contract import (
    GRID_DEPTH,
    GRID_WIDTH_SEG1,
    GRID_WIDTH_SEG2,
    GRID_WIDTH_TOTAL,
    SEG1_SIZE,
)


def slip_vector_to_grid_torch(slip: torch.Tensor) -> torch.Tensor:
    if slip.shape[-1] != 3030:
        raise ValueError(f"Expected last dimension 3030, got {tuple(slip.shape)}")
    seg1 = slip[..., :SEG1_SIZE].reshape(*slip.shape[:-1], GRID_DEPTH, GRID_WIDTH_SEG1)
    seg2 = slip[..., SEG1_SIZE:].reshape(*slip.shape[:-1], GRID_DEPTH, GRID_WIDTH_SEG2)
    return torch.cat([seg1, seg2], dim=-1)


def slip_grid_to_vector_torch(grid: torch.Tensor) -> torch.Tensor:
    if tuple(grid.shape[-2:]) != (GRID_DEPTH, GRID_WIDTH_TOTAL):
        raise ValueError(f"Expected trailing grid shape {(GRID_DEPTH, GRID_WIDTH_TOTAL)}, got {tuple(grid.shape[-2:])}")
    seg1 = grid[..., :, :GRID_WIDTH_SEG1].reshape(*grid.shape[:-2], SEG1_SIZE)
    seg2 = grid[..., :, GRID_WIDTH_SEG1:].reshape(*grid.shape[:-2], GRID_DEPTH * GRID_WIDTH_SEG2)
    return torch.cat([seg1, seg2], dim=-1)


def slip_vector_to_segments_torch(slip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if slip.shape[-1] != 3030:
        raise ValueError(f"Expected last dimension 3030, got {tuple(slip.shape)}")
    seg1 = slip[..., :SEG1_SIZE].reshape(*slip.shape[:-1], GRID_DEPTH, GRID_WIDTH_SEG1)
    seg2 = slip[..., SEG1_SIZE:].reshape(*slip.shape[:-1], GRID_DEPTH, GRID_WIDTH_SEG2)
    return seg1, seg2


def slip_segments_to_vector_torch(seg1: torch.Tensor, seg2: torch.Tensor) -> torch.Tensor:
    if tuple(seg1.shape[-2:]) != (GRID_DEPTH, GRID_WIDTH_SEG1):
        raise ValueError(f"Expected segment 1 shape {(GRID_DEPTH, GRID_WIDTH_SEG1)}, got {tuple(seg1.shape[-2:])}")
    if tuple(seg2.shape[-2:]) != (GRID_DEPTH, GRID_WIDTH_SEG2):
        raise ValueError(f"Expected segment 2 shape {(GRID_DEPTH, GRID_WIDTH_SEG2)}, got {tuple(seg2.shape[-2:])}")
    return torch.cat(
        [
            seg1.reshape(*seg1.shape[:-2], SEG1_SIZE),
            seg2.reshape(*seg2.shape[:-2], GRID_DEPTH * GRID_WIDTH_SEG2),
        ],
        dim=-1,
    )


def _make_conv_stack(
    in_channels: int,
    hidden_channels: int,
    forecast_horizon: int,
    num_blocks: int,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
        nn.GELU(),
    ]
    for _ in range(num_blocks):
        layers.extend(
            [
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.GELU(),
            ]
        )
    layers.append(nn.Conv2d(hidden_channels, forecast_horizon, kernel_size=1))
    return nn.Sequential(*layers)


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    """Small residual block for stable segment-wise slip forecasting."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


def _make_residual_conv_stack(
    in_channels: int,
    hidden_channels: int,
    forecast_horizon: int,
    num_blocks: int,
) -> nn.Sequential:
    groups = _group_count(hidden_channels)
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
        nn.GroupNorm(groups, hidden_channels),
        nn.GELU(),
    ]
    layers.extend(ResidualConvBlock(hidden_channels) for _ in range(num_blocks))
    layers.extend(
        [
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, forecast_horizon, kernel_size=1),
        ]
    )
    return nn.Sequential(*layers)


class SlipConvForecastNet(nn.Module):
    """A compact model used to verify the future-slip training loop."""

    def __init__(
        self,
        history_steps: int,
        forecast_horizon: int,
        n_gnss: int = 9,
        gnss_channels: int = 16,
        hidden_channels: int = 64,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.history_steps = int(history_steps)
        self.forecast_horizon = int(forecast_horizon)
        self.gnss_encoder = nn.Sequential(
            nn.Linear(self.history_steps * n_gnss, 64),
            nn.GELU(),
            nn.Linear(64, gnss_channels),
            nn.GELU(),
        )

        in_channels = self.history_steps + gnss_channels
        self.net = _make_conv_stack(in_channels, hidden_channels, self.forecast_horizon, num_blocks)

    def forward(self, history_slip: torch.Tensor, history_gnss: torch.Tensor) -> torch.Tensor:
        slip_grid = slip_vector_to_grid_torch(history_slip)
        bsz = slip_grid.shape[0]
        gnss_features = self.gnss_encoder(history_gnss.reshape(bsz, -1))
        gnss_grid = gnss_features[:, :, None, None].expand(-1, -1, GRID_DEPTH, GRID_WIDTH_TOTAL)
        x = torch.cat([slip_grid, gnss_grid], dim=1)
        delta_grid = self.net(x)
        last_grid = slip_grid[:, -1:].expand(-1, self.forecast_horizon, -1, -1)
        pred_grid = F.relu(last_grid + delta_grid)
        return slip_grid_to_vector_torch(pred_grid)


class SegmentedSlipConvForecastNet(nn.Module):
    """Forecast slip without creating artificial adjacency between disconnected fault segments."""

    def __init__(
        self,
        history_steps: int,
        forecast_horizon: int,
        n_gnss: int = 9,
        gnss_channels: int = 16,
        hidden_channels: int = 64,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.history_steps = int(history_steps)
        self.forecast_horizon = int(forecast_horizon)
        self.gnss_encoder = nn.Sequential(
            nn.Linear(self.history_steps * n_gnss, 64),
            nn.GELU(),
            nn.Linear(64, gnss_channels),
            nn.GELU(),
        )

        in_channels = self.history_steps + gnss_channels
        self.seg1_net = _make_conv_stack(in_channels, hidden_channels, self.forecast_horizon, num_blocks)
        self.seg2_net = _make_conv_stack(in_channels, hidden_channels, self.forecast_horizon, num_blocks)

    def _expand_gnss(self, history_gnss: torch.Tensor, width: int) -> torch.Tensor:
        bsz = history_gnss.shape[0]
        gnss_features = self.gnss_encoder(history_gnss.reshape(bsz, -1))
        return gnss_features[:, :, None, None].expand(-1, -1, GRID_DEPTH, width)

    def forward(self, history_slip: torch.Tensor, history_gnss: torch.Tensor) -> torch.Tensor:
        seg1, seg2 = slip_vector_to_segments_torch(history_slip)
        seg1_input = torch.cat([seg1, self._expand_gnss(history_gnss, GRID_WIDTH_SEG1)], dim=1)
        seg2_input = torch.cat([seg2, self._expand_gnss(history_gnss, GRID_WIDTH_SEG2)], dim=1)

        seg1_delta = self.seg1_net(seg1_input)
        seg2_delta = self.seg2_net(seg2_input)
        seg1_pred = F.relu(seg1[:, -1:].expand(-1, self.forecast_horizon, -1, -1) + seg1_delta)
        seg2_pred = F.relu(seg2[:, -1:].expand(-1, self.forecast_horizon, -1, -1) + seg2_delta)
        return slip_segments_to_vector_torch(seg1_pred, seg2_pred)


class SegmentedResidualForecastNet(nn.Module):
    """Segment-aware residual model used as the default full-training baseline."""

    def __init__(
        self,
        history_steps: int,
        forecast_horizon: int,
        n_gnss: int = 9,
        gnss_channels: int = 16,
        hidden_channels: int = 64,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.history_steps = int(history_steps)
        self.forecast_horizon = int(forecast_horizon)
        self.gnss_encoder = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(self.history_steps * n_gnss),
            nn.Linear(self.history_steps * n_gnss, 128),
            nn.GELU(),
            nn.Linear(128, gnss_channels),
            nn.GELU(),
        )

        in_channels = self.history_steps + gnss_channels
        self.seg1_net = _make_residual_conv_stack(in_channels, hidden_channels, self.forecast_horizon, num_blocks)
        self.seg2_net = _make_residual_conv_stack(in_channels, hidden_channels, self.forecast_horizon, num_blocks)

    @staticmethod
    def _expand_gnss_features(gnss_features: torch.Tensor, width: int) -> torch.Tensor:
        return gnss_features[:, :, None, None].expand(-1, -1, GRID_DEPTH, width)

    def forward(self, history_slip: torch.Tensor, history_gnss: torch.Tensor) -> torch.Tensor:
        seg1, seg2 = slip_vector_to_segments_torch(history_slip)
        gnss_features = self.gnss_encoder(history_gnss)
        seg1_input = torch.cat([seg1, self._expand_gnss_features(gnss_features, GRID_WIDTH_SEG1)], dim=1)
        seg2_input = torch.cat([seg2, self._expand_gnss_features(gnss_features, GRID_WIDTH_SEG2)], dim=1)

        seg1_delta = self.seg1_net(seg1_input)
        seg2_delta = self.seg2_net(seg2_input)
        seg1_pred = F.relu(seg1[:, -1:].expand(-1, self.forecast_horizon, -1, -1) + seg1_delta)
        seg2_pred = F.relu(seg2[:, -1:].expand(-1, self.forecast_horizon, -1, -1) + seg2_delta)
        return slip_segments_to_vector_torch(seg1_pred, seg2_pred)
