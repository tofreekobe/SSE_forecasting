# -*- coding: utf-8 -*-
"""Forecasting data contract for the SSE refactor.

This module is deliberately separate from the legacy ``SSEDataset``. It exposes
future-slip forecasting samples with explicit history/future windows, global
GNSS normalization, and a nonnegative-compatible slip transform.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
from torch.utils.data import Dataset


SLIP_START = 1
SLIP_END = 3031
GNSS_START = 3031
GNSS_END = 3040
DEFAULT_HISTORY_RATIO = 0.7
DEFAULT_FORECAST_HORIZON = 50
DEFAULT_FORECAST_START = 60
GRID_DEPTH = 15
GRID_WIDTH_SEG1 = 166
GRID_WIDTH_SEG2 = 36
GRID_WIDTH_TOTAL = 202
SEG1_SIZE = GRID_DEPTH * GRID_WIDTH_SEG1


def parse_event_id(path: str | Path) -> int:
    match = re.search(r"fault_sse_catalog_(\d+)\.txt$", str(path).replace("\\", "/"))
    if not match:
        raise ValueError(f"Cannot parse event id from path: {path}")
    return int(match.group(1))


@dataclass(frozen=True)
class EventFile:
    event_id: int
    path: Path


@dataclass(frozen=True)
class EventSplits:
    train: list[int]
    val: list[int]
    test: list[int]
    protocol: str


@dataclass(frozen=True)
class SlipLog1pTransform:
    """Nonnegative-compatible slip transform."""

    scale: float

    def encode(self, slip: np.ndarray) -> np.ndarray:
        return np.log1p(np.maximum(slip, 0.0) / self.scale).astype(np.float32)

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        return (np.expm1(encoded) * self.scale).astype(np.float32)


@dataclass(frozen=True)
class GNSSNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def encode(self, gnss: np.ndarray) -> np.ndarray:
        return ((gnss - self.mean) / self.std).astype(np.float32)

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        return (encoded * self.std + self.mean).astype(np.float32)


@dataclass(frozen=True)
class ForecastContractStats:
    split_protocol: str
    train_event_count: int
    val_event_count: int
    test_event_count: int
    history_steps: int
    forecast_horizon: int
    slip_scale: float
    gnss_mean: list[float]
    gnss_std: list[float]

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")


def scan_event_files(data_dir: str | Path) -> list[EventFile]:
    root = Path(data_dir)
    files = sorted(root.glob("**/*.txt"), key=lambda p: parse_event_id(p))
    return [EventFile(parse_event_id(path), path) for path in files]


def event_file_map(data_dir: str | Path) -> dict[int, Path]:
    return {record.event_id: record.path for record in scan_event_files(data_dir)}


def make_event_splits(
    data_dir: str | Path,
    protocol: Literal["random", "blocked"] = "random",
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> EventSplits:
    event_ids = [record.event_id for record in scan_event_files(data_dir)]
    if not event_ids:
        raise RuntimeError(f"No SSE event files found in {data_dir}")

    ids = event_ids[:]
    if protocol == "random":
        rng = random.Random(seed)
        rng.shuffle(ids)
    elif protocol == "blocked":
        ids = sorted(ids)
    else:
        raise ValueError(f"Unknown split protocol: {protocol}")

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = sorted(ids[:n_train])
    val = sorted(ids[n_train : n_train + n_val])
    test = sorted(ids[n_train + n_val :])
    return EventSplits(train=train, val=val, test=test, protocol=protocol)


def load_event_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(path, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    slip = raw[:, SLIP_START:SLIP_END]
    gnss = raw[:, GNSS_START:GNSS_END]
    if slip.shape[1] != 3030 or gnss.shape[1] != 9:
        raise ValueError(f"Unexpected event shape in {path}: slip={slip.shape}, gnss={gnss.shape}")
    if np.isnan(slip).any() or np.isnan(gnss).any():
        raise ValueError(f"NaN found in {path}")
    return slip, gnss


def slip_vector_to_grid(slip: np.ndarray) -> np.ndarray:
    """Map slip vectors to the physical 15x202 two-segment fault grid."""

    if slip.shape[-1] != 3030:
        raise ValueError(f"Expected last dimension 3030, got {slip.shape}")
    seg1 = slip[..., :SEG1_SIZE].reshape(*slip.shape[:-1], GRID_DEPTH, GRID_WIDTH_SEG1)
    seg2 = slip[..., SEG1_SIZE:].reshape(*slip.shape[:-1], GRID_DEPTH, GRID_WIDTH_SEG2)
    return np.concatenate([seg1, seg2], axis=-1)


def slip_grid_to_vector(grid: np.ndarray) -> np.ndarray:
    """Inverse of slip_vector_to_grid."""

    if grid.shape[-2:] != (GRID_DEPTH, GRID_WIDTH_TOTAL):
        raise ValueError(f"Expected trailing grid shape {(GRID_DEPTH, GRID_WIDTH_TOTAL)}, got {grid.shape[-2:]}")
    seg1 = grid[..., :, :GRID_WIDTH_SEG1].reshape(*grid.shape[:-2], SEG1_SIZE)
    seg2 = grid[..., :, GRID_WIDTH_SEG1:].reshape(*grid.shape[:-2], GRID_DEPTH * GRID_WIDTH_SEG2)
    return np.concatenate([seg1, seg2], axis=-1)


def fit_slip_transform(
    data_dir: str | Path,
    event_ids: Iterable[int],
    quantile: float = 0.95,
    min_scale: float = 1e-6,
    max_samples: int = 2_000_000,
    seed: int = 13,
) -> SlipLog1pTransform:
    paths = event_file_map(data_dir)
    rng = np.random.default_rng(seed)
    positives = np.empty(0, dtype=np.float32)
    for event_id in event_ids:
        slip, _ = load_event_arrays(paths[event_id])
        active = slip[slip > 1e-4]
        if active.size:
            positives = np.concatenate([positives, active.astype(np.float32)])
            if positives.size > max_samples:
                keep = rng.choice(positives.size, size=max_samples, replace=False)
                positives = positives[keep]
    if positives.size == 0:
        return SlipLog1pTransform(scale=1.0)
    scale = float(np.quantile(positives, quantile))
    return SlipLog1pTransform(scale=max(scale, min_scale))


def fit_gnss_normalizer(data_dir: str | Path, event_ids: Iterable[int]) -> GNSSNormalizer:
    paths = event_file_map(data_dir)
    count = 0
    total = np.zeros(9, dtype=np.float64)
    total_sq = np.zeros(9, dtype=np.float64)
    for event_id in event_ids:
        _, gnss = load_event_arrays(paths[event_id])
        count += gnss.shape[0]
        total += gnss.sum(axis=0, dtype=np.float64)
        total_sq += (gnss.astype(np.float64) ** 2).sum(axis=0)
    if count == 0:
        raise RuntimeError("Cannot fit GNSS normalizer on an empty training split")
    mean = total / count
    var = np.maximum(total_sq / count - mean**2, 1e-12)
    std = np.sqrt(var)
    return GNSSNormalizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


def infer_history_steps(path: str | Path, history_ratio: float = DEFAULT_HISTORY_RATIO) -> int:
    slip, _ = load_event_arrays(path)
    return int(math.floor(slip.shape[0] * history_ratio))


class SSEForecastDataset(Dataset):
    """Forecasting samples with explicit history and future windows."""

    def __init__(
        self,
        data_dir: str | Path,
        event_ids: Iterable[int],
        slip_transform: SlipLog1pTransform,
        gnss_normalizer: GNSSNormalizer,
        history_steps: int,
        forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
        return_physical: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.paths = event_file_map(data_dir)
        self.event_ids = list(event_ids)
        self.slip_transform = slip_transform
        self.gnss_normalizer = gnss_normalizer
        self.history_steps = int(history_steps)
        self.forecast_horizon = int(forecast_horizon)
        self.return_physical = return_physical

    def __len__(self) -> int:
        return len(self.event_ids)

    def __getitem__(self, idx: int) -> dict[str, object]:
        event_id = self.event_ids[idx]
        slip, gnss = load_event_arrays(self.paths[event_id])
        start = self.history_steps
        end = start + self.forecast_horizon
        if end > slip.shape[0]:
            raise ValueError(
                f"Event {event_id} has T={slip.shape[0]}, cannot take history={start}, horizon={self.forecast_horizon}"
            )

        history_slip = slip[:start]
        future_slip = slip[start:end]
        history_gnss = gnss[:start]
        future_gnss = gnss[start:end]

        item: dict[str, object] = {
            "event_id": event_id,
            "history_gnss": torch.from_numpy(self.gnss_normalizer.encode(history_gnss)),
            "history_slip": torch.from_numpy(self.slip_transform.encode(history_slip)),
            "future_gnss": torch.from_numpy(self.gnss_normalizer.encode(future_gnss)),
            "future_slip": torch.from_numpy(self.slip_transform.encode(future_slip)),
            "metadata": {
                "history_steps": self.history_steps,
                "forecast_horizon": self.forecast_horizon,
                "slip_scale": self.slip_transform.scale,
            },
        }
        if self.return_physical:
            item.update(
                {
                    "history_slip_physical": torch.from_numpy(history_slip.astype(np.float32)),
                    "future_slip_physical": torch.from_numpy(future_slip.astype(np.float32)),
                    "history_gnss_physical": torch.from_numpy(history_gnss.astype(np.float32)),
                    "future_gnss_physical": torch.from_numpy(future_gnss.astype(np.float32)),
                }
            )
        return item


def build_forecast_contract(
    data_dir: str | Path,
    protocol: Literal["random", "blocked"] = "random",
    seed: int = 42,
    history_ratio: float = DEFAULT_HISTORY_RATIO,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    forecast_start: int | None = None,
) -> tuple[EventSplits, SlipLog1pTransform, GNSSNormalizer, ForecastContractStats]:
    splits = make_event_splits(data_dir, protocol=protocol, seed=seed)
    paths = event_file_map(data_dir)
    first_path = paths[splits.train[0]]
    history_steps = int(forecast_start) if forecast_start is not None else infer_history_steps(first_path, history_ratio)
    if history_steps + forecast_horizon > load_event_arrays(first_path)[0].shape[0]:
        raise ValueError("history_ratio + forecast_horizon exceeds available timesteps")

    slip_transform = fit_slip_transform(data_dir, splits.train)
    gnss_normalizer = fit_gnss_normalizer(data_dir, splits.train)
    stats = ForecastContractStats(
        split_protocol=protocol,
        train_event_count=len(splits.train),
        val_event_count=len(splits.val),
        test_event_count=len(splits.test),
        history_steps=history_steps,
        forecast_horizon=forecast_horizon,
        slip_scale=slip_transform.scale,
        gnss_mean=gnss_normalizer.mean.tolist(),
        gnss_std=gnss_normalizer.std.tolist(),
    )
    return splits, slip_transform, gnss_normalizer, stats
