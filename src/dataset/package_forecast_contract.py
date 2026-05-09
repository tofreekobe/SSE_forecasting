# -*- coding: utf-8 -*-
"""Forecasting contract helpers for the compressed HF diagnostics package."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Iterable, Iterator, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset.forecast_contract import (
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_FORECAST_START,
    DEFAULT_HISTORY_RATIO,
    EventSplits,
    ForecastContractStats,
    GNSSNormalizer,
    SlipLog1pTransform,
)


def scan_package_event_ids(package_dir: str | Path) -> list[int]:
    manifest_path = Path(package_dir) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.csv in {package_dir}")
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        return sorted(int(row["event_id"]) for row in csv.DictReader(f))


def make_event_splits_from_ids(
    event_ids: Iterable[int],
    protocol: Literal["random", "blocked"] = "random",
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> EventSplits:
    ids = list(event_ids)
    if not ids:
        raise RuntimeError("Cannot split an empty event id list")
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
    return EventSplits(
        train=sorted(ids[:n_train]),
        val=sorted(ids[n_train : n_train + n_val]),
        test=sorted(ids[n_train + n_val :]),
        protocol=protocol,
    )


def iter_package_events(
    package_dir: str | Path,
    event_ids: Iterable[int] | None = None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    root = Path(package_dir)
    wanted = set(event_ids) if event_ids is not None else None
    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.csv in {package_dir}")

    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    shard_targets: dict[str, set[int] | None] = {}
    for row in rows:
        shard_file = row.get("shard_file", "")
        if not shard_file:
            continue
        event_id = int(row["event_id"])
        if wanted is None:
            if shard_file not in seen:
                shard_targets[shard_file] = None
                seen.add(shard_file)
        elif event_id in wanted:
            shard_targets.setdefault(shard_file, set()).add(event_id)

    for shard_file, ids_in_shard in shard_targets.items():
        with np.load(root / shard_file) as shard:
            shard_ids = shard["event_id"]
            slips = shard["slip"]
            gnss = shard["gnss"]
            for event_id, slip, gnss_event in zip(shard_ids, slips, gnss):
                event_id_int = int(event_id)
                if ids_in_shard is None or event_id_int in ids_in_shard:
                    yield event_id_int, slip.astype(np.float32), gnss_event.astype(np.float32)


def fit_package_slip_transform(
    package_dir: str | Path,
    event_ids: Iterable[int],
    quantile: float = 0.95,
    min_scale: float = 1e-6,
    max_samples: int = 2_000_000,
    seed: int = 13,
) -> SlipLog1pTransform:
    rng = np.random.default_rng(seed)
    positives = np.empty(0, dtype=np.float32)
    for _, slip, _ in iter_package_events(package_dir, event_ids):
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


def fit_package_gnss_normalizer(package_dir: str | Path, event_ids: Iterable[int]) -> GNSSNormalizer:
    count = 0
    total = np.zeros(9, dtype=np.float64)
    total_sq = np.zeros(9, dtype=np.float64)
    for _, _, gnss in iter_package_events(package_dir, event_ids):
        count += gnss.shape[0]
        total += gnss.sum(axis=0, dtype=np.float64)
        total_sq += (gnss.astype(np.float64) ** 2).sum(axis=0)
    if count == 0:
        raise RuntimeError("Cannot fit GNSS normalizer on an empty training split")
    mean = total / count
    var = np.maximum(total_sq / count - mean**2, 1e-12)
    return GNSSNormalizer(mean=mean.astype(np.float32), std=np.sqrt(var).astype(np.float32))


def infer_package_history_steps(package_dir: str | Path, history_ratio: float = DEFAULT_HISTORY_RATIO) -> int:
    for _, slip, _ in iter_package_events(package_dir):
        return int(math.floor(slip.shape[0] * history_ratio))
    raise RuntimeError(f"No package events found in {package_dir}")


def build_package_forecast_contract(
    package_dir: str | Path,
    protocol: Literal["random", "blocked"] = "random",
    seed: int = 42,
    history_ratio: float = DEFAULT_HISTORY_RATIO,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    forecast_start: int | None = None,
) -> tuple[EventSplits, SlipLog1pTransform, GNSSNormalizer, ForecastContractStats]:
    event_ids = scan_package_event_ids(package_dir)
    splits = make_event_splits_from_ids(event_ids, protocol=protocol, seed=seed)
    history_steps = int(forecast_start) if forecast_start is not None else infer_package_history_steps(package_dir, history_ratio)
    for _, slip, _ in iter_package_events(package_dir, splits.train[:1]):
        if history_steps + forecast_horizon > slip.shape[0]:
            raise ValueError("history_ratio + forecast_horizon exceeds available timesteps")
        break

    slip_transform = fit_package_slip_transform(package_dir, splits.train)
    gnss_normalizer = fit_package_gnss_normalizer(package_dir, splits.train)
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


class SSEPackageForecastDataset(Dataset):
    """In-memory forecasting dataset backed by compressed HF package shards."""

    def __init__(
        self,
        package_dir: str | Path,
        event_ids: Iterable[int],
        slip_transform: SlipLog1pTransform,
        gnss_normalizer: GNSSNormalizer,
        history_steps: int,
        forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
        return_physical: bool = False,
    ) -> None:
        self.event_ids = list(event_ids)
        self.slip_transform = slip_transform
        self.gnss_normalizer = gnss_normalizer
        self.history_steps = int(history_steps)
        self.forecast_horizon = int(forecast_horizon)
        self.return_physical = return_physical
        self.samples: list[dict[str, object]] = []

        for event_id, slip, gnss in iter_package_events(package_dir, self.event_ids):
            start = self.history_steps
            end = start + self.forecast_horizon
            if end > slip.shape[0]:
                raise ValueError(
                    f"Event {event_id} has T={slip.shape[0]}, cannot take history={start}, "
                    f"horizon={self.forecast_horizon}"
                )
            history_slip = slip[:start]
            future_slip = slip[start:end]
            history_gnss = gnss[:start]
            future_gnss = gnss[start:end]

            sample: dict[str, object] = {
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
                sample.update(
                    {
                        "history_slip_physical": torch.from_numpy(history_slip.astype(np.float32)),
                        "future_slip_physical": torch.from_numpy(future_slip.astype(np.float32)),
                        "history_gnss_physical": torch.from_numpy(history_gnss.astype(np.float32)),
                        "future_gnss_physical": torch.from_numpy(future_gnss.astype(np.float32)),
                    }
                )
            self.samples.append(sample)

        loaded_ids = {int(sample["event_id"]) for sample in self.samples}
        missing = sorted(set(self.event_ids) - loaded_ids)
        if missing:
            raise RuntimeError(f"Package missing requested event ids: {missing[:10]}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.samples[idx]
