# -*- coding: utf-8 -*-
"""Streaming physical-unit baselines for SSE future-slip forecasting."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from src.dataset.forecast_contract import event_file_map, load_event_arrays
from src.dataset.package_forecast_contract import iter_package_events


DEFAULT_HORIZONS = (1, 5, 10, 30, 50)


def compute_train_mean_slip(data_dir: str | Path, train_event_ids: Iterable[int]) -> np.ndarray:
    paths = event_file_map(data_dir)
    count = 0
    total = None
    for event_id in train_event_ids:
        slip, _ = load_event_arrays(paths[event_id])
        if total is None:
            total = np.zeros_like(slip, dtype=np.float64)
        total += slip.astype(np.float64)
        count += 1
    if count == 0 or total is None:
        raise RuntimeError("Cannot compute mean baseline on empty training split")
    return (total / count).astype(np.float32)


def compute_package_train_mean_slip(package_dir: str | Path, train_event_ids: Iterable[int]) -> np.ndarray:
    count = 0
    total = None
    for _, slip, _ in iter_package_events(package_dir, train_event_ids):
        if total is None:
            total = np.zeros_like(slip, dtype=np.float64)
        total += slip.astype(np.float64)
        count += 1
    if count == 0 or total is None:
        raise RuntimeError("Cannot compute mean baseline on empty training split")
    return (total / count).astype(np.float32)


def _event_metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float]:
    err = pred - y
    mse = float(np.mean(err**2))
    sse = float(np.sum(err**2))
    yt = y.reshape(-1)
    sst = float(np.sum((yt - float(np.mean(yt))) ** 2))
    true_m0 = np.sum(y, axis=1)
    pred_m0 = np.sum(pred, axis=1)
    m0_rel = float(np.mean(np.abs(pred_m0 - true_m0) / np.maximum(np.abs(true_m0), 1e-8)))
    return mse, sse, sst, m0_rel


def evaluate_physical_baselines(
    data_dir: str | Path,
    event_ids: Iterable[int],
    train_mean_slip: np.ndarray,
    history_steps: int,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> dict[str, dict[str, float]]:
    paths = event_file_map(data_dir)
    accum = {
        str(h): {
            "event_count": 0.0,
            "zero_mse": 0.0,
            "mean_mse": 0.0,
            "persistence_mse": 0.0,
            "zero_sse": 0.0,
            "mean_sse": 0.0,
            "persistence_sse": 0.0,
            "sst": 0.0,
            "zero_m0": 0.0,
            "mean_m0": 0.0,
            "persistence_m0": 0.0,
        }
        for h in horizons
    }

    for event_id in event_ids:
        slip, _ = load_event_arrays(paths[event_id])
        last_history = slip[history_steps - 1]
        for h in horizons:
            if history_steps + h > slip.shape[0]:
                continue
            y = slip[history_steps : history_steps + h]
            zero = np.zeros_like(y)
            mean = train_mean_slip[history_steps : history_steps + h]
            persistence = np.repeat(last_history[None, :], h, axis=0)

            zero_mse, zero_sse, sst, zero_m0 = _event_metrics(y, zero)
            mean_mse, mean_sse, _, mean_m0 = _event_metrics(y, mean)
            persistence_mse, persistence_sse, _, persistence_m0 = _event_metrics(y, persistence)

            row = accum[str(h)]
            row["event_count"] += 1
            row["zero_mse"] += zero_mse
            row["mean_mse"] += mean_mse
            row["persistence_mse"] += persistence_mse
            row["zero_sse"] += zero_sse
            row["mean_sse"] += mean_sse
            row["persistence_sse"] += persistence_sse
            row["sst"] += sst
            row["zero_m0"] += zero_m0
            row["mean_m0"] += mean_m0
            row["persistence_m0"] += persistence_m0

    metrics: dict[str, dict[str, float]] = {}
    for h, row in accum.items():
        n = max(row["event_count"], 1.0)
        metrics[h] = {
            "horizon": float(h),
            "event_count": row["event_count"],
            "rmse_zero": math.sqrt(row["zero_mse"] / n),
            "rmse_mean": math.sqrt(row["mean_mse"] / n),
            "rmse_persistence": math.sqrt(row["persistence_mse"] / n),
            "r2_zero": 1.0 - row["zero_sse"] / max(row["sst"], 1e-12),
            "r2_mean": 1.0 - row["mean_sse"] / max(row["sst"], 1e-12),
            "r2_persistence": 1.0 - row["persistence_sse"] / max(row["sst"], 1e-12),
            "m0_rel_abs_zero": row["zero_m0"] / n,
            "m0_rel_abs_mean": row["mean_m0"] / n,
            "m0_rel_abs_persistence": row["persistence_m0"] / n,
        }
    return metrics


def evaluate_package_physical_baselines(
    package_dir: str | Path,
    event_ids: Iterable[int],
    train_mean_slip: np.ndarray,
    history_steps: int,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> dict[str, dict[str, float]]:
    accum = {
        str(h): {
            "event_count": 0.0,
            "zero_mse": 0.0,
            "mean_mse": 0.0,
            "persistence_mse": 0.0,
            "zero_sse": 0.0,
            "mean_sse": 0.0,
            "persistence_sse": 0.0,
            "sst": 0.0,
            "zero_m0": 0.0,
            "mean_m0": 0.0,
            "persistence_m0": 0.0,
        }
        for h in horizons
    }

    for _, slip, _ in iter_package_events(package_dir, event_ids):
        last_history = slip[history_steps - 1]
        for h in horizons:
            if history_steps + h > slip.shape[0]:
                continue
            y = slip[history_steps : history_steps + h]
            zero = np.zeros_like(y)
            mean = train_mean_slip[history_steps : history_steps + h]
            persistence = np.repeat(last_history[None, :], h, axis=0)

            zero_mse, zero_sse, sst, zero_m0 = _event_metrics(y, zero)
            mean_mse, mean_sse, _, mean_m0 = _event_metrics(y, mean)
            persistence_mse, persistence_sse, _, persistence_m0 = _event_metrics(y, persistence)

            row = accum[str(h)]
            row["event_count"] += 1
            row["zero_mse"] += zero_mse
            row["mean_mse"] += mean_mse
            row["persistence_mse"] += persistence_mse
            row["zero_sse"] += zero_sse
            row["mean_sse"] += mean_sse
            row["persistence_sse"] += persistence_sse
            row["sst"] += sst
            row["zero_m0"] += zero_m0
            row["mean_m0"] += mean_m0
            row["persistence_m0"] += persistence_m0

    metrics: dict[str, dict[str, float]] = {}
    for h, row in accum.items():
        n = max(row["event_count"], 1.0)
        metrics[h] = {
            "horizon": float(h),
            "event_count": row["event_count"],
            "rmse_zero": math.sqrt(row["zero_mse"] / n),
            "rmse_mean": math.sqrt(row["mean_mse"] / n),
            "rmse_persistence": math.sqrt(row["persistence_mse"] / n),
            "r2_zero": 1.0 - row["zero_sse"] / max(row["sst"], 1e-12),
            "r2_mean": 1.0 - row["mean_sse"] / max(row["sst"], 1e-12),
            "r2_persistence": 1.0 - row["persistence_sse"] / max(row["sst"], 1e-12),
            "m0_rel_abs_zero": row["zero_m0"] / n,
            "m0_rel_abs_mean": row["mean_m0"] / n,
            "m0_rel_abs_persistence": row["persistence_m0"] / n,
        }
    return metrics


def write_metrics(output_dir: str | Path, split_name: str, metrics: dict[str, dict[str, float]]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"baseline_{split_name}.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    rows = [metrics[h] for h in sorted(metrics, key=lambda x: int(x))]
    csv_path = out / f"baseline_{split_name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
