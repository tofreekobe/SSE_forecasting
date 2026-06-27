# -*- coding: utf-8 -*-
"""Demonstration-only GNSS-to-slip inversion proxy for SSE events.

This is not a replacement for a physics-based or separately trained inversion
model. It provides an operational demo artifact: fit a small ridge map from
GNSS history features to the current slip field, then export metrics and maps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.forecast_contract import GRID_DEPTH, GRID_WIDTH_SEG1, GRID_WIDTH_SEG2
from src.dataset.package_forecast_contract import iter_package_events


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _gnss_features(gnss_history: np.ndarray) -> np.ndarray:
    steps = np.arange(gnss_history.shape[0], dtype=np.float32)
    centered_steps = steps - steps.mean()
    denom = float(np.sum(centered_steps**2)) or 1.0
    trend = (centered_steps[:, None] * (gnss_history - gnss_history.mean(axis=0))).sum(axis=0) / denom
    return np.concatenate(
        [
            gnss_history.mean(axis=0),
            gnss_history.std(axis=0),
            gnss_history[-1],
            trend.astype(np.float32),
        ]
    ).astype(np.float32)


def _split_vector(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seg1_size = GRID_DEPTH * GRID_WIDTH_SEG1
    return (
        arr[:seg1_size].reshape(GRID_DEPTH, GRID_WIDTH_SEG1),
        arr[seg1_size:].reshape(GRID_DEPTH, GRID_WIDTH_SEG2),
    )


def _fit_ridge(
    package_dir: Path,
    train_ids: list[int],
    history_steps: int,
    max_train_events: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = train_ids[:max_train_events] if max_train_events > 0 else train_ids
    features = []
    targets = []
    for _, slip, gnss in iter_package_events(package_dir, selected):
        features.append(_gnss_features(gnss[:history_steps]))
        targets.append(slip[history_steps - 1].astype(np.float32))
    if not features:
        raise RuntimeError("No training events found for inversion proxy")

    x = np.stack(features, axis=0).astype(np.float64)
    y = np.stack(targets, axis=0).astype(np.float64)
    mean_x = x.mean(axis=0, keepdims=True)
    std_x = np.maximum(x.std(axis=0, keepdims=True), 1e-8)
    x_norm = (x - mean_x) / std_x
    design = np.concatenate([np.ones((x_norm.shape[0], 1), dtype=np.float64), x_norm], axis=1)
    reg = alpha * np.eye(design.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + reg, design.T @ y)
    norm = np.concatenate([mean_x.reshape(-1), std_x.reshape(-1)]).astype(np.float32)
    return weights.astype(np.float32), norm


def _predict(weights: np.ndarray, norm: np.ndarray, gnss_history: np.ndarray) -> np.ndarray:
    n = norm.shape[0] // 2
    mean_x = norm[:n]
    std_x = norm[n:]
    x = (_gnss_features(gnss_history) - mean_x) / std_x
    design = np.concatenate([np.ones(1, dtype=np.float32), x.astype(np.float32)])
    return np.maximum(design @ weights, 0.0).astype(np.float32)


def _load_event(package_dir: Path, event_id: int) -> tuple[np.ndarray, np.ndarray]:
    for loaded_id, slip, gnss in iter_package_events(package_dir, [event_id]):
        if int(loaded_id) == int(event_id):
            return slip, gnss
    raise RuntimeError(f"Event {event_id} not found")


def _write_plot(path: Path, event_id: int, true_slip: np.ndarray, pred_slip: np.ndarray, mean_slip: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_seg1, true_seg2 = _split_vector(true_slip)
    pred_seg1, pred_seg2 = _split_vector(pred_slip)
    err_seg1, err_seg2 = _split_vector(np.abs(pred_slip - true_slip))
    mean_seg1, mean_seg2 = _split_vector(mean_slip)
    panels = [
        ("True segment 1", true_seg1),
        ("Proxy inversion segment 1", pred_seg1),
        ("Mean baseline segment 1", mean_seg1),
        ("Abs error segment 1", err_seg1),
        ("True segment 2", true_seg2),
        ("Proxy inversion segment 2", pred_seg2),
        ("Mean baseline segment 2", mean_seg2),
        ("Abs error segment 2", err_seg2),
    ]
    vmax = max(float(true_slip.max()), float(pred_slip.max()), float(mean_slip.max()), 1e-12)

    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, (title, grid) in zip(axes.reshape(-1), panels):
        panel_vmax = None if "error" in title.lower() else vmax
        im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0.0, vmax=panel_vmax)
        ax.set_title(title)
        ax.set_xlabel("Along strike")
        ax.set_ylabel("Depth")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Event {event_id}: demonstration GNSS-to-slip inversion proxy", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a demonstration GNSS-to-slip inversion proxy.")
    parser.add_argument("--run-dir", required=True, help="Forecast run directory with split_event_ids.json and stats.")
    parser.add_argument("--package-dir", default="hf_dataset_package")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--event-id", type=int, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-train-events", type=int, default=1200)
    parser.add_argument("--alpha", type=float, default=10.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    package_dir = Path(args.package_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "demo_inversion_proxy"
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = _load_json(run_dir / "split_event_ids.json")
    stats = _load_json(run_dir / "forecast_contract_stats.json")
    history_steps = int(stats["history_steps"])
    train_ids = [int(x) for x in splits["train"]]
    split_ids = [int(x) for x in splits[args.split]]
    event_id = int(args.event_id) if args.event_id is not None else split_ids[args.index]

    weights, norm = _fit_ridge(package_dir, train_ids, history_steps, args.max_train_events, args.alpha)
    slip, gnss = _load_event(package_dir, event_id)
    true_slip = slip[history_steps - 1].astype(np.float32)
    pred_slip = _predict(weights, norm, gnss[:history_steps])

    train_targets = []
    selected_train = train_ids[: args.max_train_events] if args.max_train_events > 0 else train_ids
    for _, train_slip, _ in iter_package_events(package_dir, selected_train):
        train_targets.append(train_slip[history_steps - 1].astype(np.float32))
    mean_slip = np.mean(np.stack(train_targets, axis=0), axis=0).astype(np.float32)

    rmse = float(np.sqrt(np.mean((pred_slip - true_slip) ** 2)))
    mean_rmse = float(np.sqrt(np.mean((mean_slip - true_slip) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(true_slip**2)))
    true_m0 = float(np.sum(true_slip))
    pred_m0 = float(np.sum(pred_slip))
    mean_m0 = float(np.sum(mean_slip))
    summary = {
        "mode": "demo_inversion_proxy",
        "warning": "Demonstration-only ridge proxy; not a professional slip inversion model.",
        "event_id": event_id,
        "history_steps": history_steps,
        "train_event_count": len(selected_train),
        "alpha": args.alpha,
        "rmse_proxy": rmse,
        "rmse_train_mean": mean_rmse,
        "rmse_zero": zero_rmse,
        "rmse_improvement_vs_mean_pct": 100.0 * (mean_rmse - rmse) / max(mean_rmse, 1e-12),
        "true_sum_slip": true_m0,
        "proxy_sum_slip": pred_m0,
        "mean_sum_slip": mean_m0,
        "proxy_m0_rel_abs": abs(pred_m0 - true_m0) / max(abs(true_m0), 1e-8),
        "mean_m0_rel_abs": abs(mean_m0 - true_m0) / max(abs(true_m0), 1e-8),
    }

    prefix = output_dir / f"inversion_proxy_event_{event_id}"
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        true_slip=true_slip,
        proxy_slip=pred_slip,
        mean_slip=mean_slip,
    )
    _write_plot(prefix.with_suffix(".png"), event_id, true_slip, pred_slip, mean_slip)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
