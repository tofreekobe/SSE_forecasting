# -*- coding: utf-8 -*-
"""Plot true/model/persistence forecast examples from a trained SSE checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.forecast_contract import GRID_DEPTH, GRID_WIDTH_SEG1, GRID_WIDTH_SEG2
from src.dataset.package_forecast_contract import (
    SSEPackageForecastDataset,
    fit_package_gnss_normalizer,
    fit_package_slip_transform,
)
from src.models.small_forecast_net import (
    SegmentedResidualForecastNet,
    SegmentedSlipConvForecastNet,
    SlipConvForecastNet,
)


MODEL_CLASSES = {
    "segmented_residual": SegmentedResidualForecastNet,
    "segmented": SegmentedSlipConvForecastNet,
    "plain": SlipConvForecastNet,
}


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_event_ids(run_dir: Path, split: str, max_events: int) -> tuple[list[int], list[int]]:
    small_ids = run_dir / "event_ids.json"
    if small_ids.exists():
        ids = [int(x) for x in _load_json(small_ids)]
        return ids, ids[:max_events]

    split_path = run_dir / "split_event_ids.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Cannot find event_ids.json or split_event_ids.json in {run_dir}")
    splits = _load_json(split_path)
    train_ids = [int(x) for x in splits["train"]]
    selected = [int(x) for x in splits[split]][:max_events]
    return train_ids, selected


def _build_model(saved_args: dict, device: torch.device) -> torch.nn.Module:
    model_type = saved_args.get("model_type", "segmented_residual")
    model_cls = MODEL_CLASSES[model_type]
    model = model_cls(
        history_steps=int(saved_args["forecast_start"]),
        forecast_horizon=int(saved_args["forecast_horizon"]),
        hidden_channels=int(saved_args["hidden_channels"]),
    )
    return model.to(device)


def _vector_to_grid(arr: np.ndarray) -> np.ndarray:
    seg1_size = GRID_DEPTH * GRID_WIDTH_SEG1
    seg1 = arr[:seg1_size].reshape(GRID_DEPTH, GRID_WIDTH_SEG1)
    seg2 = arr[seg1_size:].reshape(GRID_DEPTH, GRID_WIDTH_SEG2)
    return np.concatenate([seg1, seg2], axis=1)


def _write_history_plot(run_dir: Path, output_dir: Path) -> None:
    history_path = run_dir / "training_history.csv"
    if not history_path.exists():
        return
    rows: list[dict[str, float]] = []
    with history_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items() if value != ""})
    if not rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = [row["epoch"] for row in rows]
    if "train_loss" in rows[0]:
        axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train loss")
    if "encoded_mse" in rows[0]:
        axes[0].plot(epochs, [row["encoded_mse"] for row in rows], label="encoded mse")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_title("Optimization")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    if "physical_rmse" in rows[0]:
        axes[1].plot(epochs, [row["physical_rmse"] for row in rows], label="train subset rmse")
    if "val_rmse" in rows[0]:
        axes[1].plot(epochs, [row["val_rmse"] for row in rows], label="val rmse")
    if "train_rmse" in rows[0]:
        axes[1].plot(epochs, [row["train_rmse"] for row in rows], label="train rmse")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Physical RMSE")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot examples from an SSE future-slip checkpoint.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--package-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for plotting, but torch.cuda.is_available() is false")

    artifact = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = artifact["args"]
    package_dir = args.package_dir or saved_args.get("package_dir", "hf_dataset_package")
    train_ids, selected_ids = _infer_event_ids(run_dir, args.split, args.max_events)

    slip_transform = fit_package_slip_transform(package_dir, train_ids)
    gnss_normalizer = fit_package_gnss_normalizer(package_dir, train_ids)
    dataset = SSEPackageForecastDataset(
        package_dir,
        selected_ids,
        slip_transform,
        gnss_normalizer,
        history_steps=int(saved_args["forecast_start"]),
        forecast_horizon=int(saved_args["forecast_horizon"]),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = _build_model(saved_args, device)
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for batch in loader:
        event_id = int(batch["event_id"].item())
        history_slip = batch["history_slip"].to(device)
        history_gnss = batch["history_gnss"].to(device)
        target = batch["future_slip"].numpy()
        with torch.no_grad():
            pred = model(history_slip, history_gnss).cpu().numpy()
        persistence = np.repeat(batch["history_slip"][:, -1:, :].numpy(), target.shape[1], axis=1)

        pred_physical = slip_transform.decode(pred)[0]
        target_physical = slip_transform.decode(target)[0]
        persistence_physical = slip_transform.decode(persistence)[0]

        pred_m0 = pred_physical.sum(axis=1)
        target_m0 = target_physical.sum(axis=1)
        persistence_m0 = persistence_physical.sum(axis=1)
        rmse = float(np.sqrt(np.mean((pred_physical - target_physical) ** 2)))
        persistence_rmse = float(np.sqrt(np.mean((persistence_physical - target_physical) ** 2)))
        rows.append(
            {
                "event_id": event_id,
                "model_rmse": rmse,
                "persistence_rmse": persistence_rmse,
                "rmse_ratio": rmse / max(persistence_rmse, 1e-12),
            }
        )

        final_idx = target_physical.shape[0] - 1
        maps = [
            ("True final slip", _vector_to_grid(target_physical[final_idx])),
            ("Model final slip", _vector_to_grid(pred_physical[final_idx])),
            ("Abs error", _vector_to_grid(np.abs(pred_physical[final_idx] - target_physical[final_idx]))),
        ]
        fig = plt.figure(figsize=(13, 7))
        ax0 = fig.add_subplot(2, 1, 1)
        steps = np.arange(1, target_m0.shape[0] + 1)
        ax0.plot(steps, target_m0, label="true", linewidth=2)
        ax0.plot(steps, pred_m0, label="model", linewidth=2)
        ax0.plot(steps, persistence_m0, label="persistence", linestyle="--")
        ax0.set_title(f"Event {event_id}: future moment proxy")
        ax0.set_xlabel("Forecast step")
        ax0.set_ylabel("Sum slip")
        ax0.grid(alpha=0.25)
        ax0.legend()

        vmax = max(float(np.max(maps[0][1])), float(np.max(maps[1][1])), 1e-12)
        for i, (title, grid) in enumerate(maps, start=1):
            ax = fig.add_subplot(2, 3, 3 + i)
            im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax if i < 3 else None)
            ax.set_title(title)
            ax.set_xlabel("Along strike grid")
            ax.set_ylabel("Depth grid")
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

        fig.tight_layout()
        fig.savefig(output_dir / f"forecast_event_{event_id}.png", dpi=180)
        plt.close(fig)

    _write_history_plot(run_dir, output_dir)
    (output_dir / "forecast_example_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "examples": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
