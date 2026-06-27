# -*- coding: utf-8 -*-
"""Run a trained SSE future-slip model on one event and export usable artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.forecast_contract import (
    GRID_DEPTH,
    GRID_WIDTH_SEG1,
    GRID_WIDTH_SEG2,
    GNSSNormalizer,
    SlipLog1pTransform,
)
from src.dataset.package_forecast_contract import SSEPackageForecastDataset
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


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_event(run_dir: Path, split: str, event_id: int | None, index: int) -> int:
    if event_id is not None:
        return int(event_id)
    split_path = run_dir / "split_event_ids.json"
    if not split_path.exists():
        event_ids_path = run_dir / "event_ids.json"
        if event_ids_path.exists():
            ids = [int(x) for x in _load_json(event_ids_path)]
            return ids[index]
        raise FileNotFoundError(f"Missing split_event_ids.json in {run_dir}")
    splits = _load_json(split_path)
    ids = [int(x) for x in splits[split]]
    return ids[index]


def _load_transforms(run_dir: Path, artifact_args: dict) -> tuple[SlipLog1pTransform, GNSSNormalizer]:
    stats_path = run_dir / "forecast_contract_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing forecast_contract_stats.json in {run_dir}")
    stats = _load_json(stats_path)
    return (
        SlipLog1pTransform(scale=float(stats["slip_scale"])),
        GNSSNormalizer(
            mean=np.array(stats["gnss_mean"], dtype=np.float32),
            std=np.array(stats["gnss_std"], dtype=np.float32),
        ),
    )


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


def _write_m0_csv(path: Path, true_m0: np.ndarray, pred_m0: np.ndarray, persistence_m0: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "true_sum_slip", "model_sum_slip", "persistence_sum_slip"])
        writer.writeheader()
        for idx, (yt, yp, yb) in enumerate(zip(true_m0, pred_m0, persistence_m0), start=1):
            writer.writerow(
                {
                    "step": idx,
                    "true_sum_slip": float(yt),
                    "model_sum_slip": float(yp),
                    "persistence_sum_slip": float(yb),
                }
            )


def _write_plot(
    path: Path,
    event_id: int,
    true_physical: np.ndarray,
    pred_physical: np.ndarray,
    persistence_physical: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_m0 = true_physical.sum(axis=1)
    pred_m0 = pred_physical.sum(axis=1)
    persistence_m0 = persistence_physical.sum(axis=1)
    final_idx = true_physical.shape[0] - 1
    vmax = max(float(true_physical[final_idx].max()), float(pred_physical[final_idx].max()), 1e-12)

    fig = plt.figure(figsize=(13, 7))
    ax0 = fig.add_subplot(2, 1, 1)
    steps = np.arange(1, len(true_m0) + 1)
    ax0.plot(steps, true_m0, label="true", linewidth=2)
    ax0.plot(steps, pred_m0, label="model", linewidth=2)
    ax0.plot(steps, persistence_m0, label="persistence", linestyle="--")
    ax0.set_title(f"Event {event_id}: 50-step future slip moment proxy")
    ax0.set_xlabel("Forecast step")
    ax0.set_ylabel("Sum slip")
    ax0.grid(alpha=0.25)
    ax0.legend()

    panels = [
        ("True final slip", _vector_to_grid(true_physical[final_idx]), vmax),
        ("Model final slip", _vector_to_grid(pred_physical[final_idx]), vmax),
        ("Abs error", _vector_to_grid(np.abs(pred_physical[final_idx] - true_physical[final_idx])), None),
    ]
    for idx, (title, grid, panel_vmax) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 3, 3 + idx)
        im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0.0, vmax=panel_vmax)
        ax.set_title(title)
        ax.set_xlabel("Along strike grid")
        ax.set_ylabel("Depth grid")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one-event SSE future-slip inference.")
    parser.add_argument("--run-dir", required=True, help="Training run directory containing model.pt and split_event_ids.json.")
    parser.add_argument("--package-dir", default="hf_dataset_package")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--event-id", type=int, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--save-arrays", action="store_true", help="Also save physical true/model/persistence arrays as NPZ.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    checkpoint_path = run_dir / "model.pt"
    artifact = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = artifact["args"]
    event_id = _select_event(run_dir, args.split, args.event_id, args.index)
    slip_transform, gnss_normalizer = _load_transforms(run_dir, saved_args)

    dataset = SSEPackageForecastDataset(
        args.package_dir,
        [event_id],
        slip_transform,
        gnss_normalizer,
        history_steps=int(saved_args["forecast_start"]),
        forecast_horizon=int(saved_args["forecast_horizon"]),
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))

    model = _build_model(saved_args, device)
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred_encoded = model(batch["history_slip"].to(device), batch["history_gnss"].to(device)).cpu().numpy()

    target_encoded = batch["future_slip"].numpy()
    persistence_encoded = np.repeat(batch["history_slip"][:, -1:, :].numpy(), target_encoded.shape[1], axis=1)
    pred_physical = slip_transform.decode(pred_encoded)[0]
    true_physical = slip_transform.decode(target_encoded)[0]
    persistence_physical = slip_transform.decode(persistence_encoded)[0]

    true_m0 = true_physical.sum(axis=1)
    pred_m0 = pred_physical.sum(axis=1)
    persistence_m0 = persistence_physical.sum(axis=1)
    model_rmse = float(np.sqrt(np.mean((pred_physical - true_physical) ** 2)))
    persistence_rmse = float(np.sqrt(np.mean((persistence_physical - true_physical) ** 2)))
    summary = {
        "event_id": event_id,
        "run_dir": str(run_dir),
        "split": args.split,
        "model_type": saved_args.get("model_type", "n/a"),
        "forecast_start": int(saved_args["forecast_start"]),
        "forecast_horizon": int(saved_args["forecast_horizon"]),
        "model_rmse": model_rmse,
        "persistence_rmse": persistence_rmse,
        "rmse_improvement_pct": 100.0 * (persistence_rmse - model_rmse) / max(persistence_rmse, 1e-12),
        "model_m0_rel_abs": float(np.mean(np.abs(pred_m0 - true_m0) / np.maximum(np.abs(true_m0), 1e-8))),
        "persistence_m0_rel_abs": float(
            np.mean(np.abs(persistence_m0 - true_m0) / np.maximum(np.abs(true_m0), 1e-8))
        ),
    }

    prefix = output_dir / f"forecast_event_{event_id}"
    (prefix.with_suffix(".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_m0_csv(prefix.with_suffix(".m0.csv"), true_m0, pred_m0, persistence_m0)
    _write_plot(prefix.with_suffix(".png"), event_id, true_physical, pred_physical, persistence_physical)
    if args.save_arrays:
        np.savez_compressed(
            prefix.with_suffix(".arrays.npz"),
            true_slip=true_physical,
            model_slip=pred_physical,
            persistence_slip=persistence_physical,
            true_m0=true_m0,
            model_m0=pred_m0,
            persistence_m0=persistence_m0,
        )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
