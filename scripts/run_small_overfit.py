# -*- coding: utf-8 -*-
"""Run a small overfit check for the SSE future-slip forecasting contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.forecast_contract import DEFAULT_FORECAST_START, ForecastContractStats, GNSSNormalizer
from src.dataset.package_forecast_contract import (
    SSEPackageForecastDataset,
    fit_package_gnss_normalizer,
    fit_package_slip_transform,
    make_event_splits_from_ids,
    scan_package_event_ids,
)
from src.models.forecast_baselines import (
    compute_package_train_mean_slip,
    evaluate_package_physical_baselines,
    write_metrics,
)
from src.models.small_forecast_net import SegmentedResidualForecastNet, SegmentedSlipConvForecastNet, SlipConvForecastNet


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, active_weight: float) -> torch.Tensor:
    if active_weight <= 1.0:
        return F.mse_loss(pred, target)
    weight = torch.ones_like(target)
    weight = torch.where(target > 0.0, weight * active_weight, weight)
    return torch.mean(weight * (pred - target) ** 2)


@torch.no_grad()
def evaluate_model(model, loader, slip_transform, device: torch.device) -> dict[str, float]:
    model.eval()
    pred_encoded = []
    true_encoded = []
    for batch in loader:
        history_slip = batch["history_slip"].to(device)
        history_gnss = batch["history_gnss"].to(device)
        target = batch["future_slip"].to(device)
        pred = model(history_slip, history_gnss)
        pred_encoded.append(pred.cpu().numpy())
        true_encoded.append(target.cpu().numpy())

    pred_encoded_np = np.concatenate(pred_encoded, axis=0)
    true_encoded_np = np.concatenate(true_encoded, axis=0)
    encoded_mse = float(np.mean((pred_encoded_np - true_encoded_np) ** 2))

    pred_physical = slip_transform.decode(pred_encoded_np)
    true_physical = slip_transform.decode(true_encoded_np)
    err = pred_physical - true_physical
    rmse = float(np.sqrt(np.mean(err**2)))
    yt = true_physical.reshape(-1)
    yp = pred_physical.reshape(-1)
    r2 = float(1.0 - np.sum((yp - yt) ** 2) / max(np.sum((yt - float(np.mean(yt))) ** 2), 1e-12))
    true_m0 = np.sum(true_physical, axis=2)
    pred_m0 = np.sum(pred_physical, axis=2)
    m0_rel_abs = float(np.mean(np.abs(pred_m0 - true_m0) / np.maximum(np.abs(true_m0), 1e-8)))
    return {
        "encoded_mse": encoded_mse,
        "physical_rmse": rmse,
        "physical_r2": r2,
        "m0_rel_abs": m0_rel_abs,
    }


def write_history(output_dir: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with (output_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        plt.plot([row["epoch"] for row in rows], [row["encoded_mse"] for row in rows], label="encoded MSE")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curve.png", dpi=160)
        plt.close()
    except Exception:
        pass


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    model_classes = {
        "segmented_residual": SegmentedResidualForecastNet,
        "segmented": SegmentedSlipConvForecastNet,
        "plain": SlipConvForecastNet,
    }
    model_cls = model_classes[args.model_type]
    return model_cls(
        history_steps=args.forecast_start,
        forecast_horizon=args.forecast_horizon,
        hidden_channels=args.hidden_channels,
    )


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description="Small overfit check for SSE future-slip forecasting.")
    parser.add_argument("--package-dir", default="hf_dataset_package")
    parser.add_argument("--output-dir", default="small_overfit_results")
    parser.add_argument("--max-events", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forecast-start", type=int, default=DEFAULT_FORECAST_START)
    parser.add_argument("--forecast-horizon", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--model-type", choices=["segmented_residual", "segmented", "plain"], default="segmented_residual")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--active-weight", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ids = scan_package_event_ids(args.package_dir)
    splits = make_event_splits_from_ids(all_ids, protocol="random", seed=args.seed)
    event_ids = splits.train[: args.max_events]
    slip_transform = fit_package_slip_transform(args.package_dir, event_ids)
    gnss_normalizer = fit_package_gnss_normalizer(args.package_dir, event_ids)
    stats = ForecastContractStats(
        split_protocol="small_overfit_random_train_subset",
        train_event_count=len(event_ids),
        val_event_count=0,
        test_event_count=0,
        history_steps=args.forecast_start,
        forecast_horizon=args.forecast_horizon,
        slip_scale=slip_transform.scale,
        gnss_mean=gnss_normalizer.mean.tolist(),
        gnss_std=gnss_normalizer.std.tolist(),
    )
    stats.to_json(out_dir / "forecast_contract_stats.json")
    (out_dir / "event_ids.json").write_text(json.dumps(event_ids, indent=2), encoding="utf-8")

    dataset = SSEPackageForecastDataset(
        args.package_dir,
        event_ids,
        slip_transform,
        gnss_normalizer,
        history_steps=args.forecast_start,
        forecast_horizon=args.forecast_horizon,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    train_mean = compute_package_train_mean_slip(args.package_dir, event_ids)
    baseline_metrics = evaluate_package_physical_baselines(
        args.package_dir,
        event_ids,
        train_mean,
        history_steps=args.forecast_start,
        horizons=(1, 5, 10, 30, 50),
    )
    write_metrics(out_dir, "overfit_subset", baseline_metrics)

    device = choose_device(args.device)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in loader:
            history_slip = batch["history_slip"].to(device)
            history_gnss = batch["history_gnss"].to(device)
            target = batch["future_slip"].to(device)
            pred = model(history_slip, history_gnss)
            loss = weighted_mse(pred, target, args.active_weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            metrics = evaluate_model(model, eval_loader, slip_transform, device)
            row = {
                "epoch": float(epoch),
                "train_loss": total_loss / max(total_batches, 1),
                **metrics,
            }
            history.append(row)
            print(
                f"epoch={epoch:04d} loss={row['train_loss']:.6g} "
                f"rmse={row['physical_rmse']:.6g} r2={row['physical_r2']:.6g} "
                f"m0={row['m0_rel_abs']:.6g}",
                flush=True,
            )

    final_metrics = evaluate_model(model, eval_loader, slip_transform, device)
    h50 = baseline_metrics[str(args.forecast_horizon)]
    final_metrics.update(
        {
            "event_count": float(len(event_ids)),
            "forecast_start": float(args.forecast_start),
            "forecast_horizon": float(args.forecast_horizon),
            "model_type": args.model_type,
            "baseline_h50_persistence_rmse": float(h50["rmse_persistence"]),
            "baseline_h50_mean_rmse": float(h50["rmse_mean"]),
            "rmse_vs_persistence_ratio": final_metrics["physical_rmse"] / max(float(h50["rmse_persistence"]), 1e-12),
            "success": bool(final_metrics["physical_rmse"] < 0.5 * float(h50["rmse_persistence"])),
        }
    )

    write_history(out_dir, history)
    (out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "metrics": final_metrics}, out_dir / "model.pt")
    print(json.dumps(final_metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
