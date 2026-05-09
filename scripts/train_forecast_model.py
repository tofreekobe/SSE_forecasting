# -*- coding: utf-8 -*-
"""Train the refactored future-slip forecasting model on random or blocked splits."""

from __future__ import annotations

import argparse
import shutil
import csv
import json
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

from src.dataset.forecast_contract import DEFAULT_FORECAST_START, ForecastContractStats
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
from src.models.small_forecast_net import SegmentedSlipConvForecastNet, SlipConvForecastNet


def _limit(ids: list[int], max_items: int | None) -> list[int]:
    return ids if max_items is None else ids[:max_items]


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, active_weight: float) -> torch.Tensor:
    if active_weight <= 1.0:
        return F.mse_loss(pred, target)
    weight = torch.where(target > 0.0, torch.full_like(target, active_weight), torch.ones_like(target))
    return torch.mean(weight * (pred - target) ** 2)


def forecast_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    active_weight: float,
    m0_loss_weight: float,
    slip_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mse_loss = weighted_mse(pred, target, active_weight)
    if m0_loss_weight <= 0.0:
        return mse_loss, {"encoded_mse_loss": mse_loss.detach(), "m0_loss": torch.zeros((), device=pred.device)}

    pred_physical = torch.expm1(torch.clamp(pred, min=0.0)) * slip_scale
    target_physical = torch.expm1(target) * slip_scale
    pred_m0 = pred_physical.sum(dim=2)
    true_m0 = target_physical.sum(dim=2)
    m0_loss = torch.mean(torch.abs(pred_m0 - true_m0) / torch.clamp(torch.abs(true_m0), min=1e-8))
    total = mse_loss + m0_loss_weight * m0_loss
    return total, {"encoded_mse_loss": mse_loss.detach(), "m0_loss": m0_loss.detach()}


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


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_make_writer(output_dir: Path, tensorboard_dir: str | None):
    if tensorboard_dir == "off":
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return None
    log_dir = Path(tensorboard_dir) if tensorboard_dir else output_dir / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(log_dir))


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    model_cls = SegmentedSlipConvForecastNet if args.model_type == "segmented" else SlipConvForecastNet
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


def write_report(output_dir: Path, payload: dict[str, object]) -> None:
    protocol = payload["protocol"]
    final_val = payload["final_val"]
    baseline_val = payload["baseline_val_h50"]
    status = payload["status"]
    lines = [
        f"# Forecast Training Report ({protocol})",
        "",
        f"Status: `{status}`",
        "",
        "## Data",
        "",
        f"- Train events: {payload['train_event_count']}",
        f"- Val events: {payload['val_event_count']}",
        f"- Test events: {payload['test_event_count']}",
        f"- Forecast start: {payload['forecast_start']}",
        f"- Forecast horizon: {payload['forecast_horizon']}",
        f"- Model type: `{payload['model_type']}`",
        "",
        "## Final Validation",
        "",
        f"- Model h50 RMSE: {final_val['physical_rmse']:.6g}",
        f"- Model h50 R2: {final_val['physical_r2']:.6g}",
        f"- Model h50 M0 rel abs: {final_val['m0_rel_abs']:.6g}",
        f"- Persistence h50 RMSE: {baseline_val['rmse_persistence']:.6g}",
        f"- Mean h50 RMSE: {baseline_val['rmse_mean']:.6g}",
    ]
    (output_dir / "forecast_training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train future-slip forecasting model.")
    parser.add_argument("--package-dir", default="hf_dataset_package")
    parser.add_argument("--output-dir", default="forecast_training_results")
    parser.add_argument("--protocol", choices=["random", "blocked"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forecast-start", type=int, default=DEFAULT_FORECAST_START)
    parser.add_argument("--forecast-horizon", type=int, default=50)
    parser.add_argument("--max-train-events", type=int, default=None)
    parser.add_argument("--max-val-events", type=int, default=None)
    parser.add_argument("--max-test-events", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--model-type", choices=["segmented", "plain"], default="segmented")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--active-weight", type=float, default=1.0)
    parser.add_argument("--m0-loss-weight", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision when a GPU is available.")
    parser.add_argument("--tensorboard-dir", default=None, help="TensorBoard log dir. Use 'off' to disable.")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir) / args.protocol
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ids = scan_package_event_ids(args.package_dir)
    splits = make_event_splits_from_ids(all_ids, protocol=args.protocol, seed=args.seed)
    train_ids = _limit(splits.train, args.max_train_events)
    val_ids = _limit(splits.val, args.max_val_events)
    test_ids = _limit(splits.test, args.max_test_events)
    if not train_ids or not val_ids:
        raise RuntimeError("Training and validation splits must be non-empty")

    slip_transform = fit_package_slip_transform(args.package_dir, train_ids)
    gnss_normalizer = fit_package_gnss_normalizer(args.package_dir, train_ids)
    stats = ForecastContractStats(
        split_protocol=args.protocol,
        train_event_count=len(train_ids),
        val_event_count=len(val_ids),
        test_event_count=len(test_ids),
        history_steps=args.forecast_start,
        forecast_horizon=args.forecast_horizon,
        slip_scale=slip_transform.scale,
        gnss_mean=gnss_normalizer.mean.tolist(),
        gnss_std=gnss_normalizer.std.tolist(),
    )
    stats.to_json(output_dir / "forecast_contract_stats.json")
    (output_dir / "split_event_ids.json").write_text(
        json.dumps({"train": train_ids, "val": val_ids, "test": test_ids}, indent=2),
        encoding="utf-8",
    )

    train_ds = SSEPackageForecastDataset(
        args.package_dir,
        train_ids,
        slip_transform,
        gnss_normalizer,
        history_steps=args.forecast_start,
        forecast_horizon=args.forecast_horizon,
    )
    val_ds = SSEPackageForecastDataset(
        args.package_dir,
        val_ids,
        slip_transform,
        gnss_normalizer,
        history_steps=args.forecast_start,
        forecast_horizon=args.forecast_horizon,
    )
    test_ds = (
        SSEPackageForecastDataset(
            args.package_dir,
            test_ids,
            slip_transform,
            gnss_normalizer,
            history_steps=args.forecast_start,
            forecast_horizon=args.forecast_horizon,
        )
        if test_ids
        else None
    )

    device = choose_device(args.device)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
        if test_ds
        else None
    )

    train_mean = compute_package_train_mean_slip(args.package_dir, train_ids)
    val_baselines = evaluate_package_physical_baselines(
        args.package_dir,
        val_ids,
        train_mean,
        history_steps=args.forecast_start,
        horizons=(1, 5, 10, 30, 50),
    )
    write_metrics(output_dir, "val_baseline", val_baselines)
    if test_ids:
        test_baselines = evaluate_package_physical_baselines(
            args.package_dir,
            test_ids,
            train_mean,
            history_steps=args.forecast_start,
            horizons=(1, 5, 10, 30, 50),
        )
        write_metrics(output_dir, "test_baseline", test_baselines)
    else:
        test_baselines = {}

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    writer = maybe_make_writer(output_dir, args.tensorboard_dir)
    print(
        f"device={device}, amp={use_amp}, model={args.model_type}, "
        f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}",
        flush=True,
    )

    history = []
    best_val_rmse = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in train_loader:
            history_slip = batch["history_slip"].to(device)
            history_gnss = batch["history_gnss"].to(device)
            target = batch["future_slip"].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(history_slip, history_gnss)
                loss, loss_parts = forecast_loss(
                    pred,
                    target,
                    active_weight=args.active_weight,
                    m0_loss_weight=args.m0_loss_weight,
                    slip_scale=slip_transform.scale,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())
            total_batches += 1

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            train_metrics = evaluate_model(model, train_loader, slip_transform, device)
            val_metrics = evaluate_model(model, val_loader, slip_transform, device)
            row = {
                "epoch": float(epoch),
                "train_loss": total_loss / max(total_batches, 1),
                "train_rmse": train_metrics["physical_rmse"],
                "val_rmse": val_metrics["physical_rmse"],
                "val_r2": val_metrics["physical_r2"],
                "val_m0_rel_abs": val_metrics["m0_rel_abs"],
            }
            history.append(row)
            if writer is not None:
                writer.add_scalar("loss/train_total", row["train_loss"], epoch)
                writer.add_scalar("metrics/train_rmse", row["train_rmse"], epoch)
                writer.add_scalar("metrics/val_rmse", row["val_rmse"], epoch)
                writer.add_scalar("metrics/val_r2", row["val_r2"], epoch)
                writer.add_scalar("metrics/val_m0_rel_abs", row["val_m0_rel_abs"], epoch)
            print(
                f"epoch={epoch:04d} loss={row['train_loss']:.6g} "
                f"train_rmse={row['train_rmse']:.6g} val_rmse={row['val_rmse']:.6g} "
                f"val_r2={row['val_r2']:.6g}",
                flush=True,
            )
            if val_metrics["physical_rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["physical_rmse"]
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_train = evaluate_model(model, train_loader, slip_transform, device)
    final_val = evaluate_model(model, val_loader, slip_transform, device)
    final_test = evaluate_model(model, test_loader, slip_transform, device) if test_loader else None

    h = str(args.forecast_horizon)
    status = "PASS_BASELINE" if final_val["physical_rmse"] < val_baselines[h]["rmse_persistence"] else "BELOW_BASELINE"
    payload: dict[str, object] = {
        "protocol": args.protocol,
        "status": status,
        "train_event_count": len(train_ids),
        "val_event_count": len(val_ids),
        "test_event_count": len(test_ids),
        "forecast_start": args.forecast_start,
        "forecast_horizon": args.forecast_horizon,
        "model_type": args.model_type,
        "final_train": final_train,
        "final_val": final_val,
        "final_test": final_test,
        "baseline_val_h50": val_baselines[h],
        "baseline_test_h50": test_baselines.get(h),
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_history(output_dir / "training_history.csv", history)
    write_report(output_dir, payload)
    artifact = {"model_state_dict": model.state_dict(), "args": vars(args), "metrics": payload}
    torch.save(artifact, output_dir / "model.pt")
    if writer is not None:
        writer.close()
    pai_output_model = os.environ.get("PAI_OUTPUT_MODEL")
    if pai_output_model:
        pai_out = Path(pai_output_model)
        pai_out.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, pai_out / "model.pt")
        shutil.copy2(output_dir / "metrics.json", pai_out / "metrics.json")
        shutil.copy2(output_dir / "forecast_training_report.md", pai_out / "forecast_training_report.md")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
