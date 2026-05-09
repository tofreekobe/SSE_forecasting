# -*- coding: utf-8 -*-
"""Run SSE future-slip forecasting data contract and hard baselines."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.forecast_contract import DEFAULT_FORECAST_START, build_forecast_contract
from src.dataset.package_forecast_contract import build_package_forecast_contract
from src.models.forecast_baselines import (
    DEFAULT_HORIZONS,
    compute_package_train_mean_slip,
    compute_train_mean_slip,
    evaluate_package_physical_baselines,
    evaluate_physical_baselines,
    write_metrics,
)


def run_for_protocol(args, protocol: str) -> None:
    if args.package_dir:
        splits, _, _, stats = build_package_forecast_contract(
            package_dir=args.package_dir,
            protocol=protocol,
            seed=args.seed,
            history_ratio=args.history_ratio,
            forecast_horizon=max(args.horizons),
            forecast_start=args.forecast_start,
        )
        train_mean = compute_package_train_mean_slip(args.package_dir, splits.train)
        evaluator = evaluate_package_physical_baselines
        source = args.package_dir
    else:
        splits, _, _, stats = build_forecast_contract(
            data_dir=args.data_dir,
            protocol=protocol,
            seed=args.seed,
            history_ratio=args.history_ratio,
            forecast_horizon=max(args.horizons),
            forecast_start=args.forecast_start,
        )
        train_mean = compute_train_mean_slip(args.data_dir, splits.train)
        evaluator = evaluate_physical_baselines
        source = args.data_dir

    out_dir = Path(args.output_dir) / protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    stats.to_json(out_dir / "forecast_contract_stats.json")

    val_metrics = evaluator(
        source,
        splits.val,
        train_mean,
        history_steps=stats.history_steps,
        horizons=args.horizons,
    )
    test_metrics = evaluator(
        source,
        splits.test,
        train_mean,
        history_steps=stats.history_steps,
        horizons=args.horizons,
    )
    write_metrics(out_dir, "val", val_metrics)
    write_metrics(out_dir, "test", test_metrics)

    h = str(max(args.horizons))
    print(
        f"[{protocol}] h={h} test persistence RMSE={test_metrics[h]['rmse_persistence']:.6g}, "
        f"R2={test_metrics[h]['r2_persistence']:.6g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate hard SSE forecasting baselines.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--package-dir", default=None, help="Optional compressed HF dataset package directory.")
    parser.add_argument("--output-dir", default="baseline_results")
    parser.add_argument("--split", choices=["random", "blocked", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-ratio", type=float, default=0.7)
    parser.add_argument(
        "--forecast-start",
        type=int,
        default=DEFAULT_FORECAST_START,
        help="Forecast start timestep. Defaults to the active SSE onset region; set empty only by editing code.",
    )
    parser.add_argument("--horizons", nargs="*", type=int, default=list(DEFAULT_HORIZONS))
    args = parser.parse_args()

    protocols = ["random", "blocked"] if args.split == "both" else [args.split]
    for protocol in protocols:
        run_for_protocol(args, protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
