# -*- coding: utf-8 -*-
"""Collect experiment-matrix metrics into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "run_name",
    "protocol",
    "model_type",
    "input_mode",
    "epochs",
    "m0_loss_weight",
    "train_event_count",
    "val_event_count",
    "test_event_count",
    "test_rmse",
    "test_persistence_rmse",
    "test_mean_rmse",
    "test_rmse_improvement_pct",
    "test_r2",
    "test_m0_rel_abs",
    "test_m0_change_pct",
    "publication_gate",
    "metrics_path",
]


def _get(payload: dict[str, Any], key: str, default: Any = "") -> Any:
    value = payload.get(key, default)
    return default if value is None else value


def _round(value: Any, digits: int = 6) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def row_from_metrics(metrics_path: Path, matrix_dir: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    protocol_dir = metrics_path.parent
    run_dir = protocol_dir.parent
    final_test = payload.get("final_test") or {}
    baseline_test = payload.get("baseline_test_h50") or {}
    row = {
        "run_name": run_dir.name,
        "protocol": _get(payload, "protocol", protocol_dir.name),
        "model_type": _get(payload, "model_type"),
        "input_mode": _get(payload, "input_mode", "full"),
        "epochs": _get(payload, "epochs"),
        "m0_loss_weight": _get(payload, "m0_loss_weight"),
        "train_event_count": _get(payload, "train_event_count"),
        "val_event_count": _get(payload, "val_event_count"),
        "test_event_count": _get(payload, "test_event_count"),
        "test_rmse": _round(final_test.get("physical_rmse", "")),
        "test_persistence_rmse": _round(baseline_test.get("rmse_persistence", "")),
        "test_mean_rmse": _round(baseline_test.get("rmse_mean", "")),
        "test_rmse_improvement_pct": _round(_get(payload, "test_rmse_improvement_pct"), 3),
        "test_r2": _round(final_test.get("physical_r2", "")),
        "test_m0_rel_abs": _round(final_test.get("m0_rel_abs", "")),
        "test_m0_change_pct": _round(_get(payload, "test_m0_change_pct"), 3),
        "publication_gate": _get(payload, "publication_gate"),
        "metrics_path": str(metrics_path.relative_to(matrix_dir)),
    }
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], matrix_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SSE Experiment Matrix Summary",
        "",
        f"- Matrix directory: `{matrix_dir}`",
        f"- Completed runs: `{len(rows)}`",
        "",
        "| Run | Split | Model | Input | M0 loss | h50 RMSE | Persistence | Gain | R2 | M0 rel abs | Gate |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {protocol} | {model_type} | {input_mode} | {m0_loss_weight} | "
            "{test_rmse} | {test_persistence_rmse} | {test_rmse_improvement_pct}% | "
            "{test_r2} | {test_m0_rel_abs} | {publication_gate} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect SSE experiment matrix metrics.")
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    matrix_dir = Path(args.matrix_dir)
    rows = [
        row_from_metrics(path, matrix_dir)
        for path in sorted(matrix_dir.glob("*/**/metrics.json"))
        if path.parent.name in {"random", "blocked"}
    ]
    rows.sort(key=lambda item: (str(item["run_name"]), str(item["protocol"])))

    output_csv = Path(args.output_csv) if args.output_csv else matrix_dir / "experiment_matrix_summary.csv"
    output_md = Path(args.output_md) if args.output_md else matrix_dir / "experiment_matrix_summary.md"
    write_csv(output_csv, rows)
    write_markdown(output_md, rows, matrix_dir)
    print(json.dumps({"rows": len(rows), "csv": str(output_csv), "markdown": str(output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
