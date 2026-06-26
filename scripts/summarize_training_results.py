# -*- coding: utf-8 -*-
"""Summarize SSE training metrics into a compact readable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        return f"{float(value):.{digits}g}"
    except Exception:
        return str(value)


def _fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return str(int(round(float(value))))
    except Exception:
        return str(value)


def _line_for_split(name: str, metrics: dict[str, Any]) -> list[str]:
    final = metrics.get("final_test") or metrics.get("final_val") or {}
    baseline = metrics.get("baseline_test_h50") or metrics.get("baseline_val_h50") or {}
    improvement = metrics.get("test_rmse_improvement_pct", metrics.get("val_rmse_improvement_pct"))
    m0_change = metrics.get("test_m0_change_pct")
    return [
        f"## {name}",
        "",
        f"- Status: `{metrics.get('status', 'n/a')}`",
        f"- Publication gate: `{metrics.get('publication_gate', 'n/a')}`",
        f"- Model type: `{metrics.get('model_type', 'n/a')}`",
        f"- Events: train={metrics.get('train_event_count', 'n/a')}, val={metrics.get('val_event_count', 'n/a')}, test={metrics.get('test_event_count', 'n/a')}",
        f"- Model h50 RMSE: `{_fmt(final.get('physical_rmse'))}`",
        f"- Persistence h50 RMSE: `{_fmt(baseline.get('rmse_persistence'))}`",
        f"- Mean h50 RMSE: `{_fmt(baseline.get('rmse_mean'))}`",
        f"- RMSE improvement vs persistence: `{_fmt(improvement)}%`",
        f"- Model h50 R2: `{_fmt(final.get('physical_r2'))}`",
        f"- Model M0 rel abs: `{_fmt(final.get('m0_rel_abs'))}`",
        f"- M0 change vs persistence: `{_fmt(m0_change)}%`",
        "",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize SSE training results.")
    parser.add_argument("--small-overfit-dir", default="/mnt/data/sse_outputs/small_overfit")
    parser.add_argument("--training-dir", default="/mnt/data/sse_outputs/forecast_training_results")
    parser.add_argument("--output", default="/mnt/data/sse_outputs/training_summary.md")
    args = parser.parse_args()

    lines = ["# SSE Training Summary", ""]
    small = _read_json(Path(args.small_overfit_dir) / "metrics.json")
    if small:
        lines.extend(
            [
                "## Small Overfit",
                "",
                f"- Success: `{_fmt(small.get('success'))}`",
                f"- Model type: `{small.get('model_type', 'n/a')}`",
                f"- Event count: `{_fmt_int(small.get('event_count'))}`",
                f"- Model h50 RMSE: `{_fmt(small.get('physical_rmse'))}`",
                f"- Persistence h50 RMSE: `{_fmt(small.get('baseline_h50_persistence_rmse'))}`",
                f"- RMSE / persistence: `{_fmt(small.get('rmse_vs_persistence_ratio'))}`",
                f"- R2: `{_fmt(small.get('physical_r2'))}`",
                f"- M0 rel abs: `{_fmt(small.get('m0_rel_abs'))}`",
                "",
            ]
        )
    else:
        lines.extend(["## Small Overfit", "", "- No metrics found.", ""])

    for split in ("random", "blocked"):
        metrics = _read_json(Path(args.training_dir) / split / "metrics.json")
        if metrics:
            lines.extend(_line_for_split(split, metrics))
        else:
            lines.extend([f"## {split}", "", "- No metrics found.", ""])

    report = "\n".join(lines).rstrip() + "\n"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
