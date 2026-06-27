# -*- coding: utf-8 -*-
"""Build paper-ready Markdown tables from an SSE experiment matrix summary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _fmt(value: str, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return value


def _pct(value: str) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return value


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _get(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(names[0])


def make_main_table(rows: list[dict[str, str]]) -> str:
    selected = [r for r in rows if _get(r, "run", "run_name") == "main_residual_full"]
    selected.sort(key=lambda r: _get(r, "split", "protocol"))
    lines = [
        "## Main Full-Data Results",
        "",
        "| Split | Model | h50 RMSE | Persistence | Gain | R2 | M0 rel abs | Gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in selected:
        lines.append(
            "| {split} | {model} | {rmse} | {persistence} | {gain} | {r2} | {m0} | {gate} |".format(
                split=_get(r, "split", "protocol"),
                model=r["model_type"],
                rmse=_fmt(r["test_rmse"]),
                persistence=_fmt(_get(r, "persistence_rmse", "test_persistence_rmse")),
                gain=_pct(r["test_rmse_improvement_pct"]),
                r2=_fmt(r["test_r2"]),
                m0=_fmt(r["test_m0_rel_abs"]),
                gate=r["publication_gate"],
            )
        )
    return "\n".join(lines)


def make_ablation_table(rows: list[dict[str, str]]) -> str:
    selected = [r for r in rows if _get(r, "run", "run_name") != "main_residual_full"]
    selected.sort(key=lambda r: (_get(r, "run", "run_name"), _get(r, "split", "protocol")))
    lines = [
        "## Ablation Results",
        "",
        "| Run | Split | Model | Input | M0 loss | h50 RMSE | Gain | R2 | M0 rel abs | Gate |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in selected:
        lines.append(
            "| {run} | {split} | {model} | {input_mode} | {m0_loss} | {rmse} | {gain} | {r2} | {m0} | {gate} |".format(
                run=_get(r, "run", "run_name"),
                split=_get(r, "split", "protocol"),
                model=r["model_type"],
                input_mode=r["input_mode"],
                m0_loss=_fmt(r["m0_loss_weight"], 3),
                rmse=_fmt(r["test_rmse"]),
                gain=_pct(r["test_rmse_improvement_pct"]),
                r2=_fmt(r["test_r2"]),
                m0=_fmt(r["test_m0_rel_abs"]),
                gate=r["publication_gate"],
            )
        )
    return "\n".join(lines)


def make_notes(rows: list[dict[str, str]]) -> str:
    lines = ["## Auto Notes", ""]
    main = [r for r in rows if _get(r, "run", "run_name") == "main_residual_full"]
    if main:
        best = min(main, key=lambda r: float(r["test_rmse"]))
        lines.append(
            f"- Main model best split result is `{_get(best, 'split', 'protocol')}` with h50 RMSE `{_fmt(best['test_rmse'])}`."
        )
    ablations = [r for r in rows if _get(r, "run", "run_name") != "main_residual_full"]
    if ablations:
        best_ablation = min(ablations, key=lambda r: float(r["test_rmse"]))
        lines.append(
            "- Best completed ablation so far is "
            f"`{_get(best_ablation, 'run', 'run_name')}/{_get(best_ablation, 'split', 'protocol')}` "
            f"with h50 RMSE `{_fmt(best_ablation['test_rmse'])}`."
        )
    lines.append("- Treat this file as generated evidence; interpretive claims should still be checked against the full experiment context.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper-ready result tables from experiment_matrix_summary.csv")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    rows = read_rows(Path(args.summary_csv))
    out = "\n\n".join([make_main_table(rows), make_ablation_table(rows), make_notes(rows)]) + "\n"
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(out, encoding="utf-8")
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
