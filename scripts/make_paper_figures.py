# -*- coding: utf-8 -*-
"""Create paper-facing summary figures from SSE training outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_history(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items() if value != ""})
    return rows


def _save_method_overview(output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    path = output_dir / "fig_method_overview.png"
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)

    boxes = [
        (0.4, 3.15, 2.2, 1.15, "history GNSS\n60 x 9"),
        (0.4, 1.15, 2.2, 1.15, "history slip\n60 x 3030"),
        (3.4, 2.15, 2.25, 1.35, "global stats\nlog1p slip\nGNSS norm"),
        (6.4, 3.1, 2.25, 1.2, "segment 1 CNN\n15 x 166"),
        (6.4, 0.95, 2.25, 1.2, "segment 2 CNN\n15 x 36"),
        (9.55, 2.05, 2.1, 1.35, "future slip\n50 x 3030"),
    ]
    colors = ["#c8d8ea", "#d9e7c8", "#f0dfbd", "#d6d0ea", "#d6d0ea", "#c9e3df"]
    for (x, y, w, h, text), color in zip(boxes, colors):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="#30343b", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    arrows = [
        ((2.65, 3.72), (3.35, 2.95)),
        ((2.65, 1.72), (3.35, 2.65)),
        ((5.7, 2.95), (6.35, 3.7)),
        ((5.7, 2.55), (6.35, 1.55)),
        ((8.7, 3.7), (9.5, 2.9)),
        ((8.7, 1.55), (9.5, 2.35)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=15, linewidth=1.2, color="#30343b"))

    ax.text(6.4, 4.55, "Two disconnected subfaults are modeled separately", fontsize=12, weight="bold")
    ax.text(0.4, 0.35, "Primary claim: 50-step synthetic SSE slip-field forecasting with physical-unit baselines", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_performance_summary(training_dir: Path, output_dir: Path) -> Path | None:
    metrics_by_split = {split: _read_json(training_dir / split / "metrics.json") for split in ("random", "blocked")}
    metrics_by_split = {k: v for k, v in metrics_by_split.items() if v}
    if not metrics_by_split:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = output_dir / "fig_performance_summary.png"
    splits = list(metrics_by_split)
    x = np.arange(len(splits))
    width = 0.24
    model_rmse = []
    persistence_rmse = []
    mean_rmse = []
    gains = []
    m0_change = []
    for split in splits:
        metrics = metrics_by_split[split]
        final = metrics.get("final_test") or metrics.get("final_val") or {}
        baseline = metrics.get("baseline_test_h50") or metrics.get("baseline_val_h50") or {}
        model_rmse.append(float(final.get("physical_rmse", np.nan)))
        persistence_rmse.append(float(baseline.get("rmse_persistence", np.nan)))
        mean_rmse.append(float(baseline.get("rmse_mean", np.nan)))
        gains.append(float(metrics.get("test_rmse_improvement_pct", metrics.get("val_rmse_improvement_pct", np.nan))))
        m0_change.append(float(metrics.get("test_m0_change_pct", np.nan)))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].bar(x - width, model_rmse, width, label="model")
    axes[0].bar(x, persistence_rmse, width, label="persistence")
    axes[0].bar(x + width, mean_rmse, width, label="mean")
    axes[0].set_xticks(x, splits)
    axes[0].set_title("h50 RMSE")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x, gains, color="#4b8f8c")
    axes[1].axhline(5.0, color="#7f1d1d", linestyle="--", linewidth=1, label="random gate")
    axes[1].axhline(2.0, color="#a16207", linestyle=":", linewidth=1.4, label="blocked gate")
    axes[1].set_xticks(x, splits)
    axes[1].set_title("RMSE improvement vs persistence (%)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    axes[2].bar(x, m0_change, color="#6f6aa8")
    axes[2].axhline(10.0, color="#7f1d1d", linestyle="--", linewidth=1, label="max allowed worsening")
    axes[2].set_xticks(x, splits)
    axes[2].set_title("M0 error change vs persistence (%)")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_training_curves(training_dir: Path, output_dir: Path) -> Path | None:
    histories = {split: _read_history(training_dir / split / "training_history.csv") for split in ("random", "blocked")}
    histories = {k: v for k, v in histories.items() if v}
    if not histories:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = output_dir / "fig_training_curves_combined.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for split, rows in histories.items():
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(epochs, [row["train_loss"] for row in rows], label=f"{split} train loss")
        axes[1].plot(epochs, [row["val_rmse"] for row in rows], label=f"{split} val rmse")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_title("Optimization")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Validation physical RMSE")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_event_montage(training_dir: Path, output_dir: Path) -> Path | None:
    image_paths = []
    for split in ("random", "blocked"):
        figures_dir = training_dir / split / "figures"
        image_paths.extend(sorted(figures_dir.glob("forecast_event_*.png"))[:2])
    if not image_paths:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    path = output_dir / "fig_event_montage.png"
    cols = min(2, len(image_paths))
    rows = int(np.ceil(len(image_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 4.3 * rows))
    axes_arr = np.array(axes).reshape(-1)
    for ax, image_path in zip(axes_arr, image_paths):
        ax.imshow(mpimg.imread(image_path))
        ax.set_title(image_path.parent.parent.name + " / " + image_path.stem)
        ax.set_axis_off()
    for ax in axes_arr[len(image_paths) :]:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_image_prompts(output_dir: Path) -> Path:
    path = output_dir / "chatgpt_image_prompts.md"
    path.write_text(
        "\n".join(
            [
                "# Optional Figure Polishing Prompts",
                "",
                "Use these only for visual polishing. Keep all numeric metrics and axes from the generated figures.",
                "",
                "## Concept Figure",
                "",
                "Create a clean scientific workflow diagram for synthetic slow slip event forecasting: sparse GNSS history and two disconnected fault-slip history grids feed a global normalization block, then two separate segment-aware residual CNN branches, then a 50-step future slip-field output. Use restrained scientific colors and no decorative background.",
                "",
                "## Results Figure",
                "",
                "Create a journal-style multi-panel layout that preserves these panels: h50 RMSE model vs persistence vs mean, RMSE improvement percent with random and blocked gates, M0 relative error change, validation RMSE curves, and event-level final slip maps. Use clear panel labels and leave room for captions.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper-facing SSE forecast figures.")
    parser.add_argument("--training-dir", default="forecast_training_5070ti_lite")
    parser.add_argument("--output-dir", default="paper_figures")
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[str] = []
    for maybe_path in (
        _save_method_overview(output_dir),
        _save_performance_summary(training_dir, output_dir),
        _save_training_curves(training_dir, output_dir),
        _save_event_montage(training_dir, output_dir),
        _write_image_prompts(output_dir),
    ):
        if maybe_path is not None:
            outputs.append(str(maybe_path))

    manifest = {"training_dir": str(training_dir), "output_dir": str(output_dir), "figures": outputs}
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
