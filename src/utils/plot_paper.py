# -*- coding: utf-8 -*-
"""Publication-style plotting helpers for SSE diagnostics and reports.

The functions are intentionally lightweight and robust: they accept numpy
arrays, PyTorch tensors, lists, or None, and always write both PNG and PDF.
They do not fabricate performance claims; placeholder panels are labeled as
missing data when real values are not provided.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _to_numpy(x: Any) -> np.ndarray:
    """Convert common tensor/array/list inputs to a detached numpy array."""
    if x is None:
        return np.asarray([])
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    return np.asarray(x)


def _output_path(output_dir: str | os.PathLike[str] | None, out_path: str, default_name: str) -> Path:
    if output_dir is not None:
        path = Path(output_dir) / default_name
    else:
        path = Path(out_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig: plt.Figure, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return str(path)


def _as_slip_map(slip: np.ndarray, index: int = 0) -> np.ndarray:
    arr = _to_numpy(slip)
    if arr.size == 0:
        return np.zeros((15, 202), dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        return arr[min(index, arr.shape[0] - 1)]
    if arr.ndim == 2 and arr.shape[-1] == 3030:
        return arr[min(index, arr.shape[0] - 1)].reshape(15, 202)
    if arr.ndim == 1 and arr.size == 3030:
        return arr.reshape(15, 202)
    if arr.ndim == 2:
        return arr
    return np.squeeze(arr)


def plot_data_characteristics(
    gnss_data,
    slip_data,
    seg1_width: int = 166,
    output_dir: str | os.PathLike[str] | None = None,
    out_path: str = "fig_A_data_chars.png",
) -> str:
    path = _output_path(output_dir, out_path, "fig_A_data_chars.png")
    gnss = _to_numpy(gnss_data)
    slip = _to_numpy(slip_data)
    if gnss.ndim == 3:
        gnss = gnss[0]
    if gnss.size == 0:
        gnss = np.zeros((1, 9), dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    labels = ["S1-E", "S1-N", "S1-U", "S2-E", "S2-N", "S2-U", "S3-E", "S3-N", "S3-U"]
    for ch in range(min(gnss.shape[1], 9)):
        axes[0].plot(gnss[:, ch] + ch * 0.02, label=labels[ch], linewidth=0.8)
    if slip.size:
        flat = slip.reshape(slip.shape[0], -1) if slip.ndim >= 2 else slip.reshape(1, -1)
        active = flat.max(axis=1) > 1e-4
        for i, flag in enumerate(active[: gnss.shape[0]]):
            if flag:
                axes[0].axvspan(i - 0.5, i + 0.5, color="orange", alpha=0.25)
        active_idx = int(np.argmax(active)) if np.any(active) else min(len(active) // 2, len(active) - 1)
    else:
        active_idx = 0
    axes[0].set_title("GNSS traces and active slip windows")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Offset amplitude")

    slip_map = _as_slip_map(slip, active_idx)
    im = axes[1].imshow(slip_map, cmap="viridis", aspect="auto")
    axes[1].axvline(x=seg1_width, color="white", linestyle="--", linewidth=2)
    axes[1].set_title(f"Fault grid stitching (t={active_idx})")
    fig.colorbar(im, ax=axes[1])
    return _save(fig, path)


def plot_training_dynamics(
    log_csv_path: str | os.PathLike[str] | None = None,
    physics_weight_matrix=None,
    stage1_end: int | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    log_dir: str = "logs/stcr_net_training",
    ckpt_path: str | None = None,
    out_path: str = "fig_B_training.png",
) -> str:
    path = _output_path(output_dir, out_path, "fig_B_training.png")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    csv_path = Path(log_csv_path) if log_csv_path else None
    if csv_path is None or not csv_path.exists():
        matches = glob.glob(str(PROJECT_ROOT / log_dir / "**" / "metrics.csv"), recursive=True)
        csv_path = Path(matches[-1]) if matches else None

    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        for col, label in [
            ("train/loss_op", "L_operator"),
            ("train/loss_wmse", "L_wmse"),
            ("val_loss", "val_loss"),
        ]:
            if col in df.columns:
                sub = df.dropna(subset=[col])
                if not sub.empty:
                    axes[0].plot(sub.get("epoch", sub.index), sub[col], label=label)
        if stage1_end is not None:
            axes[0].axvline(stage1_end, color="black", linestyle="--", linewidth=1)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No training CSV provided", ha="center", va="center")
    axes[0].set_title("Training dynamics")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    weight = _to_numpy(physics_weight_matrix)
    if weight.size == 0 and ckpt_path:
        try:
            import torch

            ckpt = torch.load(PROJECT_ROOT / ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt.get("model_state_dict", {}))
            for key, value in state.items():
                if "physics_layer.greens_function.weight" in key:
                    weight = _to_numpy(value)
                    break
        except Exception:
            weight = np.asarray([])
    if weight.size:
        if weight.ndim > 2:
            weight = np.squeeze(weight)
        shown = weight[:, :: max(1, weight.shape[1] // 300)] if weight.ndim == 2 else weight.reshape(1, -1)
        vmax = float(np.nanmax(np.abs(shown))) or 1.0
        im = axes[1].imshow(shown, cmap="RdBu_r", aspect="auto", norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
        fig.colorbar(im, ax=axes[1])
    else:
        axes[1].text(0.5, 0.5, "No physics matrix provided", ha="center", va="center")
    axes[1].set_title("Pseudo-Okada operator")
    return _save(fig, path)


def plot_inversion_results(
    true_slip,
    pred_slip,
    true_gnss=None,
    pred_gnss=None,
    output_dir: str | os.PathLike[str] | None = None,
    out_path: str = "fig_C_inversion.png",
) -> str:
    path = _output_path(output_dir, out_path, "fig_C_inversion.png")
    true_arr = _to_numpy(true_slip)
    pred_arr = _to_numpy(pred_slip)
    if true_arr.ndim == 4:
        true_arr = true_arr[0]
    if pred_arr.ndim == 4:
        pred_arr = pred_arr[0]
    t_len = max(true_arr.shape[0] if true_arr.ndim >= 3 else 1, 1)
    t_idx = [int(t_len * 0.1), int(t_len * 0.5), max(int(t_len * 0.9), 0)]

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, t in enumerate(t_idx):
        t = min(t, t_len - 1)
        true_map = _as_slip_map(true_arr, t)
        pred_map = _as_slip_map(pred_arr, t)
        residual = pred_map - true_map
        axes[0, i].imshow(true_map, aspect="auto", cmap="viridis")
        axes[0, i].set_title(f"True slip t={t}")
        axes[1, i].imshow(pred_map, aspect="auto", cmap="viridis")
        axes[1, i].set_title(f"Predicted slip t={t}")
        vmax = float(np.nanmax(np.abs(residual))) + 1e-8
        axes[2, i].imshow(residual, aspect="auto", cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
        axes[2, i].set_title(f"Residual t={t}")
    return _save(fig, path)


def plot_forecasting_results(
    history_slip,
    true_future,
    pred_future,
    output_dir: str | os.PathLike[str] | None = None,
    out_path: str = "fig_D_forecasting.png",
) -> str:
    path = _output_path(output_dir, out_path, "fig_D_forecasting.png")
    true_arr = _to_numpy(true_future)
    pred_arr = _to_numpy(pred_future)
    if true_arr.ndim == 4:
        true_arr = true_arr[0]
    if pred_arr.ndim == 4:
        pred_arr = pred_arr[0]
    n = max(pred_arr.shape[0] if pred_arr.ndim >= 3 else 1, 1)
    steps = [0, min(10, n - 1), min(20, n - 1)]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for i, step in enumerate(steps):
        pred_map = _as_slip_map(pred_arr, step)
        axes[i].imshow(pred_map, aspect="auto", cmap="viridis")
        axes[i].set_title(f"Predicted T+{step}")
    if true_arr.size and pred_arr.size:
        max_steps = min(true_arr.shape[0], pred_arr.shape[0])
        rmse = [
            float(np.sqrt(np.mean((pred_arr[i].reshape(-1) - true_arr[i].reshape(-1)) ** 2)))
            for i in range(max_steps)
        ]
        axes[3].plot(np.arange(max_steps), rmse)
        axes[3].set_title("Forecast RMSE")
        axes[3].set_xlabel("Horizon")
    else:
        axes[3].text(0.5, 0.5, "No true future provided", ha="center", va="center")
    return _save(fig, path)


def plot_ablation_and_moment(
    true_slip=None,
    pred_slip=None,
    ablation_results: dict[str, dict[str, float]] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    out_path: str = "fig_E_ablation.png",
) -> str:
    path = _output_path(output_dir, out_path, "fig_E_ablation.png")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    true_arr = _to_numpy(true_slip)
    pred_arr = _to_numpy(pred_slip)
    if true_arr.size and pred_arr.size:
        true_m0 = true_arr.reshape(true_arr.shape[0], -1).sum(axis=1)
        pred_m0 = pred_arr.reshape(pred_arr.shape[0], -1).sum(axis=1)
        axes[0].scatter(true_m0, pred_m0, alpha=0.5)
        lo = min(float(true_m0.min()), float(pred_m0.min()))
        hi = max(float(true_m0.max()), float(pred_m0.max()))
        axes[0].plot([lo, hi], [lo, hi], "k--")
    else:
        axes[0].text(0.5, 0.5, "No moment data provided", ha="center", va="center")
    axes[0].set_title("Seismic moment fidelity")
    axes[0].set_xlabel("True M0")
    axes[0].set_ylabel("Predicted M0")

    if ablation_results:
        labels = list(ablation_results.keys())
        inv = [ablation_results[k].get("inversion_R2", np.nan) for k in labels]
        fore = [ablation_results[k].get("forecasting_R2", np.nan) for k in labels]
        x = np.arange(len(labels))
        axes[1].bar(x - 0.2, inv, width=0.4, label="Inversion R2")
        axes[1].bar(x + 0.2, fore, width=0.4, label="Forecasting R2")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=20, ha="right")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No ablation results provided", ha="center", va="center")
    axes[1].set_title("Ablation results")
    return _save(fig, path)


if __name__ == "__main__":
    print("plot_paper.py provides plotting functions; call them from inference scripts.")
