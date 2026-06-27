# -*- coding: utf-8 -*-
"""HF-ready diagnostics for the SSE feasibility gate.

This module intentionally stays independent from the current training code. It
can read raw event txt files, build a compact Hugging Face dataset package, and
run the same diagnostics locally or inside a Hugging Face Job.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SLIP_COL_START = 1
SLIP_COL_END = 3031
GNSS_COL_START = 3031
GNSS_COL_END = 3040
EXPECTED_TIMESTEPS = 273
EXPECTED_COLS = 3040
EXPECTED_EVENTS = 6000
DEFAULT_HORIZONS = (1, 5, 10, 30, 50)
ACTIVE_THRESHOLDS = (1e-4, 1e-3, 1e-2)
SCHEMA_VERSION = "sse-hf-diagnostics-v1"


@dataclass(frozen=True)
class EventRecord:
    event_id: int
    path: Path


def parse_event_id(path: str | Path) -> int:
    match = re.search(r"fault_sse_catalog_(\d+)\.txt$", str(path).replace("\\", "/"))
    if not match:
        raise ValueError(f"Cannot parse event id from path: {path}")
    return int(match.group(1))


def scan_event_files(data_dir: str | Path) -> list[EventRecord]:
    root = Path(data_dir)
    files = sorted(root.glob("**/*.txt"), key=lambda p: (parse_event_id(p), str(p)))
    return [EventRecord(parse_event_id(path), path) for path in files]


def select_records(
    records: list[EventRecord],
    max_events: int | None = None,
    seed: int = 123,
    sample: bool = False,
) -> list[EventRecord]:
    if max_events is None or max_events >= len(records):
        return records
    if sample:
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(len(records)), max_events))
        return [records[i] for i in idx]
    return records[:max_events]


def load_raw_event(path: str | Path) -> np.ndarray:
    raw = np.loadtxt(path, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    return raw


def split_event(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return raw[:, SLIP_COL_START:SLIP_COL_END], raw[:, GNSS_COL_START:GNSS_COL_END]


def _safe_float(value: Any) -> float:
    out = float(value)
    if math.isnan(out) or math.isinf(out):
        return float("nan")
    return out


def summarize_event(record: EventRecord, raw: np.ndarray) -> dict[str, Any]:
    slip, gnss = split_event(raw)
    nan_count = int(np.isnan(raw).sum())
    inf_count = int(np.isinf(raw).sum())
    bad_reasons: list[str] = []
    if raw.shape[0] != EXPECTED_TIMESTEPS:
        bad_reasons.append(f"T={raw.shape[0]}")
    if raw.shape[1] != EXPECTED_COLS:
        bad_reasons.append(f"cols={raw.shape[1]}")
    if nan_count:
        bad_reasons.append(f"nan={nan_count}")
    if inf_count:
        bad_reasons.append(f"inf={inf_count}")
    if slip.size and np.nanmin(slip) < -1e-8:
        bad_reasons.append("negative_slip")

    row: dict[str, Any] = {
        "event_id": record.event_id,
        "source_path": str(record.path),
        "n_timesteps": int(raw.shape[0]),
        "n_cols": int(raw.shape[1]),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "bad_reason": ";".join(bad_reasons),
    }
    if slip.size:
        row.update(
            {
                "slip_min": _safe_float(np.nanmin(slip)),
                "slip_max": _safe_float(np.nanmax(slip)),
                "slip_mean": _safe_float(np.nanmean(slip)),
                "slip_std": _safe_float(np.nanstd(slip)),
                "slip_sum_total": _safe_float(np.nansum(slip)),
                "slip_positive_ratio": _safe_float(np.mean(slip > 0)),
            }
        )
        for threshold in ACTIVE_THRESHOLDS:
            key = f"slip_active_ratio_gt_{threshold:g}".replace("-", "m")
            row[key] = _safe_float(np.mean(slip > threshold))
    if gnss.size:
        row.update(
            {
                "gnss_min": _safe_float(np.nanmin(gnss)),
                "gnss_max": _safe_float(np.nanmax(gnss)),
                "gnss_mean": _safe_float(np.nanmean(gnss)),
                "gnss_std": _safe_float(np.nanstd(gnss)),
            }
        )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _try_write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)
        return True
    except Exception:
        return False


def build_hf_dataset_package(
    data_dir: str | Path,
    output_dir: str | Path,
    shard_size: int = 64,
    max_events: int | None = None,
    sample: bool = False,
    seed: int = 123,
    include_arrays: bool = True,
) -> dict[str, Any]:
    """Create a HF-uploadable package from raw SSE txt files."""

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "events").mkdir(parents=True, exist_ok=True)

    records = select_records(scan_event_files(data_dir), max_events, seed, sample)
    manifest_rows: list[dict[str, Any]] = []
    shard_slip: list[np.ndarray] = []
    shard_gnss: list[np.ndarray] = []
    shard_event_ids: list[int] = []
    shard_index = 0

    def flush_shard() -> None:
        nonlocal shard_index, shard_slip, shard_gnss, shard_event_ids
        if not shard_event_ids:
            return
        shard_name = f"shard_{shard_index:05d}.npz"
        shard_path = output_dir / "events" / shard_name
        np.savez_compressed(
            shard_path,
            event_id=np.asarray(shard_event_ids, dtype=np.int32),
            slip=np.stack(shard_slip, axis=0).astype(np.float32),
            gnss=np.stack(shard_gnss, axis=0).astype(np.float32),
        )
        for local_index, event_id in enumerate(shard_event_ids):
            for row in manifest_rows:
                if row["event_id"] == event_id:
                    row["shard_file"] = f"events/{shard_name}"
                    row["shard_index"] = local_index
                    break
        shard_index += 1
        shard_slip = []
        shard_gnss = []
        shard_event_ids = []

    for record in records:
        raw = load_raw_event(record.path)
        row = summarize_event(record, raw)
        row["shard_file"] = ""
        row["shard_index"] = -1
        manifest_rows.append(row)
        if include_arrays and not row["bad_reason"]:
            slip, gnss = split_event(raw)
            shard_slip.append(slip)
            shard_gnss.append(gnss)
            shard_event_ids.append(record.event_id)
            if len(shard_event_ids) >= shard_size:
                flush_shard()
    flush_shard()

    _write_csv(output_dir / "manifest.csv", manifest_rows)
    _write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    parquet_written = _try_write_parquet(output_dir / "manifest.parquet", manifest_rows)

    stats_src = data_dir / "normalization_stats.json"
    if stats_src.exists():
        shutil.copy2(stats_src, output_dir / "normalization_stats.json")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_data_dir": str(data_dir),
        "event_count": len(records),
        "expected_events": EXPECTED_EVENTS,
        "shard_size": shard_size,
        "include_arrays": include_arrays,
        "manifest_parquet_written": parquet_written,
        "active_thresholds": list(ACTIVE_THRESHOLDS),
        "forecast_horizons": list(DEFAULT_HORIZONS),
        "notes": [
            "Private HF dataset package for SSE feasibility diagnostics.",
            "Use scripts/run_hf_diagnostics.py to produce gate outputs.",
        ],
    }
    module_src = Path(__file__)
    shutil.copy2(module_src, output_dir / "hf_sse_diagnostics.py")
    (output_dir / "run_diagnostics.py").write_text(
        "import os\n"
        "from hf_sse_diagnostics import cli_diagnostics\n"
        "if __name__ == '__main__':\n"
        "    root = os.path.abspath(os.path.dirname(__file__))\n"
        "    raise SystemExit(cli_diagnostics(['--input-dir', root, '--output-dir', os.path.join(root, 'diagnostics')]))\n",
        encoding="utf-8",
    )
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(_dataset_card(metadata), encoding="utf-8")
    return metadata


def _dataset_card(metadata: dict[str, Any]) -> str:
    return (
        "---\n"
        "license: other\n"
        "task_categories:\n"
        "- time-series-forecasting\n"
        "tags:\n"
        "- slow-slip-events\n"
        "- geophysics\n"
        "- gnss\n"
        "- private-diagnostics\n"
        "---\n\n"
        "# SSE Slow Slip Private Diagnostics Dataset\n\n"
        "This private dataset package is generated for pre-refactor feasibility "
        "diagnostics. It contains a manifest plus compressed event shards with "
        "slip and GNSS arrays.\n\n"
        "Run diagnostics with:\n\n"
        "```bash\n"
        "python scripts/run_hf_diagnostics.py --input-dir . --output-dir diagnostics\n"
        "```\n\n"
        f"Schema version: `{metadata['schema_version']}`\n"
    )


def read_manifest(package_dir: str | Path) -> list[dict[str, str]]:
    path = Path(package_dir) / "manifest.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def iter_package_events(package_dir: str | Path) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    package_dir = Path(package_dir)
    seen: set[str] = set()
    for row in read_manifest(package_dir):
        shard_file = row.get("shard_file", "")
        if not shard_file or shard_file in seen:
            continue
        seen.add(shard_file)
        with np.load(package_dir / shard_file) as shard:
            event_ids = shard["event_id"]
            slips = shard["slip"]
            gnss = shard["gnss"]
            for i, event_id in enumerate(event_ids):
                yield int(event_id), slips[i], gnss[i]


def _init_horizon_stats(horizons: Iterable[int]) -> dict[int, dict[str, float]]:
    return {
        h: {
            "mse_zero_sum": 0.0,
            "mse_persistence_sum": 0.0,
            "mse_mean_sum": 0.0,
            "event_count": 0.0,
            "r2_sse": 0.0,
            "r2_sst": 0.0,
            "m0_persistence_rel_abs_sum": 0.0,
            "m0_zero_rel_abs_sum": 0.0,
        }
        for h in horizons
    }


def _corr_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _load_legacy_stats(package_dir: Path, explicit_path: str | Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.append(package_dir / "normalization_stats.json")
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                stats = json.load(f)
            mean = np.asarray(stats["slip_mean"], dtype=np.float32)
            std = np.maximum(np.asarray(stats["slip_std"], dtype=np.float32), 1e-8)
            return mean, std
    return None


def run_diagnostics(
    package_dir: str | Path,
    output_dir: str | Path,
    normalization_stats: str | Path | None = None,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    package_dir = Path(package_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = tuple(int(h) for h in horizons)

    manifest = read_manifest(package_dir)
    bad_rows = [row for row in manifest if row.get("bad_reason")]
    event_ids = [int(row["event_id"]) for row in manifest if row.get("event_id")]
    missing_ids: list[int] = []
    if event_ids:
        ids = set(event_ids)
        missing_ids = [i for i in range(min(ids), max(ids) + 1) if i not in ids]

    horizon_stats = _init_horizon_stats(horizons)
    slip_sum_values: list[np.ndarray] = []
    slip_max_values: list[np.ndarray] = []
    gnss_values: list[np.ndarray] = []
    future_gnss_corr: dict[int, list[float]] = {h: [] for h in horizons if h in (10, 30, 50)}
    summary_autocorr: dict[int, dict[str, list[float]]] = {
        h: {"sum": [], "max": []} for h in horizons
    }
    legacy = _load_legacy_stats(package_dir, normalization_stats)
    legacy_neg = []
    legacy_abs_active = []
    legacy_pos_active = []
    total_n = 0
    total_sum = 0.0
    total_sq = 0.0
    slip_min = float("inf")
    slip_max = -float("inf")
    pos_count = 0
    active_counts = {threshold: 0 for threshold in ACTIVE_THRESHOLDS}
    event_count_with_arrays = 0

    for _, slip, gnss in iter_package_events(package_dir):
        event_count_with_arrays += 1
        slip_min = min(slip_min, float(np.min(slip)))
        slip_max = max(slip_max, float(np.max(slip)))
        total_n += int(slip.size)
        total_sum += float(np.sum(slip))
        total_sq += float(np.sum(slip * slip))
        pos_count += int(np.sum(slip > 0))
        for threshold in ACTIVE_THRESHOLDS:
            active_counts[threshold] += int(np.sum(slip > threshold))

        sum_s = np.sum(slip, axis=1)
        max_s = np.max(slip, axis=1)
        slip_sum_values.append(sum_s)
        slip_max_values.append(max_s)
        gnss_values.append(gnss)

        event_mean = np.mean(slip, axis=0, keepdims=True)
        for h in horizons:
            if h >= slip.shape[0]:
                continue
            y = slip[h:]
            persistence = slip[:-h]
            zero = np.zeros_like(y)
            mean_pred = np.repeat(event_mean, y.shape[0], axis=0)
            horizon_stats[h]["mse_zero_sum"] += float(np.mean((y - zero) ** 2))
            horizon_stats[h]["mse_persistence_sum"] += float(np.mean((y - persistence) ** 2))
            horizon_stats[h]["mse_mean_sum"] += float(np.mean((y - mean_pred) ** 2))
            horizon_stats[h]["event_count"] += 1.0
            yt = y.reshape(-1)
            yp = persistence.reshape(-1)
            horizon_stats[h]["r2_sse"] += float(np.sum((yt - yp) ** 2))
            horizon_stats[h]["r2_sst"] += float(np.sum((yt - float(np.mean(yt))) ** 2))
            true_m0 = np.sum(y, axis=1)
            pred_m0 = np.sum(persistence, axis=1)
            zero_m0 = np.zeros_like(true_m0)
            denom = np.maximum(np.abs(true_m0), 1e-8)
            horizon_stats[h]["m0_persistence_rel_abs_sum"] += float(np.mean(np.abs(pred_m0 - true_m0) / denom))
            horizon_stats[h]["m0_zero_rel_abs_sum"] += float(np.mean(np.abs(zero_m0 - true_m0) / denom))
            summary_autocorr[h]["sum"].append(_corr_or_nan(sum_s[:-h], sum_s[h:]))
            summary_autocorr[h]["max"].append(_corr_or_nan(max_s[:-h], max_s[h:]))
            if h in future_gnss_corr:
                cors = [_corr_or_nan(gnss[:-h, j], sum_s[h:]) for j in range(gnss.shape[1])]
                valid = [abs(c) for c in cors if not math.isnan(c)]
                if valid:
                    future_gnss_corr[h].append(max(valid))

        if legacy is not None:
            mean, std = legacy
            z = (slip - mean) / std
            legacy_neg.append(float(np.mean(z < 0)))
            legacy_abs_active.append(float(np.mean(np.abs(z) > 1e-4)))
            legacy_pos_active.append(float(np.mean(z > 1e-4)))

    physical = {
        "event_count_with_arrays": event_count_with_arrays,
        "slip_min": slip_min if total_n else float("nan"),
        "slip_max": slip_max if total_n else float("nan"),
        "slip_mean": total_sum / total_n if total_n else float("nan"),
        "slip_std": math.sqrt(max(total_sq / total_n - (total_sum / total_n) ** 2, 0.0)) if total_n else float("nan"),
        "slip_positive_ratio": pos_count / total_n if total_n else float("nan"),
        "slip_active_ratios": {
            f"gt_{threshold:g}": active_counts[threshold] / total_n if total_n else float("nan")
            for threshold in ACTIVE_THRESHOLDS
        },
    }

    baseline_rows: list[dict[str, Any]] = []
    baseline_metrics: dict[str, Any] = {}
    for h in horizons:
        stats = horizon_stats[h]
        n = max(stats["event_count"], 1.0)
        rmse_zero = math.sqrt(stats["mse_zero_sum"] / n)
        rmse_persistence = math.sqrt(stats["mse_persistence_sum"] / n)
        rmse_mean = math.sqrt(stats["mse_mean_sum"] / n)
        r2 = 1.0 - stats["r2_sse"] / max(stats["r2_sst"], 1e-12)
        row = {
            "horizon": h,
            "event_count": int(stats["event_count"]),
            "rmse_zero": rmse_zero,
            "rmse_mean": rmse_mean,
            "rmse_persistence": rmse_persistence,
            "persistence_r2": r2,
            "m0_rel_abs_zero": stats["m0_zero_rel_abs_sum"] / n,
            "m0_rel_abs_persistence": stats["m0_persistence_rel_abs_sum"] / n,
            "sum_autocorr_median": _nanmedian(summary_autocorr[h]["sum"]),
            "max_autocorr_median": _nanmedian(summary_autocorr[h]["max"]),
        }
        baseline_rows.append(row)
        baseline_metrics[str(h)] = row
    _write_csv(output_dir / "baseline_metrics.csv", baseline_rows)

    coupling = {
        "future_sum_from_current_gnss_abs_corr_median": {
            str(h): _nanmedian(values) for h, values in future_gnss_corr.items()
        }
    }
    if slip_sum_values and gnss_values:
        sums = np.concatenate(slip_sum_values)
        maxes = np.concatenate(slip_max_values)
        gnss_all = np.concatenate(gnss_values, axis=0)
        coupling["slip_sum_to_gnss_corr"] = [_corr_or_nan(sums, gnss_all[:, j]) for j in range(gnss_all.shape[1])]
        coupling["slip_max_to_gnss_corr"] = [_corr_or_nan(maxes, gnss_all[:, j]) for j in range(gnss_all.shape[1])]

    legacy_conflict = {
        "stats_available": legacy is not None,
        "zscore_negative_ratio_mean": _nanmean(legacy_neg),
        "zscore_abs_active_ratio_mean": _nanmean(legacy_abs_active),
        "zscore_pos_active_ratio_mean": _nanmean(legacy_pos_active),
        "softplus_conflict": bool(legacy is not None and _nanmean(legacy_neg) > 0.05),
    }

    gate = evaluate_gates(manifest, bad_rows, missing_ids, baseline_metrics, legacy_conflict)
    results = {
        "schema_version": SCHEMA_VERSION,
        "package_dir": str(package_dir),
        "integrity": {
            "manifest_event_count": len(manifest),
            "expected_events": EXPECTED_EVENTS,
            "bad_event_count": len(bad_rows),
            "bad_event_ratio": len(bad_rows) / max(len(manifest), 1),
            "missing_event_count_in_observed_range": len(missing_ids),
            "missing_event_ids_preview": missing_ids[:50],
        },
        "physical": physical,
        "coupling": coupling,
        "baseline_metrics": baseline_metrics,
        "legacy_scheme_counterexample": legacy_conflict,
        "gate": gate,
    }
    (output_dir / "hf_diagnostics.json").write_text(
        json.dumps(_json_sanitize(results), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (output_dir / "hf_diagnostics.md").write_text(render_markdown(results), encoding="utf-8")
    write_plots(output_dir, baseline_rows, slip_sum_values, legacy_conflict)
    return results


def evaluate_gates(
    manifest: list[dict[str, str]],
    bad_rows: list[dict[str, str]],
    missing_ids: list[int],
    baseline_metrics: dict[str, Any],
    legacy_conflict: dict[str, Any],
) -> dict[str, Any]:
    gate1 = len(bad_rows) / max(len(manifest), 1) < 0.005 and len(missing_ids) == 0
    h50 = baseline_metrics.get("50", {})
    rmse_zero = float(h50.get("rmse_zero", float("inf")))
    rmse_persistence = float(h50.get("rmse_persistence", float("inf")))
    persistence_r2 = float(h50.get("persistence_r2", float("-inf")))
    gate2 = rmse_persistence < rmse_zero and persistence_r2 > 0
    gate3 = {
        "status": "PENDING_MODEL",
        "random_split_required_rmse_improvement": 0.05,
        "blocked_split_required_rmse_improvement": 0.02,
        "max_allowed_m0_error_worsening": 0.10,
    }
    if not gate1 or not gate2:
        conclusion = "NO_GO"
    elif legacy_conflict.get("softplus_conflict"):
        conclusion = "GO_WITH_CHANGES"
    else:
        conclusion = "GO_WITH_CHANGES"
    return {
        "gate1_data_valid": gate1,
        "gate2_task_learnable_proxy": gate2,
        "gate3_publication_model_gate": gate3,
        "conclusion": conclusion,
        "required_corrections": [
            "Do not z-score slip targets before a nonnegative softplus decoder.",
            "Do not fit the physics operator on per-event robust-scaled GNSS.",
            "Do not claim forecasting unless future_slip loss is trained and evaluated.",
            "Do not use mock predictions or hard-coded ablation scores in reports.",
        ],
    }


def render_markdown(results: dict[str, Any]) -> str:
    integrity = results["integrity"]
    physical = results["physical"]
    gate = results["gate"]
    legacy = results["legacy_scheme_counterexample"]
    lines = [
        "# HF SSE Diagnostics",
        "",
        f"Conclusion: `{gate['conclusion']}`",
        "",
        "## Integrity",
        "",
        f"- Events in manifest: {integrity['manifest_event_count']}",
        f"- Bad events: {integrity['bad_event_count']} ({integrity['bad_event_ratio']:.4%})",
        f"- Missing IDs in observed range: {integrity['missing_event_count_in_observed_range']}",
        "",
        "## Physical Scale",
        "",
        f"- Slip min/max: {physical['slip_min']:.6g} / {physical['slip_max']:.6g}",
        f"- Slip mean/std: {physical['slip_mean']:.6g} / {physical['slip_std']:.6g}",
        f"- Positive slip ratio: {physical['slip_positive_ratio']:.4%}",
        f"- Active ratios: {json.dumps(physical['slip_active_ratios'], sort_keys=True)}",
        "",
        "## Baselines",
        "",
        "| Horizon | RMSE zero | RMSE mean | RMSE persistence | Persistence R2 | M0 rel abs persistence |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for h, row in results["baseline_metrics"].items():
        lines.append(
            f"| {h} | {row['rmse_zero']:.6g} | {row['rmse_mean']:.6g} | "
            f"{row['rmse_persistence']:.6g} | {row['persistence_r2']:.6g} | "
            f"{row['m0_rel_abs_persistence']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Legacy Scheme Counterexample",
            "",
            f"- Stats available: {legacy['stats_available']}",
            f"- Z-score negative target ratio: {legacy['zscore_negative_ratio_mean']:.4%}",
            f"- Z-score abs active ratio: {legacy['zscore_abs_active_ratio_mean']:.4%}",
            f"- Softplus conflict: {legacy['softplus_conflict']}",
            "",
            "## Gates",
            "",
            f"- Gate 1 data valid: {gate['gate1_data_valid']}",
            f"- Gate 2 task learnable proxy: {gate['gate2_task_learnable_proxy']}",
            f"- Gate 3 publication model gate: {gate['gate3_publication_model_gate']['status']}",
            "",
            "## Required Corrections",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in gate["required_corrections"]])
    lines.append("")
    return "\n".join(lines)


def write_plots(
    output_dir: Path,
    baseline_rows: list[dict[str, Any]],
    slip_sum_values: list[np.ndarray],
    legacy_conflict: dict[str, Any],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if baseline_rows:
        horizons = [row["horizon"] for row in baseline_rows]
        plt.figure(figsize=(7, 4))
        plt.plot(horizons, [row["rmse_zero"] for row in baseline_rows], label="zero")
        plt.plot(horizons, [row["rmse_mean"] for row in baseline_rows], label="mean")
        plt.plot(horizons, [row["rmse_persistence"] for row in baseline_rows], label="persistence")
        plt.xlabel("Forecast horizon")
        plt.ylabel("RMSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "baseline_rmse.png", dpi=160)
        plt.close()

    if slip_sum_values:
        sums = np.concatenate(slip_sum_values)
        plt.figure(figsize=(7, 4))
        plt.hist(sums, bins=80)
        plt.xlabel("Per-timestep total slip")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "slip_sum_distribution.png", dpi=160)
        plt.close()

    if legacy_conflict.get("stats_available"):
        labels = ["z<0", "abs active", "pos active"]
        values = [
            legacy_conflict["zscore_negative_ratio_mean"],
            legacy_conflict["zscore_abs_active_ratio_mean"],
            legacy_conflict["zscore_pos_active_ratio_mean"],
        ]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, values)
        plt.ylim(0, 1)
        plt.ylabel("Ratio")
        plt.tight_layout()
        plt.savefig(output_dir / "legacy_zscore_conflict.png", dpi=160)
        plt.close()


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _nanmedian(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def snapshot_hf_repo(repo_id: str, revision: str | None = None, token: str | None = None) -> str:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("Install huggingface_hub to read a HF dataset repo") from exc
    return snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision, token=token)


def cli_prepare(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a HF-uploadable SSE diagnostics dataset package.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="hf_dataset_package")
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args(argv)
    metadata = build_hf_dataset_package(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        max_events=args.max_events,
        sample=args.sample,
        seed=args.seed,
        include_arrays=not args.manifest_only,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=True))
    return 0


def cli_diagnostics(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SSE feasibility diagnostics locally or from a HF dataset repo.")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--output-dir", default="diagnostics/hf")
    parser.add_argument("--normalization-stats", default=None)
    parser.add_argument("--horizons", nargs="*", type=int, default=list(DEFAULT_HORIZONS))
    args = parser.parse_args(argv)
    if args.repo_id:
        package_dir = snapshot_hf_repo(args.repo_id, revision=args.revision, token=args.token)
    elif args.input_dir:
        package_dir = args.input_dir
    else:
        raise SystemExit("Provide --input-dir or --repo-id")
    results = run_diagnostics(
        package_dir=package_dir,
        output_dir=args.output_dir,
        normalization_stats=args.normalization_stats,
        horizons=args.horizons,
    )
    print(json.dumps(results["gate"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_diagnostics())
