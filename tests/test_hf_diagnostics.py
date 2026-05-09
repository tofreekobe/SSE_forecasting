# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.diagnostics.hf_sse_diagnostics import (
    build_hf_dataset_package,
    parse_event_id,
    run_diagnostics,
)


def _write_event(path: Path, event_id: int, scale: float) -> None:
    t = 273
    raw = np.zeros((t, 3040), dtype=np.float32)
    raw[:, 0] = 2024
    ramp = np.linspace(0, scale, t, dtype=np.float32).reshape(t, 1)
    slip = np.repeat(ramp, 3030, axis=1)
    gnss = np.zeros((t, 9), dtype=np.float32)
    gnss[:, 0] = ramp[:, 0] * 0.1
    gnss[:, 1] = ramp[:, 0] * 0.2
    raw[:, 1:3031] = slip
    raw[:, 3031:3040] = gnss
    np.savetxt(path / f"fault_sse_catalog_{event_id}.txt", raw, fmt="%.6f")


def test_parse_event_id():
    assert parse_event_id("data/sse1-500/fault_sse_catalog_42.txt") == 42


def test_package_and_diagnostics_smoke(tmp_path):
    data_dir = tmp_path / "data" / "sse1-500"
    data_dir.mkdir(parents=True)
    _write_event(data_dir, 1, 0.1)
    _write_event(data_dir, 2, 0.2)

    stats = {
        "slip_mean": [0.05] * 3030,
        "slip_std": [0.01] * 3030,
        "gnss_mean": [0.0] * 9,
        "gnss_std": [1.0] * 9,
    }
    (tmp_path / "data" / "normalization_stats.json").write_text(json.dumps(stats), encoding="utf-8")

    package_dir = tmp_path / "package"
    build_hf_dataset_package(
        data_dir=tmp_path / "data",
        output_dir=package_dir,
        shard_size=1,
        include_arrays=True,
    )
    assert (package_dir / "manifest.csv").exists()
    assert (package_dir / "hf_sse_diagnostics.py").exists()
    assert len(list((package_dir / "events").glob("*.npz"))) == 2

    out_dir = tmp_path / "diagnostics"
    result = run_diagnostics(package_dir, out_dir, horizons=(1, 5, 10))
    assert (out_dir / "hf_diagnostics.json").exists()
    assert (out_dir / "hf_diagnostics.md").exists()
    assert (out_dir / "baseline_metrics.csv").exists()
    assert result["gate"]["conclusion"] in {"GO_WITH_CHANGES", "NO_GO"}
    assert result["legacy_scheme_counterexample"]["softplus_conflict"] is True
