# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset.forecast_contract import (
    SSEForecastDataset,
    build_forecast_contract,
    load_event_arrays,
)
from src.dataset.package_forecast_contract import SSEPackageForecastDataset, build_package_forecast_contract
from src.models.forecast_baselines import (
    compute_package_train_mean_slip,
    compute_train_mean_slip,
    evaluate_package_physical_baselines,
    evaluate_physical_baselines,
)
from src.models.small_forecast_net import SegmentedResidualForecastNet, SegmentedSlipConvForecastNet, SlipConvForecastNet
from scripts.train_forecast_model import evaluate_model


def _write_event(path: Path, event_id: int, amplitude: float) -> None:
    t = 20
    raw = np.zeros((t, 3040), dtype=np.float32)
    raw[:, 0] = 2024
    ramp = np.linspace(0, amplitude, t, dtype=np.float32).reshape(t, 1)
    raw[:, 1:3031] = np.repeat(ramp, 3030, axis=1)
    raw[:, 3031:3040] = np.repeat(ramp * 0.1, 9, axis=1)
    np.savetxt(path / f"fault_sse_catalog_{event_id}.txt", raw, fmt="%.6f")


def _write_package(package_dir: Path) -> None:
    events_dir = package_dir / "events"
    events_dir.mkdir(parents=True)
    event_ids = np.arange(1, 11, dtype=np.int32)
    slips = []
    gnss = []
    for event_id in event_ids:
        ramp = np.linspace(0, float(event_id) / 100.0, 20, dtype=np.float32).reshape(20, 1)
        slips.append(np.repeat(ramp, 3030, axis=1))
        gnss.append(np.repeat(ramp * 0.1, 9, axis=1))
    np.savez_compressed(
        events_dir / "shard_00000.npz",
        event_id=event_ids,
        slip=np.stack(slips, axis=0),
        gnss=np.stack(gnss, axis=0),
    )
    with (package_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("event_id,shard_file,shard_index\n")
        for idx, event_id in enumerate(event_ids):
            f.write(f"{int(event_id)},events/shard_00000.npz,{idx}\n")


def test_forecast_contract_outputs_expected_keys(tmp_path):
    data_dir = tmp_path / "data" / "sse1-500"
    data_dir.mkdir(parents=True)
    for event_id in range(1, 11):
        _write_event(data_dir, event_id, event_id / 100.0)

    splits, slip_transform, gnss_norm, stats = build_forecast_contract(
        tmp_path / "data",
        protocol="blocked",
        history_ratio=0.5,
        forecast_horizon=5,
    )
    ds = SSEForecastDataset(
        tmp_path / "data",
        splits.train,
        slip_transform,
        gnss_norm,
        history_steps=stats.history_steps,
        forecast_horizon=5,
        return_physical=True,
    )
    item = ds[0]
    assert set(["history_gnss", "history_slip", "future_gnss", "future_slip", "metadata"]).issubset(item)
    assert item["history_slip"].shape == (10, 3030)
    assert item["future_slip"].shape == (5, 3030)
    assert float(item["history_slip"].min()) >= 0.0

    decoded = slip_transform.decode(item["future_slip"].numpy())
    assert np.allclose(decoded, item["future_slip_physical"].numpy(), atol=1e-6)


def test_physical_baselines_smoke(tmp_path):
    data_dir = tmp_path / "data" / "sse1-500"
    data_dir.mkdir(parents=True)
    for event_id in range(1, 11):
        _write_event(data_dir, event_id, event_id / 100.0)

    splits, _, _, stats = build_forecast_contract(
        tmp_path / "data",
        protocol="random",
        seed=1,
        history_ratio=0.5,
        forecast_horizon=5,
    )
    mean = compute_train_mean_slip(tmp_path / "data", splits.train)
    metrics = evaluate_physical_baselines(
        tmp_path / "data",
        splits.test,
        mean,
        history_steps=stats.history_steps,
        horizons=(1, 5),
    )
    assert metrics["1"]["event_count"] > 0
    assert metrics["1"]["rmse_persistence"] < metrics["1"]["rmse_zero"]


def test_package_contract_and_baselines_smoke(tmp_path):
    package_dir = tmp_path / "hf_package"
    _write_package(package_dir)

    splits, _, _, stats = build_package_forecast_contract(
        package_dir,
        protocol="blocked",
        history_ratio=0.5,
        forecast_horizon=5,
    )
    mean = compute_package_train_mean_slip(package_dir, splits.train)
    metrics = evaluate_package_physical_baselines(
        package_dir,
        splits.test,
        mean,
        history_steps=stats.history_steps,
        horizons=(1, 5),
    )
    assert stats.train_event_count == 7
    assert mean.shape == (20, 3030)
    assert metrics["5"]["event_count"] > 0
    assert metrics["1"]["rmse_persistence"] < metrics["1"]["rmse_zero"]


def test_small_forecast_net_forward_shape():
    model = SlipConvForecastNet(history_steps=10, forecast_horizon=5, hidden_channels=8)
    history_slip = np.random.default_rng(1).random((2, 10, 3030), dtype=np.float32)
    history_gnss = np.random.default_rng(2).random((2, 10, 9), dtype=np.float32)

    pred = model(torch.from_numpy(history_slip), torch.from_numpy(history_gnss))

    assert pred.shape == (2, 5, 3030)
    assert float(pred.detach().min()) >= 0.0


def test_segmented_small_forecast_net_forward_shape():
    model = SegmentedSlipConvForecastNet(history_steps=10, forecast_horizon=5, hidden_channels=8)
    history_slip = np.random.default_rng(3).random((2, 10, 3030), dtype=np.float32)
    history_gnss = np.random.default_rng(4).random((2, 10, 9), dtype=np.float32)

    pred = model(torch.from_numpy(history_slip), torch.from_numpy(history_gnss))

    assert pred.shape == (2, 5, 3030)
    assert float(pred.detach().min()) >= 0.0


def test_segmented_residual_forecast_net_forward_shape():
    model = SegmentedResidualForecastNet(history_steps=10, forecast_horizon=5, hidden_channels=8)
    history_slip = np.random.default_rng(5).random((2, 10, 3030), dtype=np.float32)
    history_gnss = np.random.default_rng(6).random((2, 10, 9), dtype=np.float32)

    pred = model(torch.from_numpy(history_slip), torch.from_numpy(history_gnss))

    assert pred.shape == (2, 5, 3030)
    assert float(pred.detach().min()) >= 0.0


def test_streaming_forecast_evaluation_respects_max_batches(tmp_path):
    package_dir = tmp_path / "hf_package"
    _write_package(package_dir)
    splits, slip_transform, gnss_norm, stats = build_package_forecast_contract(
        package_dir,
        protocol="blocked",
        history_ratio=0.5,
        forecast_horizon=5,
    )
    ds = SSEPackageForecastDataset(
        package_dir,
        splits.train,
        slip_transform,
        gnss_norm,
        history_steps=stats.history_steps,
        forecast_horizon=5,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    model = SegmentedResidualForecastNet(history_steps=stats.history_steps, forecast_horizon=5, hidden_channels=8)

    metrics = evaluate_model(model, loader, slip_transform, torch.device("cpu"), max_batches=1)

    assert metrics["batch_count"] == 1.0
    assert metrics["event_count"] == 2.0
    assert np.isfinite(metrics["physical_rmse"])
    assert np.isfinite(metrics["physical_r2"])
