# -*- coding: utf-8 -*-
"""Audit the compressed SSE package against the 74 GiB raw event catalog."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def scan_raw_files(data_dir: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted(data_dir.glob("**/fault_sse_catalog_*.txt")):
        stem = path.stem
        event_id = int(stem.rsplit("_", 1)[1])
        if event_id in files:
            raise RuntimeError(f"Duplicate raw event id {event_id}: {files[event_id]} and {path}")
        files[event_id] = path
    return files


def read_manifest(package_dir: Path) -> list[dict[str, str]]:
    with (package_dir / "manifest.csv").open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(package_dir: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys())
    for name in ("manifest.csv", "manifest.jsonl"):
        path = package_dir / name
        if path.exists():
            backup = package_dir / f"{name}.bak"
            if not backup.exists():
                path.replace(backup)
    with (package_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (package_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def repair_manifest_indices(package_dir: Path, rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], int]:
    by_event = {int(row["event_id"]): dict(row) for row in rows}
    changed = 0
    for shard_path in sorted((package_dir / "events").glob("*.npz")):
        shard_file = f"events/{shard_path.name}"
        with np.load(shard_path) as shard:
            for local_index, event_id in enumerate(shard["event_id"].astype(int).tolist()):
                row = by_event[event_id]
                old_file = row.get("shard_file", "")
                old_index = int(row.get("shard_index", -1))
                if old_file != shard_file or old_index != local_index:
                    row["shard_file"] = shard_file
                    row["shard_index"] = local_index
                    changed += 1
    return [by_event[event_id] for event_id in sorted(by_event)], changed


def load_package_event(package_dir: Path, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(package_dir / row["shard_file"]) as shard:
        idx = int(row["shard_index"])
        event_id = int(row["event_id"])
        if int(shard["event_id"][idx]) != event_id:
            matches = np.where(shard["event_id"].astype(int) == event_id)[0]
            if matches.size != 1:
                raise RuntimeError(f"Cannot locate event {event_id} in {row['shard_file']}")
            idx = int(matches[0])
        return shard["slip"][idx].astype(np.float32), shard["gnss"][idx].astype(np.float32)


def compare_event(raw_path: Path, package_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    raw = np.loadtxt(raw_path, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    raw_slip = raw[:, 1:3031]
    raw_gnss = raw[:, 3031:3040]
    pkg_slip, pkg_gnss = load_package_event(package_dir, row)
    return {
        "event_id": int(row["event_id"]),
        "raw_path": str(raw_path),
        "raw_shape": list(raw.shape),
        "package_slip_shape": list(pkg_slip.shape),
        "package_gnss_shape": list(pkg_gnss.shape),
        "max_abs_slip_diff": float(np.max(np.abs(raw_slip - pkg_slip))),
        "max_abs_gnss_diff": float(np.max(np.abs(raw_gnss - pkg_gnss))),
        "raw_slip_sum": float(raw_slip.sum(dtype=np.float64)),
        "package_slip_sum": float(pkg_slip.sum(dtype=np.float64)),
    }


def choose_audit_ids(ids: list[int], mode: str, samples: int, seed: int) -> list[int]:
    if mode == "all":
        return ids
    anchors = [ids[0], ids[-1]]
    boundary_ids = [500, 501, 1000, 1001, 1500, 1501, 3000, 3001, 5500, 5501]
    chosen = {event_id for event_id in anchors + boundary_ids if event_id in ids}
    remaining = [event_id for event_id in ids if event_id not in chosen]
    rng = random.Random(seed)
    if samples > len(chosen):
        chosen.update(rng.sample(remaining, min(samples - len(chosen), len(remaining))))
    return sorted(chosen)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    failed = [row for row in payload["comparisons"] if row["max_abs_slip_diff"] > 0 or row["max_abs_gnss_diff"] > 0]
    lines = [
        "# Full SSE Dataset Package Audit",
        "",
        f"- Raw data directory: `{payload['data_dir']}`",
        f"- Package directory: `{payload['package_dir']}`",
        f"- Raw event count: `{payload['raw_event_count']}`",
        f"- Manifest event count: `{payload['manifest_event_count']}`",
        f"- Raw total bytes: `{payload['raw_total_bytes']}`",
        f"- Raw total GiB: `{payload['raw_total_gib']:.3f}`",
        f"- Package total bytes: `{payload['package_total_bytes']}`",
        f"- Package total GiB: `{payload['package_total_gib']:.3f}`",
        f"- Event IDs continuous 1..6000: `{payload['event_ids_continuous_1_6000']}`",
        f"- Manifest index repair changes: `{payload['manifest_index_changes']}`",
        f"- Compared events: `{len(payload['comparisons'])}`",
        f"- Exact comparison failures: `{len(failed)}`",
        "",
        "## Compared Events",
        "",
        "| Event | Raw shape | Slip diff max | GNSS diff max |",
        "| ---: | --- | ---: | ---: |",
    ]
    for row in payload["comparisons"]:
        lines.append(
            f"| {row['event_id']} | {row['raw_shape']} | "
            f"{row['max_abs_slip_diff']:.6g} | {row['max_abs_gnss_diff']:.6g} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit/repair compressed SSE package against raw full data.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--package-dir", default="hf_dataset_package")
    parser.add_argument("--mode", choices=["sample", "all"], default="sample")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--repair-manifest", action="store_true")
    parser.add_argument("--output-json", default="diagnostics_full_local/full_dataset_package_audit.json")
    parser.add_argument("--output-md", default="diagnostics_full_local/full_dataset_package_audit.md")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    package_dir = Path(args.package_dir)
    raw_files = scan_raw_files(data_dir)
    rows = read_manifest(package_dir)
    repaired_rows, changes = repair_manifest_indices(package_dir, rows)
    if args.repair_manifest and changes:
        write_manifest(package_dir, repaired_rows)
        rows = read_manifest(package_dir)
    else:
        rows = repaired_rows

    raw_ids = sorted(raw_files)
    manifest_ids = sorted(int(row["event_id"]) for row in rows)
    row_by_id = {int(row["event_id"]): row for row in rows}
    audit_ids = choose_audit_ids(manifest_ids, args.mode, args.samples, args.seed)
    comparisons = [compare_event(raw_files[event_id], package_dir, row_by_id[event_id]) for event_id in audit_ids]

    raw_total_bytes = sum(path.stat().st_size for path in raw_files.values())
    package_total_bytes = sum(path.stat().st_size for path in package_dir.rglob("*") if path.is_file())
    payload = {
        "data_dir": str(data_dir),
        "package_dir": str(package_dir),
        "raw_event_count": len(raw_files),
        "manifest_event_count": len(rows),
        "raw_total_bytes": raw_total_bytes,
        "raw_total_gib": raw_total_bytes / (1024**3),
        "package_total_bytes": package_total_bytes,
        "package_total_gib": package_total_bytes / (1024**3),
        "raw_id_min": min(raw_ids) if raw_ids else None,
        "raw_id_max": max(raw_ids) if raw_ids else None,
        "manifest_id_min": min(manifest_ids) if manifest_ids else None,
        "manifest_id_max": max(manifest_ids) if manifest_ids else None,
        "event_ids_continuous_1_6000": raw_ids == list(range(1, 6001)) and manifest_ids == list(range(1, 6001)),
        "manifest_index_changes": changes,
        "repair_manifest_written": bool(args.repair_manifest and changes),
        "audit_mode": args.mode,
        "comparisons": comparisons,
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output_md, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
