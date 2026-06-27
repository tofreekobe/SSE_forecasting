# -*- coding: utf-8 -*-
"""Sync lightweight DSW experiment-matrix result files to the local workspace."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


DEFAULT_PATTERNS = (
    "metrics.json",
    "forecast_training_report.md",
    "training_history.csv",
    "forecast_contract_stats.json",
    "baseline_*.json",
    "baseline_*.csv",
    "experiment_matrix_summary.csv",
    "experiment_matrix_summary.md",
)


def run_command(args: list[str]) -> str:
    result = subprocess.run(args, check=True, text=True, capture_output=True)
    return result.stdout


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def list_remote_files(host: str, remote_root: str, patterns: tuple[str, ...]) -> list[str]:
    find_parts = []
    for pattern in patterns:
        find_parts.extend(["-name", shell_quote(pattern), "-o"])
    if find_parts:
        find_expr = " ".join(find_parts[:-1])
    else:
        find_expr = "-type f"
    command = f"cd {shell_quote(remote_root)} && find . -type f \\( {find_expr} \\) | sort"
    output = run_command(["ssh", host, command])
    return [line.strip()[2:] for line in output.splitlines() if line.strip().startswith("./")]


def sync_files(host: str, remote_root: str, local_root: Path, files: list[str], retries: int) -> list[str]:
    local_root.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for rel in files:
        local_path = local_root / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        remote_path = f"{host}:{remote_root.rstrip('/')}/{rel}"
        for attempt in range(1, retries + 1):
            result = subprocess.run(["scp", remote_path, str(local_path)])
            if result.returncode == 0:
                break
            if attempt == retries:
                failed.append(rel)
            else:
                print(f"Retrying {rel} after scp failure ({attempt}/{retries})")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync lightweight DSW SSE experiment results.")
    parser.add_argument("--host", default="aliyun-dsw-final-sse-via-124")
    parser.add_argument("--remote-root", default="/mnt/workspace/sse_outputs/experiment_matrix_b1f13c4_full")
    parser.add_argument("--local-root", default="dsw_results/experiment_matrix_b1f13c4_full")
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Remote filename pattern to include. Can be repeated. Defaults to metrics/reports/history/summaries.",
    )
    parser.add_argument("--retries", type=int, default=3, help="scp retries per file.")
    args = parser.parse_args()

    patterns = tuple(args.pattern) if args.pattern else DEFAULT_PATTERNS
    files = list_remote_files(args.host, args.remote_root, patterns)
    if not files:
        print("No matching remote result files found.")
        return 0
    failed = sync_files(args.host, args.remote_root, Path(args.local_root), files, max(args.retries, 1))
    synced = len(files) - len(failed)
    print(f"Synced {synced}/{len(files)} files to {os.path.abspath(args.local_root)}")
    for rel in files:
        status = "FAILED" if rel in failed else "OK"
        print(f"{status} {rel}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
