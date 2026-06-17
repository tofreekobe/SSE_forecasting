# -*- coding: utf-8 -*-
"""Download the private SSE dataset package inside a PAI DSW/DLC runtime."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare hf_dataset_package from a HF dataset repo.")
    parser.add_argument("--repo-id", default="tofreekobe/sse-slow-slip-private")
    parser.add_argument("--package-dir", default="/mnt/data/hf_dataset_package")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[
            "manifest.csv",
            "manifest.jsonl",
            "dataset_metadata.json",
            "normalization_stats.json",
            "README.md",
            "events/*.npz",
        ],
        help="HF allow pattern. Can be repeated.",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Export a private dataset read token before downloading.")

    package_dir = Path(args.package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(package_dir),
        token=token,
        allow_patterns=args.allow_pattern,
    )

    manifest = package_dir / "manifest.csv"
    events_dir = package_dir / "events"
    shard_count = len(list(events_dir.glob("*.npz"))) if events_dir.exists() else 0
    if not manifest.exists() or shard_count == 0:
        raise SystemExit(
            f"Dataset download incomplete: manifest_exists={manifest.exists()} shard_count={shard_count}"
        )

    print(f"Prepared SSE package at {package_dir}")
    print(f"NPZ shard count: {shard_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
