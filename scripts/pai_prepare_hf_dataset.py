# -*- coding: utf-8 -*-
"""Download the private SSE dataset package inside a PAI DSW/DLC runtime."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DEFAULT_PACKAGE_FILES = [
    "manifest.csv",
    "manifest.jsonl",
    "dataset_metadata.json",
    "normalization_stats.json",
    "README.md",
    "events/*.npz",
]


def _detect_repo_prefix(repo_id: str, token: str, revision: str | None) -> str:
    files = HfApi().list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    if "manifest.csv" in files:
        return ""
    if "hf_dataset_package/manifest.csv" in files:
        return "hf_dataset_package/"
    sample = "\n".join(files[:30])
    raise SystemExit(
        "Could not find manifest.csv at repo root or hf_dataset_package/manifest.csv. "
        f"First repo files:\n{sample}"
    )


def _prefix_patterns(prefix: str, patterns: list[str]) -> list[str]:
    if not prefix:
        return patterns
    return [f"{prefix}{pattern}" for pattern in patterns]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare hf_dataset_package from a HF dataset repo.")
    parser.add_argument("--repo-id", default="tofreekobe/sse-slow-slip-private")
    parser.add_argument("--package-dir", default="/mnt/data/hf_dataset_package")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=DEFAULT_PACKAGE_FILES,
        help="HF allow pattern. Can be repeated.",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Export a private dataset read token before downloading.")

    package_dir = Path(args.package_dir).resolve()
    package_dir.parent.mkdir(parents=True, exist_ok=True)

    prefix = _detect_repo_prefix(args.repo_id, token, args.revision)
    local_dir = package_dir if not prefix else package_dir.parent
    allow_patterns = _prefix_patterns(prefix, args.allow_pattern)
    print(f"Detected HF package prefix: {prefix or '<repo root>'}")
    print(f"Downloading to: {local_dir}")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(local_dir),
        token=token,
        allow_patterns=allow_patterns,
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
