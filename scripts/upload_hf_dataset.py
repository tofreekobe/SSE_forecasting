# -*- coding: utf-8 -*-
"""Upload a prepared SSE diagnostics package to a private HF dataset repo."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload an SSE diagnostics package to Hugging Face.")
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--repo-id", required=True, help="Example: tofreekobe/sse-slow-slip-private")
    parser.add_argument("--token", default=None)
    parser.add_argument("--public", action="store_true", help="Create public repo instead of private.")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except Exception as exc:
        raise SystemExit(
            "Missing huggingface_hub. Install diagnostics requirements first: "
            "pip install -r requirements-diagnostics.txt"
        ) from exc

    package_dir = Path(args.package_dir)
    if not package_dir.exists():
        raise SystemExit(f"Package directory does not exist: {package_dir}")

    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
        token=args.token,
    )
    api = HfApi(token=args.token)
    info = api.repo_info(args.repo_id, repo_type="dataset")
    print(f"Uploading to dataset repo: {info.id}")
    upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(package_dir),
        path_in_repo=".",
        token=args.token,
    )
    print("Upload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

