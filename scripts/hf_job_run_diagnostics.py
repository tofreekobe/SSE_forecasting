# -*- coding: utf-8 -*-
"""Entry point executed by Hugging Face Jobs for SSE diagnostics."""

from __future__ import annotations

import importlib.util
import json
import os
import sys

from huggingface_hub import create_repo, snapshot_download, upload_folder


def main() -> int:
    repo_id = os.environ["SSE_DATASET_REPO"]
    out_repo = os.environ.get("SSE_DIAGNOSTICS_REPO", "")
    token = os.environ.get("HF_TOKEN")
    package_dir = snapshot_download(repo_id=repo_id, repo_type="dataset", token=token)

    module_path = os.path.join(package_dir, "hf_sse_diagnostics.py")
    spec = importlib.util.spec_from_file_location("hf_sse_diagnostics", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import diagnostics module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    results = module.run_diagnostics(package_dir, "hf_diagnostics_output")
    print(json.dumps(results["gate"], indent=2, ensure_ascii=True))

    if out_repo:
        create_repo(out_repo, repo_type="dataset", private=True, exist_ok=True, token=token)
        upload_folder(
            repo_id=out_repo,
            repo_type="dataset",
            folder_path="hf_diagnostics_output",
            path_in_repo=".",
            token=token,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

