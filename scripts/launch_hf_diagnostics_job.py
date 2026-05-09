# -*- coding: utf-8 -*-
"""Launch or print the HF Jobs command for remote SSE diagnostics."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a CPU HF Job for SSE diagnostics.")
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--diagnostics-repo", default="")
    parser.add_argument("--flavor", default="cpu-basic")
    parser.add_argument("--timeout", default="6h")
    parser.add_argument("--run", action="store_true", help="Actually execute hf jobs; otherwise print command.")
    args = parser.parse_args()

    script_path = Path(__file__).with_name("hf_job_run_diagnostics.py").resolve()
    cmd = [
        "hf",
        "jobs",
        "uv",
        "run",
        "--flavor",
        args.flavor,
        "--timeout",
        args.timeout,
        "--with",
        "numpy",
        "--with",
        "matplotlib",
        "--with",
        "huggingface_hub",
        "--env",
        f"SSE_DATASET_REPO={args.dataset_repo}",
        "--env",
        f"SSE_DIAGNOSTICS_REPO={args.diagnostics_repo}",
        "--secrets",
        "HF_TOKEN",
        str(script_path),
    ]
    print(" ".join(cmd))
    if not args.run:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
