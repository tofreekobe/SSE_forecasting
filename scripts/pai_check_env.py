# -*- coding: utf-8 -*-
"""Check a PAI DSW/DLC runtime before launching SSE training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.package_forecast_contract import scan_package_event_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PAI runtime and SSE package visibility.")
    parser.add_argument("--package-dir", default="/mnt/data/hf_dataset_package")
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "package_dir": str(package_dir),
        "package_exists": package_dir.exists(),
        "manifest_exists": (package_dir / "manifest.csv").exists(),
    }
    if info["manifest_exists"]:
        event_ids = scan_package_event_ids(package_dir)
        info["event_count"] = len(event_ids)
        info["first_event_ids"] = event_ids[:5]

    print(json.dumps(info, ensure_ascii=False, indent=2))
    if not info["package_exists"] or not info["manifest_exists"]:
        raise SystemExit("SSE package not found. Mount or download hf_dataset_package before training.")
    if not info["cuda_available"]:
        print("WARNING: CUDA is not available. DSW/DLC is usable, but training will be slow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
