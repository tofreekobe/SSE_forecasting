# -*- coding: utf-8 -*-
"""Run SSE feasibility diagnostics locally or from a private HF dataset repo."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.diagnostics.hf_sse_diagnostics import cli_diagnostics


if __name__ == "__main__":
    raise SystemExit(cli_diagnostics())
