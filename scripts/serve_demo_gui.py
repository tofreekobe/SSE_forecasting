# -*- coding: utf-8 -*-
"""Launch the local static SSE forecast demo as a small browser GUI."""

from __future__ import annotations

import argparse
import http.server
import json
import socket
import subprocess
import sys
import webbrowser
from functools import partial
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_CANDIDATES = (
    Path("forecast_training_5070ti_full_streaming") / "random",
    Path("forecast_training_5070ti_full") / "random",
    Path("forecast_training_5070ti_dataset_recheck") / "random",
)


def _resolve_existing_dir(path: str | Path) -> Path:
    resolved = (PROJECT_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def _find_default_run_dir() -> Path:
    for candidate in DEFAULT_RUN_CANDIDATES:
        path = PROJECT_ROOT / candidate
        if (path / "metrics.json").exists():
            return path
    tried = ", ".join(str(candidate) for candidate in DEFAULT_RUN_CANDIDATES)
    raise FileNotFoundError(f"No default demo run found. Tried: {tried}")


def _find_free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_demo_page(
    run_dir: Path,
    package_dir: Path,
    output_dir: Path,
    split: str,
    max_events: int,
    device: str,
) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "build_forecast_demo_page.py"),
        "--run-dir",
        str(run_dir),
        "--package-dir",
        str(package_dir),
        "--output-dir",
        str(output_dir),
        "--split",
        split,
        "--max-events",
        str(max_events),
        "--device",
        device,
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True, stdout=subprocess.DEVNULL)


def _load_summary(index_path: Path) -> dict[str, object]:
    figures = sorted((index_path.parent / "figures").glob("*.png"))
    inversion = sorted((index_path.parent / "inversion_proxy").glob("*.png"))
    return {
        "index": str(index_path),
        "figure_count": len(figures),
        "inversion_proxy_count": len(inversion),
        "exists": index_path.exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="Training run directory. Defaults to the first available final local run.")
    parser.add_argument("--package-dir", default="hf_dataset_package")
    parser.add_argument("--output-dir", default=str(Path("demo_pages") / "forecast_random_full"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-build", action="store_true", help="Serve an existing index.html without regenerating figures.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically.")
    parser.add_argument("--check-only", action="store_true", help="Build/validate the demo page and exit without serving.")
    args = parser.parse_args()

    run_dir = _resolve_existing_dir(args.run_dir) if args.run_dir else _find_default_run_dir()
    package_dir = _resolve_existing_dir(args.package_dir)
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    index_path = output_dir / "index.html"

    if not args.no_build:
        _build_demo_page(run_dir, package_dir, output_dir, args.split, args.max_events, args.device)
    elif not index_path.exists():
        raise FileNotFoundError(index_path)

    summary = _load_summary(index_path)
    if args.check_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    port = _find_free_port(args.port)
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(output_dir))
    server = http.server.ThreadingHTTPServer((args.host, port), handler)
    url = f"http://{args.host}:{port}/index.html"
    print(json.dumps({**summary, "url": url}, ensure_ascii=False, indent=2))
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped SSE demo GUI.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
