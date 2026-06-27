# -*- coding: utf-8 -*-
"""Build a static browser demo page for an SSE forecast run."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}g}"
    return html.escape(str(value))


def _read_history(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _ensure_figures(run_dir: Path, package_dir: Path, figures_dir: Path, split: str, max_events: int, device: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    existing = list(figures_dir.glob("forecast_event_*.png"))
    if existing and (figures_dir / "training_curves.png").exists():
        return
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "plot_forecast_examples.py"),
        "--run-dir",
        str(run_dir),
        "--package-dir",
        str(package_dir),
        "--output-dir",
        str(figures_dir),
        "--split",
        split,
        "--max-events",
        str(max_events),
        "--device",
        device,
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _metric_cards(metrics: dict) -> str:
    final_test = metrics.get("final_test") or {}
    baseline = metrics.get("baseline_test_h50") or {}
    cards = [
        ("Model h50 RMSE", final_test.get("physical_rmse")),
        ("Persistence h50 RMSE", baseline.get("rmse_persistence")),
        ("RMSE improvement", metrics.get("test_rmse_improvement_pct")),
        ("Model h50 R2", final_test.get("physical_r2")),
        ("Model M0 rel abs", final_test.get("m0_rel_abs")),
        ("Publication gate", metrics.get("publication_gate")),
    ]
    parts = []
    for title, value in cards:
        suffix = "%" if title == "RMSE improvement" and isinstance(value, (int, float)) else ""
        parts.append(
            "<div class='metric-card'>"
            f"<div class='metric-title'>{html.escape(title)}</div>"
            f"<div class='metric-value'>{_fmt(value)}{suffix}</div>"
            "</div>"
        )
    return "\n".join(parts)


def _history_table(rows: list[dict[str, str]], max_rows: int = 8) -> str:
    if not rows:
        return "<p class='muted'>No training history found.</p>"
    selected = rows[-max_rows:]
    headers = ["epoch", "train_loss", "train_rmse", "val_rmse", "val_r2", "val_m0_rel_abs"]
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body_rows = []
    for row in selected:
        cells = "".join(f"<td>{_fmt(float(row[h])) if h in row and row[h] else 'n/a'}</td>" for h in headers)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _figure_gallery(figures_dir: Path, output_dir: Path) -> str:
    figures = sorted(figures_dir.glob("*.png"))
    if not figures:
        return "<p class='muted'>No figures generated.</p>"
    parts = []
    for figure in figures:
        rel = os.path.relpath(figure, output_dir).replace("\\", "/")
        title = figure.stem.replace("_", " ")
        parts.append(
            "<figure>"
            f"<img src='{html.escape(rel)}' alt='{html.escape(title)}'>"
            f"<figcaption>{html.escape(title)}</figcaption>"
            "</figure>"
        )
    return "\n".join(parts)


def _write_html(output_dir: Path, run_dir: Path, package_dir: Path, metrics: dict, history_rows: list[dict[str, str]], figures_dir: Path) -> None:
    protocol = metrics.get("protocol", run_dir.name)
    model_type = metrics.get("model_type", "n/a")
    input_mode = metrics.get("input_mode", "full")
    data_rows = [
        ("Run directory", str(run_dir)),
        ("Package directory", str(package_dir)),
        ("Protocol", protocol),
        ("Model type", model_type),
        ("Input mode", input_mode),
        ("Train / Val / Test events", f"{metrics.get('train_event_count')} / {metrics.get('val_event_count')} / {metrics.get('test_event_count')}"),
        ("Forecast start / horizon", f"{metrics.get('forecast_start')} / {metrics.get('forecast_horizon')}"),
        ("Raw catalog scope", "6000 events, 74.202 GiB; audited compressed package is a full representation, not a subset"),
    ]
    data_table = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in data_rows
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SSE Forecast Demo - {html.escape(str(protocol))}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #0f766e;
      --accent-2: #334155;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.55;
    }}
    header {{
      background: #0b1220;
      color: white;
      padding: 28px 32px;
    }}
    header h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    header p {{ margin: 0; color: #cbd5e1; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 18px;
    }}
    h2 {{ margin: 0 0 14px; font-size: 20px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
    }}
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfe;
    }}
    .metric-title {{ color: var(--muted); font-size: 13px; }}
    .metric-value {{ font-size: 24px; font-weight: 700; color: var(--accent); margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; vertical-align: top; }}
    th {{ color: var(--accent-2); width: 240px; }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    figure {{ margin: 0; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: white; }}
    img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ padding: 10px 12px; color: var(--muted); font-size: 13px; }}
    .muted {{ color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <h1>SSE Future-Slip Forecast Demo</h1>
    <p>Geometry-aware 50-step slip-field forecasting with physical-unit metrics.</p>
  </header>
  <main>
    <section>
      <h2>Run Summary</h2>
      <div class="metrics">
        {_metric_cards(metrics)}
      </div>
    </section>
    <section>
      <h2>Data Contract</h2>
      <table><tbody>{data_table}</tbody></table>
    </section>
    <section>
      <h2>Recent Training History</h2>
      {_history_table(history_rows)}
    </section>
    <section>
      <h2>Figures</h2>
      <div class="gallery">
        {_figure_gallery(figures_dir, output_dir)}
      </div>
    </section>
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a static SSE forecast demo page.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--package-dir", default="hf_dataset_package")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "demo_page"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = _load_json(metrics_path)
    _ensure_figures(run_dir, Path(args.package_dir), figures_dir, args.split, args.max_events, args.device)
    history_rows = _read_history(run_dir / "training_history.csv")
    _write_html(output_dir, run_dir, Path(args.package_dir), metrics, history_rows, figures_dir)
    print(json.dumps({"demo_page": str(output_dir / "index.html"), "figures_dir": str(figures_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

