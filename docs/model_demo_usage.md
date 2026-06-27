# SSE Forecast Model Demo Usage

The trained checkpoint is a future-slip forecasting model, not a professional GNSS-only inversion model.

## One-Event Forecast

```powershell
.\.venv-cu128\Scripts\python.exe scripts\demo_forecast_event.py `
  --run-dir forecast_training_5070ti_full_streaming\random `
  --package-dir hf_dataset_package `
  --split test `
  --index 0 `
  --device cuda
```

Outputs are written to `<run-dir>\demo`:

- `forecast_event_<id>.json`: event id, RMSE, persistence comparison, M0 errors.
- `forecast_event_<id>.m0.csv`: stepwise true/model/persistence summed slip.
- `forecast_event_<id>.png`: future moment curve and final slip maps.

Use `--event-id <id>` for a specific event and `--save-arrays` when full physical slip arrays are needed for MATLAB/Python post-processing.

## Paper Figures

```powershell
.\.venv-cu128\Scripts\python.exe scripts\make_paper_figures.py `
  --training-dir forecast_training_5070ti_lite `
  --output-dir paper_figures
```

Expected outputs:

- `fig_method_overview.png`
- `fig_performance_summary.png`
- `fig_training_curves_combined.png`
- `fig_event_montage.png` when event images exist
- `figure_manifest.json`
- `chatgpt_image_prompts.md`

For final paper figures, regenerate with `--training-dir forecast_training_5070ti_full_streaming` after full random/blocked training finishes.

Current final evidence uses the full 6000-event package and DSW matrix results.
The static demo already exists at:

```text
demo_pages\forecast_random_full\index.html
```

## Static Browser Demo Page

Generate a local HTML report for one trained run:

```powershell
.\.venv-cu128\Scripts\python.exe scripts\build_forecast_demo_page.py `
  --run-dir forecast_training_5070ti_full_streaming\random `
  --package-dir hf_dataset_package `
  --output-dir demo_pages\forecast_random_full `
  --split test `
  --max-events 3 `
  --device cuda
```

Open `demo_pages\forecast_random_full\index.html` in a browser. The page
summarizes the run metrics, data contract, recent training history, and example
forecast figures. It is a static reproducible demo, so it does not require a
server or add a web framework dependency.

The page also includes the ridge-regression inversion proxy figure when present.
This is useful for demonstrating the expected GNSS-to-slip workflow, but it is
not a paper-grade inversion model.

## One-Command Local GUI

For a more convenient demonstration, start a local browser GUI from the generated
static report:

```powershell
.\.venv-cu128\Scripts\python.exe scripts\serve_demo_gui.py `
  --device cuda
```

The launcher automatically uses the first available final local run, regenerates
`demo_pages\forecast_random_full\index.html` when needed, starts a local HTTP
server, and prints the URL. Use `Ctrl+C` in the terminal to stop the server.

To validate the demo without opening a long-running server:

```powershell
.\.venv-cu128\Scripts\python.exe scripts\serve_demo_gui.py `
  --device cpu `
  --check-only
```

This reports the demo page path, forecast figure count, and inversion proxy
figure count. The generated `demo_pages` directory is intentionally ignored by
Git because it is reproducible from tracked scripts and local experiment
outputs.

## Inversion Boundary

Current usable checkpoint solves:

```text
history_gnss + history_slip -> future_slip
```

It should not be described as:

```text
GNSS-only inversion -> slip
```

A paper-grade inversion module needs a separately trained and evaluated GNSS-to-slip head with its own baselines, uncertainty checks, and physical metrics.

## Demonstration Inversion Proxy

For a practical GNSS-to-slip inversion demonstration, use the ridge-regression proxy:

```powershell
.\.venv-cu128\Scripts\python.exe scripts\demo_inversion_proxy.py `
  --run-dir forecast_training_5070ti_full_streaming\random `
  --package-dir hf_dataset_package `
  --split test `
  --index 0 `
  --max-train-events 1200
```

Outputs are written to `<run-dir>\demo_inversion_proxy`:

- `inversion_proxy_event_<id>.json`: RMSE, mean/zero baseline comparison, M0 error.
- `inversion_proxy_event_<id>.png`: true slip, proxy inversion, mean baseline, and error maps for both subfaults.
- `inversion_proxy_event_<id>.npz`: arrays for independent plotting or MATLAB inspection.

This script is intentionally labeled as a proxy. It demonstrates the operational data flow for inversion, but the paper should only claim inversion after a dedicated inversion model is trained and tested.
