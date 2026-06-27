# GitHub References For SSE Forecasting Figures

These repositories are references for structure, evaluation style, or plotting conventions. They are not copied into this project.

## Time-Series Forecasting Baselines

- `thuml/Time-Series-Library`: useful as a benchmark organization reference because it separates data providers, experiment runners, model implementations, scripts, and utilities, and covers forecasting, imputation, anomaly detection, and classification.
  https://github.com/thuml/Time-Series-Library

- `amazon-science/chronos-forecasting`: useful as a modern pretrained time-series forecasting reference. Its README documents direct multi-step probabilistic forecasting interfaces and covariate-informed variants.
  https://github.com/amazon-science/chronos-forecasting

- `google-research/timesfm`: useful as a time-series foundation model reference. Its README frames TimesFM as a pretrained forecasting model and documents current model versions, horizons, quantile forecasts, and fine-tuning examples.
  https://github.com/google-research/timesfm

## Classic Deep-Learning Experiment Presentation

- `pytorch/examples`: useful as a minimal, reproducible training-script reference. For this project, the comparable figure convention is not classification accuracy but explicit loss curves, validation metrics, saved checkpoints, and deterministic command lines.
  https://github.com/pytorch/examples

- `Lightning-AI/pytorch-lightning`: useful as a modern training-loop and logging reference. The local project keeps plain PyTorch, but the figure/report structure follows the same idea: metrics per epoch, validation split, test split, and reproducible experiment metadata.
  https://github.com/Lightning-AI/pytorch-lightning

- `wandb/examples`: useful as an experiment-tracking reference for presenting training curves, metric tables, and artifact-linked runs. The local project stays file-based, but `training_history.csv`, `metrics.json`, and generated PNGs serve the same audit role.
  https://github.com/wandb/examples

## GNSS / Geodetic Plotting

- `kmaterna/GNSS_TimeSeries_Viewers`: useful for GNSS time-series plotting conventions, station stack views, offsets, seasonal terms, and slope visualizations.
  https://github.com/kmaterna/GNSS_TimeSeries_Viewers

- `GenericMappingTools/pygmt`: useful for professional geophysical map styling, colorbars, gridded data panels, and publication-quality figure conventions.
  https://github.com/GenericMappingTools/pygmt

## Slow Slip / Geodetic Inversion References

- `ryota-takagi/GriD-SSE`: directly slow-slip related; useful as a domain reference for SSE data/model organization, not as a drop-in forecasting implementation.
  https://github.com/ryota-takagi/GriD-SSE

- `HaxbyH/JapanSSEs`: slow-slip event data/material reference for Japan SSE studies.
  https://github.com/HaxbyH/JapanSSEs

- `RiveHe/SSEdetect`: slow-slip detection reference. The current project differs because the primary target is future slip-field forecasting rather than detection.
  https://github.com/RiveHe/SSEdetect

- `ustcer-jun/Geodetic-Finite-Fault-Inversion-Python-Package`: relevant to the inversion side of the project because it frames GNSS/geodetic data as finite-fault slip inversion inputs.
  https://github.com/ustcer-jun/Geodetic-Finite-Fault-Inversion-Python-Package

## Local Design Consequence

For this SSE project, the right figure set is not a generic Transformer benchmark figure. It should show:

- the corrected two-subfault geometry;
- history/future data contract;
- a geophysics-style event-window panel with GNSS history, summed slip, and separated subfault maps;
- baseline curves across h=1/5/10/30/50;
- random and blocked split metrics;
- persistence and mean baselines at h50;
- event-level moment curves and final slip maps;
- explicit limits that the current trained checkpoint is forecasting, not GNSS-only inversion.

## Generated Local Figure Set

`scripts/make_paper_figures.py` now generates:

- `fig_method_overview.png`: compact model workflow.
- `fig_data_contract_event_window.png`: geophysics-style event, GNSS, and two-subfault data contract.
- `fig_baseline_horizon_curves.png`: zero/mean/persistence baselines across forecast horizons.
- `fig_performance_summary.png`: h50 model-vs-baseline metrics and publication gates.
- `fig_training_curves_combined.png`: deep-learning training and validation curves.
- `fig_event_montage.png`: representative forecast event panels.

The generated figures intentionally separate three visual languages:

- geophysics: event windows, GNSS traces, separated subfault slip maps;
- forecasting: horizon baselines and future-step moment curves;
- deep learning: loss/validation curves, held-out split tables, and explicit gate thresholds.
