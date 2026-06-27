# Local RTX 5070 Ti Training Results

Date: 2026-06-27

Environment:

- GPU: NVIDIA GeForce RTX 5070 Ti
- Driver CUDA: 12.8
- PyTorch: 2.11.0+cu128
- CUDA capability: sm_120

Key change:

- The residual forecast head now starts from persistence by zero-initializing the final delta convolution.
- `run_small_overfit.py` restores the best checkpoint instead of reporting the final oscillating epoch.
- `plot_forecast_examples.py` generates true/model/persistence forecast plots from saved checkpoints.

Local run artifacts:

- Small overfit: `small_overfit_5070ti_persist_init_16e240/`
- Random split: `forecast_training_5070ti_lite/random/`
- Blocked split: `forecast_training_5070ti_lite/blocked/`
- Full random/blocked: `forecast_training_5070ti_full_streaming/`
- Summary: `forecast_training_5070ti_lite/training_summary.md`
- Full summary: `forecast_training_5070ti_full_streaming/training_summary.md`
- Full paper figures: `paper_figures_full/`

Metrics:

| Run | Events | Test h50 RMSE | Persistence h50 RMSE | Improvement | R2 | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| small overfit | 16 train | 0.02282 | 0.06147 | 62.88% | 0.8566 | success |
| random lite | 256/64/64 | 0.03403 | 0.06070 | 43.93% | 0.6747 | PASS |
| blocked lite | 256/64/64 | 0.02490 | 0.04606 | 45.94% | 0.7013 | PASS |
| random full | 4200/600/1200 | 0.001384 | 0.05924 | 97.66% | 0.9994 | PASS |
| blocked full | 4200/600/1200 | 0.001890 | 0.06086 | 96.89% | 0.9990 | PASS |

Interpretation:

The corrected local GPU loop proves that the refactored future-slip forecasting task is learnable on small overfit, lightweight random/blocked checks, and full 6000-event random/blocked protocols. The plotted examples show the model tracking future moment growth and final slip structure substantially better than persistence on held-out events.

Full-training notes:

- Full training used `segmented_residual`, `forecast_start=60`, `forecast_horizon=50`, batch size `16`, `50` epochs, AMP, and `train_eval_max_batches=32`.
- Validation and test metrics are full-split metrics; train metrics are intentionally capped to avoid memory-heavy full-train evaluation.
- Current results are for the compressed synthetic SSE package and do not validate real earthquake prediction or operational warning.
- The unusually high full-split scores are plausible for this synthetic setting with history slip available, but they should be followed by stricter generalization checks: event-family holdout, noise/station ablation, GNSS-only auxiliary inversion, and a target-leakage audit.
