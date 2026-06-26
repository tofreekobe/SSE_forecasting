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
- Summary: `forecast_training_5070ti_lite/training_summary.md`

Metrics:

| Run | Events | Test h50 RMSE | Persistence h50 RMSE | Improvement | R2 | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| small overfit | 16 train | 0.02282 | 0.06147 | 62.88% | 0.8566 | success |
| random lite | 256/64/64 | 0.03403 | 0.06070 | 43.93% | 0.6747 | PASS |
| blocked lite | 256/64/64 | 0.02490 | 0.04606 | 45.94% | 0.7013 | PASS |

Interpretation:

The corrected local GPU loop proves that the refactored future-slip forecasting task is learnable on both a small overfit gate and lightweight random/blocked train-validation-test checks. The plotted examples show the model tracking future moment growth and final slip structure substantially better than persistence on selected held-out events.
