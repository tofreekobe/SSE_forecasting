## Main Full-Data Results

| Split | Model | h50 RMSE | Persistence | Gain | R2 | M0 rel abs | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| blocked | segmented_residual | 0.00152 | 0.060857 | 97.50% | 0.999359 | 0.016679 | PASS |
| random | segmented_residual | 0.001422 | 0.059236 | 97.60% | 0.999408 | 0.015565 | PASS |

## Ablation Results

| Run | Split | Model | Input | M0 loss | h50 RMSE | Gain | R2 | M0 rel abs | Gate |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ablate_gnss_only | blocked | segmented_residual | gnss_only | 0.005 | 0.061813 | -1.57% | -0.059933 | 1 | FAIL |
| ablate_gnss_only | random | segmented_residual | gnss_only | 0.005 | 0.060166 | -1.57% | -0.060367 | 1 | FAIL |
| ablate_last_slip_only | blocked | segmented_residual | last_slip_only | 0.005 | 0.003041 | 95.00% | 0.997434 | 0.032517 | PASS |
| ablate_last_slip_only | random | segmented_residual | last_slip_only | 0.005 | 0.002089 | 96.47% | 0.998722 | 0.060937 | PASS |
| ablate_no_gnss | blocked | segmented_residual | no_gnss | 0.005 | 0.001927 | 96.83% | 0.99897 | 0.01731 | PASS |
| ablate_no_gnss | random | segmented_residual | no_gnss | 0.005 | 0.002351 | 96.03% | 0.998382 | 0.027774 | PASS |
| ablate_no_m0_loss | blocked | segmented_residual | full | 0 | 0.002783 | 95.43% | 0.997852 | 0.107937 | PASS |
| ablate_no_m0_loss | random | segmented_residual | full | 0 | 0.001121 | 98.11% | 0.999632 | 0.059485 | PASS |
| model_plain_full | blocked | plain | full | 0.005 | 0.005755 | 90.54% | 0.990813 | 0.02816 | PASS |
| model_plain_full | random | plain | full | 0.005 | 0.004315 | 92.72% | 0.994546 | 0.013652 | PASS |
| model_segmented_full | blocked | segmented | full | 0.005 | 0.005065 | 91.68% | 0.992884 | 0.032376 | PASS |
| model_segmented_full | random | segmented | full | 0.005 | 0.004737 | 92.00% | 0.993428 | 0.023861 | PASS |

## Auto Notes

- Main model best split result is `random` with h50 RMSE `0.001422`.
- Best completed ablation so far is `ablate_no_m0_loss/random` with h50 RMSE `0.001121`.
- Treat this file as generated evidence; interpretive claims should still be checked against the full experiment context.
