# GitHub References For SSE Forecasting Figures

These repositories are references for structure, evaluation style, or plotting conventions. They are not copied into this project.

## Time-Series Forecasting Baselines

- `thuml/Time-Series-Library`: useful as a benchmark organization reference because it separates data providers, experiment runners, model implementations, scripts, and utilities, and covers forecasting, imputation, anomaly detection, and classification.
  https://github.com/thuml/Time-Series-Library

- `amazon-science/chronos-forecasting`: useful as a modern pretrained time-series forecasting reference. Its README documents direct multi-step probabilistic forecasting interfaces and covariate-informed variants.
  https://github.com/amazon-science/chronos-forecasting

- `google-research/timesfm`: useful as a time-series foundation model reference. Its README frames TimesFM as a pretrained forecasting model and documents current model versions, horizons, quantile forecasts, and fine-tuning examples.
  https://github.com/google-research/timesfm

## GNSS / Geodetic Plotting

- `kmaterna/GNSS_TimeSeries_Viewers`: useful for GNSS time-series plotting conventions, station stack views, offsets, seasonal terms, and slope visualizations.
  https://github.com/kmaterna/GNSS_TimeSeries_Viewers

## Local Design Consequence

For this SSE project, the right figure set is not a generic Transformer benchmark figure. It should show:

- the corrected two-subfault geometry;
- history/future data contract;
- random and blocked split metrics;
- persistence and mean baselines at h50;
- event-level moment curves and final slip maps;
- explicit limits that the current trained checkpoint is forecasting, not GNSS-only inversion.
