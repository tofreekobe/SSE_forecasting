# Web Literature Update: SSE Forecasting Position

Date: 2026-06-28

This note updates the local pre-refactor review with external sources checked
online before final paper restructuring. The practical conclusion is unchanged:
the strongest defensible contribution is not generic earthquake prediction, but
geometry-aware multi-step slip-field forecasting on a full synthetic SSE
catalog, evaluated in physical units against persistence/mean/zero baselines.

## SSE-Specific Deep Learning

- [Multi-station deep learning on geodetic time series detects slow slip events
  in Cascadia](https://www.nature.com/articles/s43247-023-01107-7) presents an
  end-to-end synthetic-generator plus CNN/Transformer detector for raw
  multi-station GNSS. This validates the use of synthetic geodetic data and
  deep spatiotemporal models for SSE analysis, but the task is event detection
  rather than future fault-slip forecasting.
- [Denoising of Geodetic Time Series Using Spatiotemporal Graph Neural
  Networks](https://arxiv.org/abs/2405.03320) introduces SSEdenoiser, a
  graph-recurrent/Transformer denoising pipeline for multi-station GNSS. It is
  adjacent to our GNSS signal-learning motivation, but its output is denoised
  displacement, not a future slip field.
- [Detecting slow slip events in the Cascadia subduction zone from GNSS time
  series using deep learning](https://doi.org/10.1007/s10291-024-01701-y) uses
  vbICA, BiLSTM, and attention mechanisms for SSE detection from GNSS time
  series. This supports deep sequence modeling for SSE signals while reinforcing
  that detection is already comparatively well explored.
- [Earthquake alerting based on spatial geodetic data by spatiotemporal
  information transformation learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10500272/)
  is useful motivation for spatial geodetic learning, but it belongs to a
  real-time earthquake alerting claim space. The SSE project should not borrow
  operational warning language unless a separate, validated warning system is
  built.

## General Time-Series Model References

- [TimesNet](https://openreview.net/forum?id=ju_Uqw384Oq) motivates converting
  temporal variation into structured 2D representations, but our current
  dominant structure is fault geometry, so the first publishable model should
  remain geometry-aware rather than importing a generic benchmark backbone.
- [PatchTST](https://arxiv.org/abs/2211.14730) and
  [iTransformer](https://openreview.net/forum?id=JePfAI8fah&noteId=lHITGrmunH)
  remain reasonable later baselines for sequence modeling, especially for
  GNSS-only or low-slip-history settings.
- [TimesFM](https://github.com/google-research/timesfm) and
  [Chronos](https://github.com/amazon-science/chronos-forecasting) are valuable
  references for zero-shot/general forecasting, but they do not directly solve
  structured fault-slip field prediction with physical metrics. They should be
  cited as context or optional future baselines, not as the main method.
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library) is the
  best practical repository reference if later we add TimesNet, PatchTST, or
  iTransformer baselines under the same data contract.

## Implications For This Project

1. Keep the paper title and abstract explicit about synthetic SSE slip-field
   forecasting. Do not imply real-world earthquake prediction.
2. Keep the primary task fixed as `history_gnss + history_slip -> 50-step
   future_slip`; treat inversion and GNSS reconstruction as auxiliary evidence.
3. Emphasize what the above SSE papers do not center: direct multi-step
   forecasting of a two-subfault slip field with physical-unit metrics.
4. Keep the default model geometry-aware (`segmented_residual`) and use generic
   time-series models only as comparison baselines if time permits.
5. Report the data scope correctly: the raw catalog is 6000 events and
   74.202 GiB; the training package is a full, audited compressed
   representation, not a reduced dataset.

