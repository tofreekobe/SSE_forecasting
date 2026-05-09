# Pre-Refactor Literature Review and Research Position

## Local Materials Reviewed

- `paper/1128-draft-wyf.docx`: previous manuscript draft.
- `paper/outline.pdf`: earlier broader GeosenseNet outline.
- `paper/Denoising of Geodetic Time Series Using.pdf`: SSEdenoiser / STGNN denoising reference.
- `paper/Detecting slow slip events in the Cascadia subduction zone from GNSS.pdf`: vbICA + BiLSTM + attention SSE detection reference.
- `paper/tong-et-al-earthquake-alerting-based-on-spatial-geodetic-data-by-spatiotemporal-information-transformation-learning.pdf`: RSIT geodetic earthquake alerting reference.
- `paper/微信图片_20251203133834_256_65.png` and `paper/微信图片_20251205083122_266_65.jpg`: two-subfault geometry and three-GNSS-station visualization.

Extracted text is stored in:

- `research_notes/local_paper_extracts.json`
- `research_notes/local_paper_extracts_full.json`

## What The Literature Already Does

1. SSE detection from GNSS is now an active and credible research line.
   Costantino et al. use a realistic synthetic generator plus CNN/Transformer detector on raw Cascadia GNSS and detect 78 SSEs from 2007-2022, retrieving 87.5% of cataloged events and aligning detections with tremor peaks.

2. Single-station and denoising pipelines are improving quickly.
   Wang et al. combine vbICA with BiLSTM and channel/spatial attention, detecting 56 Cascadia SSEs from 2012-2022. Their own limitation is that the method detects events but does not directly infer the source/slip distribution.

3. Spatiotemporal graph denoising is relevant but adjacent, not identical.
   SSEdenoiser learns GNSS network structure and extracts hidden SSE-related displacement from noisy non-postprocessed GNSS. Its output can improve downstream slip inversion, but it is not itself a slip forecasting model.

4. GNSS-to-slip neural inversion is credible in coseismic and short-term SSE contexts.
   Recent coseismic work shows neural networks can estimate fault slip distributions from GNSS much faster than conventional inversion. Nakagawa et al. also report a deep-learning approach for estimating short-term SSE slip area and amount from dense GNSS-derived deformation images.

5. Real-time earthquake alerting from GNSS is a neighboring motivation, not our core claim.
   RSIT transforms high-dimensional GNSS observations into lower-dimensional dynamics and uses unpredictability/instability criteria for early warning. This supports the value of spatiotemporal GNSS learning, but our project should not claim earthquake prediction.

## Research Gap For This Project

The strongest defensible gap is:

> Existing deep-learning SSE studies mainly detect whether/when an SSE occurs or denoise GNSS time series. Much less work directly learns time-dependent fault slip fields from sparse local GNSS, and even less evaluates future slip-field forecasting under a physically defined two-subfault geometry.

This makes the project meaningful if framed as:

- synthetic but geometry-specific SSE slip-field forecasting and inversion;
- sparse three-station GNSS setting;
- two disconnected subfaults with known geometry;
- physical-unit metrics and baselines;
- honest limits about transfer to real data.

## Corrections To The Previous Manuscript Claims

The previous draft is useful as motivation, but several claims must not be carried forward until reproduced:

- Do not claim hard-coded RMSE/R2/ablation improvements from old experiments.
- Do not claim RSF constraints unless parameters and equations are actually implemented and validated.
- Do not claim earthquake early warning or secondary-disaster prediction.
- Do not claim simultaneous inversion and forecasting until both losses and evaluations are present.
- Do not treat all 3030 subfaults as one continuous 15x202 image; the figures show two disconnected subfaults.

## Scheme Optimization Before Full Refactor

1. Make the primary task future slip forecasting.
   Keep inversion/GNSS reconstruction as auxiliary evidence after forecasting baseline is beaten.

2. Use a segment-aware fault representation.
   Segment 1 is `15x166`; segment 2 is `15x36`. Convolutions and smoothness losses must not leak across the artificial concatenation boundary.

3. Preserve the corrected target transform.
   Use `log1p(slip / slip_scale)` for training and inverse-transform to physical slip for all reports.

4. Report physically meaningful metrics.
   Always include RMSE, R2, active-region metrics, M0 relative absolute error, and persistence/mean/zero baselines at h=1/5/10/30/50.

5. Use the active event window.
   Keep `forecast_start=60`, `forecast_horizon=50` as the default. A 70% split starts after the active window and creates a trivial plateau task.

6. Add publication gates.
   Random test h50 RMSE must improve over persistence by at least 5%; blocked test by at least 2%; M0 error must not worsen by more than 10% versus persistence.

## Decision

Proceed with the refactor.

The project is meaningful, but only under a narrower and more honest claim:

> a geometry-aware, physically evaluated learning pipeline for forecasting synthetic SSE slip evolution from sparse GNSS and slip history, with inversion/GNSS reconstruction treated as auxiliary rather than the main publication claim.

Immediate code change required before full training:

- Replace the plain rectangular fault CNN with a segment-aware model or mask so the two subfaults are not coupled through an artificial image boundary.

## Refactor Start Result

Implemented the first geometry-aware correction:

- Added `SegmentedSlipConvForecastNet`, which runs separate convolution branches for segment 1 (`15x166`) and segment 2 (`15x36`) and only concatenates them back at the vector output.
- Kept `SlipConvForecastNet` as `--model-type plain` for an explicit ablation.
- Made `--model-type segmented` the default for small overfit and formal training scripts.
- Added `--device auto|cpu|cuda` so local CPU debugging and PAI GPU training can be selected explicitly.

Small-overfit gate on 16 events passed locally on CPU:

- Forecast start: `60`
- Forecast horizon: `50`
- Model type: `segmented`
- Physical RMSE: `0.0240425`
- Persistence h50 RMSE: `0.0614686`
- RMSE / persistence: `0.391134`
- Physical R2: `0.840721`
- M0 relative absolute error: `0.191932`

This confirms the refactored data contract, inverse transform, physical metrics, and segment-aware model all close on a learnable small problem. The next step is PAI full training for random and blocked splits.
