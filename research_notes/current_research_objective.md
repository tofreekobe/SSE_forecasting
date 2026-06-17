# Current Research Objective

## Research Goal

Build a geometry-aware forecasting pipeline for synthetic slow slip event evolution.

The primary task is:

```text
Given history GNSS + history slip, predict 50 future steps of fault slip.
```

The defensible research claim is intentionally narrow:

- sparse three-station GNSS setting;
- two disconnected subfaults with known geometry;
- future slip-field forecasting in physical units;
- persistence/mean/zero baselines at `h=1/5/10/30/50`;
- random and blocked split evaluation.

The project does not currently claim real earthquake prediction, operational early warning, or real-data validated forecasting.

## Current Default Model

Default model type:

```text
segmented_residual
```

Design:

- split the 3030 slip vector into segment 1 (`15x166`) and segment 2 (`15x36`);
- run separate residual CNN branches for the two disconnected fault segments;
- encode GNSS history once and broadcast the latent features into both segment branches;
- predict future slip as a non-negative residual evolution from the last observed slip state;
- decode predictions back through `expm1(encoded) * slip_scale` for all physical metrics.

Why this replaces the earlier default:

- the old rectangular `15x202` convolution created artificial adjacency across the two-subfault boundary;
- plain segment CNN passed the small-overfit gate, but residual blocks and normalization are a stronger default for full training;
- `plain` and `segmented` are still available for ablation.

## Training Gates

Before paper-facing claims:

- random h50 test RMSE must improve over persistence by at least `5%`;
- blocked h50 test RMSE must improve over persistence by at least `2%`;
- M0 relative absolute error must not worsen by more than `10%` versus persistence.

Small overfit remains the sanity gate before full training. Full training should be launched only after:

- package check passes;
- small overfit can beat persistence clearly;
- random and blocked protocols both run to completion.
