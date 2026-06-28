# SSE Slow Slip Forecasting

Clean refactor of the slow slip event forecasting project.

The current pipeline treats the old project as a prototype evidence base and rebuilds a reproducible training loop:

```text
data contract -> diagnostics -> baselines -> small overfit -> full train -> inference/report
```

## Current Decision

Diagnostics support `GO_WITH_CHANGES`:

- the 6000-event catalog is structurally usable;
- slip/GNSS contain learnable signal;
- the old `z-score slip target + softplus output` design is invalid;
- the primary task is now 50-step future slip forecasting;
- inversion and GNSS reconstruction are auxiliary evidence, not the main claim.

## Repository Scope

This repository tracks code, tests, and documentation only.

It intentionally excludes:

- raw `data/`;
- compressed `hf_dataset_package/`;
- local papers and manuscript drafts in `paper/`;
- trained models, checkpoints, diagnostics, and experiment outputs;
- local virtual environments such as `.venv-cu128/`.

Use the private Hugging Face dataset or the local package directory for data access.

## Key Commands

Install project dependencies:

```powershell
python -m pip install -r requirements-pai.txt
```

Run contract and model tests:

```powershell
python -m pytest -q
```

Run a local small-overfit check:

```powershell
python scripts\run_small_overfit.py `
  --package-dir hf_dataset_package `
  --output-dir small_overfit_results `
  --max-events 16 `
  --forecast-start 60 `
  --forecast-horizon 50 `
  --model-type segmented_residual `
  --device auto
```

Run formal training:

```powershell
python scripts\train_forecast_model.py `
  --package-dir hf_dataset_package `
  --output-dir forecast_training_results `
  --protocol random `
  --forecast-start 60 `
  --forecast-horizon 50 `
  --model-type segmented_residual
```

## PAI

See [docs/pai_training.md](docs/pai_training.md) for DSW and DLC usage.

## Local RTX 5070 Ti

See [docs/local_cuda_5070ti.md](docs/local_cuda_5070ti.md) for the local `torch 2.11.0+cu128` setup.

## Research Notes

See [research_notes/pre_refactor_literature_review.md](research_notes/pre_refactor_literature_review.md) for the pre-refactor literature review and final research positioning.

## Current Full-Data Evidence

- Full raw/package audit: [docs/full_dataset_package_audit.md](docs/full_dataset_package_audit.md)
- DSW main training results: [docs/dsw_main_training_results_zh.md](docs/dsw_main_training_results_zh.md)
- Final conference paper outline: [docs/final_conference_paper_outline_zh.md](docs/final_conference_paper_outline_zh.md)
- Final paper manuscript draft: [docs/final_paper_manuscript_zh.md](docs/final_paper_manuscript_zh.md)
- Core paper section draft: [docs/final_paper_core_sections_draft_zh.md](docs/final_paper_core_sections_draft_zh.md)
- Paper-ready result tables: [docs/paper_result_tables_current.md](docs/paper_result_tables_current.md)
- Literature matrix: [docs/literature_matrix_zh.md](docs/literature_matrix_zh.md)
- Complete Chinese usage guide: [docs/complete_usage_guide_zh.md](docs/complete_usage_guide_zh.md)
- Forecast/inversion demo GUI usage: [docs/model_demo_usage.md](docs/model_demo_usage.md)
- GitHub publish guide: [docs/github_publish_guide_zh.md](docs/github_publish_guide_zh.md)
