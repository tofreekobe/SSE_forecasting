# HF Diagnostics Workflow

This workflow implements the pre-refactor feasibility gate for the SSE project.
It does not modify the training code.

## 1. Build the HF package

For a quick smoke package:

```bash
python scripts/prepare_hf_dataset.py --data-dir data --output-dir hf_dataset_smoke --max-events 8
```

For the full private package:

```bash
python scripts/prepare_hf_dataset.py --data-dir data --output-dir hf_dataset_package --shard-size 32
```

The package contains:

- `manifest.csv`, `manifest.jsonl`, and optional `manifest.parquet`
- `events/shard_*.npz`
- `normalization_stats.json` if present in `data/`
- `dataset_metadata.json`

## 2. Upload to private HF dataset repo

```bash
pip install -r requirements-diagnostics.txt
python scripts/upload_hf_dataset.py \
  --package-dir hf_dataset_package \
  --repo-id tofreekobe/sse-slow-slip-private
```

## 3. Run diagnostics locally or remotely

Local:

```bash
python scripts/run_hf_diagnostics.py --input-dir hf_dataset_package --output-dir diagnostics/hf_local
```

Remote HF dataset:

```bash
python scripts/run_hf_diagnostics.py \
  --repo-id tofreekobe/sse-slow-slip-private \
  --output-dir diagnostics/hf_remote
```

HF Jobs command:

```bash
python scripts/launch_hf_diagnostics_job.py \
  --dataset-repo tofreekobe/sse-slow-slip-private \
  --diagnostics-repo tofreekobe/sse-slow-slip-diagnostics
```

Add `--run` after checking the printed command.

## 4. Required outputs

Diagnostics write:

- `hf_diagnostics.json`
- `hf_diagnostics.md`
- `baseline_metrics.csv`
- `baseline_rmse.png`
- `slip_sum_distribution.png`
- `legacy_zscore_conflict.png`

The go/no-go conclusion is in `hf_diagnostics.json` under `gate.conclusion`.

