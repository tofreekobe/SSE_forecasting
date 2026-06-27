# Full Dataset Package Audit

Date: 2026-06-27

Purpose: verify that the compressed training package used by the refactored
forecasting experiments is a faithful, full-event representation of the raw
74.2 GiB SSE catalog.

## Result

The package is full and lossless for the training contract:

- Raw event count: `6000`
- Package manifest event count: `6000`
- Raw event IDs: continuous `1..6000`
- Raw data size: `79,673,958,000` bytes (`74.202 GiB`)
- Compressed package size after manifest repair: `3,047,677,208` bytes (`2.838 GiB`)
- Full raw-vs-package comparisons: `6000`
- Exact comparison failures: `0`
- Per-event raw shape: `[273, 3040]`
- Per-event package shapes: slip `[273, 3030]`, GNSS `[273, 9]`

The smaller package size is therefore expected compression, not a reduced
dataset. The package stores the slip and GNSS columns needed by the forecasting
contract and omits the metadata/time column already represented in the manifest.

## Correction Made

The original `manifest.csv` contained a wrong `shard_index`: it stored the shard
number for every event in a shard instead of the event's local index within that
shard. Current training was not affected because it iterates shard event IDs and
filters by `event_id`, but direct manifest-index lookup and future tooling could
be wrong.

Corrections:

- fixed the package builder in `src/diagnostics/hf_sse_diagnostics.py`;
- repaired the local `hf_dataset_package/manifest.csv` and `manifest.jsonl`;
- added `scripts/audit_full_dataset_package.py` for repeatable package repair
  and raw-vs-package verification.

## Verification Command

```powershell
.\.venv-cu128\Scripts\python.exe scripts\audit_full_dataset_package.py `
  --data-dir data `
  --package-dir hf_dataset_package `
  --mode all `
  --output-json diagnostics_full_local\full_dataset_package_audit_all.json `
  --output-md diagnostics_full_local\full_dataset_package_audit_all.md
```

The latest full audit output is stored locally under
`diagnostics_full_local/full_dataset_package_audit_all.*`.
