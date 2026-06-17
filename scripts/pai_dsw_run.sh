#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/data/sse}"
PACKAGE_DIR="${PACKAGE_DIR:-/mnt/data/hf_dataset_package}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/sse_outputs}"

cd "${PROJECT_DIR}"

python -m pip install -r requirements-pai.txt
python scripts/pai_check_env.py --package-dir "${PACKAGE_DIR}"

python scripts/run_small_overfit.py \
  --package-dir "${PACKAGE_DIR}" \
  --output-dir "${OUTPUT_DIR}/small_overfit" \
  --max-events 16 \
  --forecast-start 60 \
  --forecast-horizon 50 \
  --epochs 180 \
  --batch-size 8 \
  --hidden-channels 64 \
  --model-type segmented_residual \
  --device cuda \
  --lr 0.001 \
  --log-every 10

python scripts/train_forecast_model.py \
  --package-dir "${PACKAGE_DIR}" \
  --output-dir "${OUTPUT_DIR}/forecast_training_results" \
  --protocol random \
  --forecast-start 60 \
  --forecast-horizon 50 \
  --epochs 50 \
  --batch-size 16 \
  --num-workers 4 \
  --hidden-channels 64 \
  --model-type segmented_residual \
  --device cuda \
  --lr 0.001 \
  --active-weight 1.0 \
  --m0-loss-weight 0.01 \
  --amp \
  --tensorboard-dir "${OUTPUT_DIR}/tensorboard/random" \
  --log-every 1

python scripts/train_forecast_model.py \
  --package-dir "${PACKAGE_DIR}" \
  --output-dir "${OUTPUT_DIR}/forecast_training_results" \
  --protocol blocked \
  --forecast-start 60 \
  --forecast-horizon 50 \
  --epochs 50 \
  --batch-size 16 \
  --num-workers 4 \
  --hidden-channels 64 \
  --model-type segmented_residual \
  --device cuda \
  --lr 0.001 \
  --active-weight 1.0 \
  --m0-loss-weight 0.01 \
  --amp \
  --tensorboard-dir "${OUTPUT_DIR}/tensorboard/blocked" \
  --log-every 1
