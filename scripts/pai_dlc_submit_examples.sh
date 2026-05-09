#!/usr/bin/env bash
set -euo pipefail

# Fill these IDs from the PAI console before running in DSW Terminal.
WORKSPACE_ID="${WORKSPACE_ID:-1051963}"
DATA_SOURCE_ID="${DATA_SOURCE_ID:-data-REPLACE_ME}"
WORKER_SPEC="${WORKER_SPEC:-ecs.gn7i-c8g1.2xlarge}"
IMAGE_URI="${IMAGE_URI:-dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai/modelscope:1.28.0-pytorch2.3.1tensorflow2.16.1-gpu-py311-cu121-ubuntu22.04}"

# If you use a dedicated resource quota, set RESOURCE_ID and append:
#   --resource_id="${RESOURCE_ID}"

dlc submit pytorchjob \
  --name=sse-random-h50 \
  --workspace_id="${WORKSPACE_ID}" \
  --workers=1 \
  --worker_spec="${WORKER_SPEC}" \
  --worker_image="${IMAGE_URI}" \
  --data_sources="${DATA_SOURCE_ID}" \
  --command="cd /mnt/data/sse && python -m pip install -r requirements-pai.txt && python scripts/pai_check_env.py --package-dir /mnt/data/hf_dataset_package && python scripts/train_forecast_model.py --package-dir /mnt/data/hf_dataset_package --output-dir /mnt/data/sse_outputs/forecast_training_results --protocol random --forecast-start 60 --forecast-horizon 50 --epochs 50 --batch-size 16 --num-workers 4 --hidden-channels 64 --model-type segmented --device cuda --lr 0.001 --active-weight 1.0 --m0-loss-weight 0.01 --amp --tensorboard-dir /mnt/data/sse_outputs/tensorboard/random --log-every 1"

dlc submit pytorchjob \
  --name=sse-blocked-h50 \
  --workspace_id="${WORKSPACE_ID}" \
  --workers=1 \
  --worker_spec="${WORKER_SPEC}" \
  --worker_image="${IMAGE_URI}" \
  --data_sources="${DATA_SOURCE_ID}" \
  --command="cd /mnt/data/sse && python -m pip install -r requirements-pai.txt && python scripts/pai_check_env.py --package-dir /mnt/data/hf_dataset_package && python scripts/train_forecast_model.py --package-dir /mnt/data/hf_dataset_package --output-dir /mnt/data/sse_outputs/forecast_training_results --protocol blocked --forecast-start 60 --forecast-horizon 50 --epochs 50 --batch-size 16 --num-workers 4 --hidden-channels 64 --model-type segmented --device cuda --lr 0.001 --active-weight 1.0 --m0-loss-weight 0.01 --amp --tensorboard-dir /mnt/data/sse_outputs/tensorboard/blocked --log-every 1"
