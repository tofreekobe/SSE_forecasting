# PAI GPU Training Plan for SSE Forecasting

## Decision

Use PAI-DSW first, then PAI-DLC.

- DSW is the interactive GPU development machine: use it to verify CUDA, package paths, small overfit, and one short training run.
- DLC is the managed training job runner: use it after the exact DSW command works, so random and blocked split runs can continue without keeping an IDE session alive.
- EAS is not needed now. It is for serving a trained model, not for research iteration.

Official references used:

- DSW overview: https://help.aliyun.com/zh/pai/user-guide/dsw-overview
- Create DSW instance: https://help.aliyun.com/zh/pai/user-guide/create-and-manage-dsw-instances
- DLC quickstart: https://help.aliyun.com/zh/pai/getting-started/distributed-training-dlc-quickstart
- DLC submit command: https://help.aliyun.com/zh/pai/developer-reference/commands-used-to-submit-jobs
- PAI SDK Estimator: https://help.aliyun.com/zh/pai/developer-reference/submit-a-training-job

## Recommended DSW Instance

In region `cn-shanghai`, workspace `1051963`:

- Module: 交互式建模 DSW
- Resource: use the GPU resource covered by the saving plan when available
- First smoke instance: 1 GPU, for example A10-like `ecs.gn7i-c8g1.2xlarge`
- Full iteration instance: prefer a larger single-GPU spec if available; our current model is single-process and benefits most from faster single GPU plus enough CPU I/O
- Image: PyTorch GPU image, preferably `pytorch2.x + cu121/cu124`; the DLC quickstart lists the Shanghai ModelScope image:
  `dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai/modelscope:1.28.0-pytorch2.3.1tensorflow2.16.1-gpu-py311-cu121-ubuntu22.04`
- Mount path: `/mnt/data`

## Data Layout

Put these two directories under the same persistent mount:

```text
/mnt/data/sse/
/mnt/data/hf_dataset_package/
```

`/mnt/data/sse` contains this codebase.
`/mnt/data/hf_dataset_package` contains:

```text
manifest.csv
events/shard_*.npz
dataset_metadata.json
```

Preferred data path is OSS or CPFS mounted into DSW/DLC. If using Hugging Face download instead, do not hard-code tokens in scripts; set a fresh token in the DSW terminal environment.

## DSW Commands

After uploading/unzipping the code into `/mnt/data/sse`:

```bash
cd /mnt/data/sse
python -m pip install -r requirements-pai.txt
python scripts/pai_check_env.py --package-dir /mnt/data/hf_dataset_package
```

Run the small overfit gate:

```bash
python scripts/run_small_overfit.py \
  --package-dir /mnt/data/hf_dataset_package \
  --output-dir /mnt/data/sse_outputs/small_overfit \
  --max-events 16 \
  --forecast-start 60 \
  --forecast-horizon 50 \
  --epochs 180 \
  --batch-size 8 \
  --hidden-channels 64 \
  --model-type segmented \
  --device cuda \
  --lr 0.001 \
  --log-every 10
```

Run the first full random split:

```bash
python scripts/train_forecast_model.py \
  --package-dir /mnt/data/hf_dataset_package \
  --output-dir /mnt/data/sse_outputs/forecast_training_results \
  --protocol random \
  --forecast-start 60 \
  --forecast-horizon 50 \
  --epochs 50 \
  --batch-size 16 \
  --num-workers 4 \
  --hidden-channels 64 \
  --model-type segmented \
  --device cuda \
  --lr 0.001 \
  --active-weight 1.0 \
  --m0-loss-weight 0.01 \
  --amp \
  --tensorboard-dir /mnt/data/sse_outputs/tensorboard/random \
  --log-every 1
```

Then run the blocked split by changing `--protocol blocked` and TensorBoard output path.

Use `--device cpu` only for local debugging on machines where PyTorch can see a GPU but the installed wheel does not support that GPU architecture.

The shortcut script is:

```bash
bash scripts/pai_dsw_run.sh
```

## DLC Console Setup

Use DLC only after the DSW command above works.

Create a PyTorch job:

- Module: 分布式训练 DLC
- Job type: PyTorch
- Worker count: `1`
- Worker image: same PyTorch GPU image used in DSW
- Worker spec: same or larger GPU spec
- Dataset mount: mount the OSS/CPFS dataset to `/mnt/data`
- Start command: use the command from `scripts/pai_dlc_submit_examples.sh`
- TensorBoard summary path: `/mnt/data/sse_outputs/tensorboard/`

For this project, start with single-node single-GPU DLC. Multi-node distributed training is premature until the single-GPU model beats the publication gate.

## Gates

The model is allowed to advance only if:

- Random test h50 RMSE improves over persistence by at least 5%.
- Blocked test h50 RMSE improves over persistence by at least 2%.
- M0 relative absolute error is not more than 10% worse than persistence.

Current local baselines at `forecast_start=60`, `horizon=50`:

- Random test persistence RMSE: `0.059236`
- Blocked test persistence RMSE: `0.060857`
- Persistence M0 relative absolute error: about `0.948833`

## If DLC Cannot Be Called

Run the DSW command directly inside the DSW IDE Terminal. This is slower operationally than DLC but still uses the same GPU instance and writes all artifacts under `/mnt/data/sse_outputs`.
