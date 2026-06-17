# PAI GPU Training Plan for SSE Forecasting

## Decision

Use local IDE + SSH to PAI-DSW first, then PAI-DLC if long-running jobs need to be detached.

- DSW is the interactive GPU development machine: use SSH from the local terminal/IDE to verify CUDA, package paths, small overfit, and one short training run.
- Do not switch to the browser IDE unless SSH cannot be enabled.
- DLC is the managed training job runner: use it after the exact DSW command works, so random and blocked split runs can continue without keeping an IDE session alive.
- EAS is not needed now. It is for serving a trained model, not for research iteration.

Official references used:

- DSW overview: https://help.aliyun.com/zh/pai/user-guide/dsw-overview
- Create DSW instance: https://help.aliyun.com/zh/pai/user-guide/create-and-manage-dsw-instances
- DLC quickstart: https://help.aliyun.com/zh/pai/getting-started/distributed-training-dlc-quickstart
- DLC submit command: https://help.aliyun.com/zh/pai/developer-reference/commands-used-to-submit-jobs
- PAI SDK Estimator: https://help.aliyun.com/zh/pai/developer-reference/submit-a-training-job
- DSW SSH direct connection: `C:\Users\Administrator\Desktop\王一帆项目文件\mm\阿里云PAI的SSH.md`

## SSH-First Access Plan

Use SSH direct connection from the local machine to the DSW instance.

Preferred flow:

1. Enable SSH on the DSW instance.
2. Add the local public key to the DSW SSH public key field.
3. Enable public access if connecting from this local Windows machine outside the VPC.
4. Connect from local PowerShell with `ssh`.
5. Upload or clone the code into the mounted DSW workspace.
6. Run `pai_check_env.py`, small overfit, then random/blocked training from SSH.

Public SSH form:

```powershell
ssh -i C:\Users\Administrator\.ssh\pai_dsw_rsa root@<PUBLIC_EIP_OR_HOST> -p <PUBLIC_SSH_PORT>
```

VPC-only SSH form, usable only from an ECS/VPN/client already inside the same VPC:

```bash
ssh -i ~/.ssh/pai_dsw_rsa root@<DSW_INTERNAL_DOMAIN> -p 22
```

If the key is stored in the default path and has no passphrase, `-i` can be omitted.

## Information Needed From The User

Please provide or confirm these items before I can connect and run training through SSH:

1. SSH access mode:
   - `public`: local Windows machine connects through EIP/NAT;
   - `vpc`: connection is made from an ECS/VPN host inside the same VPC.
2. SSH host:
   - public mode: EIP or public access host shown in DSW instance details;
   - VPC mode: internal DSW domain such as `dsw-notebook-xxxx...`.
3. SSH port:
   - public mode: the DSW public access port, often not `22`, for example `1024`;
   - VPC mode: usually `22`.
4. Login user:
   - usually `root` for DSW SSH.
5. SSH key choice:
   - either let me generate `C:\Users\Administrator\.ssh\pai_dsw_rsa` and you paste `pai_dsw_rsa.pub` into DSW;
   - or provide the local private key path that already matches the public key configured in DSW.
6. DSW instance/network confirmation:
   - SSH is enabled;
   - listen port is `22`;
   - public access is enabled if using public mode;
   - NAT Gateway and EIP are configured for public mode;
   - security group inbound allows TCP 22 for the instance side.
7. Remote storage paths:
   - code path, default `/mnt/data/sse`;
   - dataset package path, default `/mnt/data/hf_dataset_package`;
   - output path, default `/mnt/data/sse_outputs`.
8. Data transfer choice:
   - data already mounted/uploaded on DSW;
   - or I should upload `hf_dataset_package` from this machine through `scp`;
   - or DSW should download it from the private HF dataset repo.
9. Remote Python environment:
   - DSW image name if known;
   - Python command if not default `python`;
   - whether CUDA PyTorch is already installed.

Cost note: public SSH requires NAT Gateway + EIP. These may keep billing even when DSW is stopped, so delete or release them when SSH access is no longer needed.

## Current DSW SSH Notes

Current intended access mode: VPC internal SSH.

Known instance details:

- Region: `cn-shanghai`
- Instance name: `SSE01`
- Instance ID: `dsw-rno9xx3bb9epj8rf2p`
- Resource: `ecs.gn7i-c8g1.2xlarge`
- GPU: `NVIDIA A10`, 1 card
- CPU: 8 cores
- Memory: 30 GiB
- Image: `dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai/pytorch:2.8.0-gpu-py312-cu126-ubuntu22.04-3995b779-1764358175`
- SSH listen port inside DSW: `22`
- Configured SSH public key begins with: `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5... administrator@Server-for-Xifan`

Important status:

- The DSW instance can access the internet through a public gateway.
- The SSH custom service has listen port `22`.
- The public SSH entry is not required if an ECS/VPN/client inside the same VPC will connect to DSW.
- The image URL is a container image registry URL, not an SSH host.

Before VPC internal SSH can work, the connection must originate from a machine inside the same VPC, such as an ECS instance, a VPN-connected local machine, or another approved VPC client. A normal local Windows machine on the public internet cannot directly resolve or reach the DSW VPC-only endpoint.

Do not confuse these ports:

- Listen port: `22`, inside the DSW instance.
- Public access port: generated by PAI/NAT, used by local `ssh -p`; often a high port such as `1024`.

Pending information:

- VPC SSH host/domain from the DSW instance details, for example `dsw-notebook-xxxx...`.
- Confirmation that the client we will use is inside the same VPC:
  - ECS jump host public SSH info; or
  - VPN/CEN/local route into the VPC; or
  - confirmation that commands will be run from another terminal already inside the VPC.
- Local private key path that matches the configured public key, or add a new public key generated on this machine.

Recommended data transfer for this instance:

1. If an ECS jump host or VPN route exists, use `scp` through the VPC path.
2. Upload the small code bundle first.
3. Upload `hf_dataset_package` second, preferably as a tar archive or with recursive `scp`.
4. If no VPC route from the local machine exists, download the private HF dataset from inside DSW as the fallback. This is operationally simpler than setting up a public inbound SSH endpoint, but may be slower from a China-region instance and requires setting a fresh HF token inside DSW.

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

## DSW Commands Over SSH

After SSH login and after uploading/unzipping the code into `/mnt/data/sse`:

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
  --model-type segmented_residual \
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
  --model-type segmented_residual \
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

## Local Upload Commands

If the project has not been uploaded to DSW yet and public SSH works:

```powershell
scp -i C:\Users\Administrator\.ssh\pai_dsw_rsa -P <PUBLIC_SSH_PORT> pai_sse_code_bundle.zip root@<PUBLIC_EIP_OR_HOST>:/mnt/data/
ssh -i C:\Users\Administrator\.ssh\pai_dsw_rsa root@<PUBLIC_EIP_OR_HOST> -p <PUBLIC_SSH_PORT> "cd /mnt/data && unzip -o pai_sse_code_bundle.zip -d sse"
```

If using VPC internal SSH through an ECS jump host:

```powershell
scp -i C:\Users\Administrator\.ssh\pai_dsw_rsa -o ProxyJump=root@<ECS_PUBLIC_IP>:22 pai_sse_code_bundle.zip root@<DSW_INTERNAL_DOMAIN>:/mnt/data/
ssh -i C:\Users\Administrator\.ssh\pai_dsw_rsa -J root@<ECS_PUBLIC_IP>:22 root@<DSW_INTERNAL_DOMAIN> "cd /mnt/data && unzip -o pai_sse_code_bundle.zip -d sse"
```

If the dataset package is not mounted on DSW and must be uploaded from this machine, expect a large transfer:

```powershell
scp -i C:\Users\Administrator\.ssh\pai_dsw_rsa -P <PUBLIC_SSH_PORT> -r hf_dataset_package root@<PUBLIC_EIP_OR_HOST>:/mnt/data/
```

With an ECS jump host:

```powershell
scp -i C:\Users\Administrator\.ssh\pai_dsw_rsa -o ProxyJump=root@<ECS_PUBLIC_IP>:22 -r hf_dataset_package root@<DSW_INTERNAL_DOMAIN>:/mnt/data/
```

If using Hugging Face download from inside DSW instead of `scp`:

```bash
cd /mnt/data
python -m pip install -U huggingface_hub
export HF_TOKEN=<fresh_private_dataset_token>
hf download tofreekobe/sse-slow-slip-private --repo-type dataset --local-dir /mnt/data/hf_dataset_package
```

Do not commit or write the HF token into project files.

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
