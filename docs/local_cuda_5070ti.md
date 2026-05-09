# Local RTX 5070 Ti CUDA Setup

## Status

The project now has a local virtual environment at:

```powershell
.venv-cu128
```

It uses:

- Python `3.13.5`
- PyTorch `2.11.0+cu128`
- CUDA runtime `12.8`
- GPU: `NVIDIA GeForce RTX 5070 Ti`
- CUDA capability: `sm_120`

The previous system environment used `torch 2.6.0+cu124`, which only included kernels up to `sm_90`. That is why CUDA was visible but training failed with:

```text
no kernel image is available for execution on the device
```

## Activate

```powershell
.\.venv-cu128\Scripts\Activate.ps1
```

If PowerShell blocks activation, run commands through the venv Python directly:

```powershell
.\.venv-cu128\Scripts\python.exe scripts\pai_check_env.py --package-dir hf_dataset_package
```

## Verify CUDA

```powershell
.\.venv-cu128\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list()); x=torch.randn(2048,2048,device='cuda'); y=x@x; torch.cuda.synchronize(); print('cuda ok')"
```

Expected essentials:

```text
2.11.0+cu128
12.8
NVIDIA GeForce RTX 5070 Ti
(12, 0)
... 'sm_120' ...
cuda ok
```

## Run Local GPU Smoke

```powershell
.\.venv-cu128\Scripts\python.exe scripts\run_small_overfit.py `
  --package-dir hf_dataset_package `
  --output-dir small_overfit_local_cuda_smoke `
  --max-events 4 `
  --forecast-start 60 `
  --forecast-horizon 50 `
  --epochs 3 `
  --batch-size 2 `
  --hidden-channels 8 `
  --model-type segmented `
  --device cuda `
  --lr 0.001 `
  --log-every 1
```

## Recommended Local Small Training

Use the local GPU for fast iteration, not final publication runs:

```powershell
.\.venv-cu128\Scripts\python.exe scripts\run_small_overfit.py `
  --package-dir hf_dataset_package `
  --output-dir small_overfit_local_cuda `
  --max-events 16 `
  --forecast-start 60 `
  --forecast-horizon 50 `
  --epochs 180 `
  --batch-size 8 `
  --hidden-channels 64 `
  --model-type segmented `
  --device cuda `
  --lr 0.001 `
  --log-every 10
```

For larger random/blocked experiments, prefer PAI-DLC/DSW.

## Recreate Environment

```powershell
python -m venv .venv-cu128
.\.venv-cu128\Scripts\python.exe -m pip install --upgrade pip
.\.venv-cu128\Scripts\python.exe -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv-cu128\Scripts\python.exe -m pip install -r requirements-pai.txt
```
