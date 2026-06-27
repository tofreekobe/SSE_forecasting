param(
    [string]$OutputRoot = "forecast_training_5070ti_full",
    [int]$Epochs = 50,
    [int]$BatchSize = 16,
    [int]$NumWorkers = 0,
    [int]$HiddenChannels = 64,
    [int]$MaxTrainEvents = 0,
    [int]$MaxValEvents = 0,
    [int]$MaxTestEvents = 0,
    [double]$LearningRate = 0.0007,
    [double]$M0LossWeight = 0.005,
    [int]$TrainEvalMaxBatches = 32,
    [int]$PlotExamples = 6
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Root ".venv-cu128\Scripts\python.exe"
$PackageDir = Join-Path $Root "hf_dataset_package"
$OutputDir = Join-Path $Root $OutputRoot

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing local CUDA Python environment: $Python"
}
if (-not (Test-Path -LiteralPath (Join-Path $PackageDir "manifest.csv"))) {
    throw "Missing hf_dataset_package manifest at $PackageDir"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$env:PYTHONUTF8 = "1"
Set-Content -Path (Join-Path $OutputDir "full_train.pid") -Value $PID
"=== $(Get-Date -Format s) launcher pid=$PID ===" | Tee-Object -FilePath (Join-Path $OutputDir "full_train.launch.log")

function Invoke-Protocol {
    param([ValidateSet("random", "blocked")][string]$Protocol)

    $LogPath = Join-Path $OutputDir "$Protocol.full_train.log"
    $LimitArgs = @()
    if ($MaxTrainEvents -gt 0) {
        $LimitArgs += @("--max-train-events", $MaxTrainEvents)
    }
    if ($MaxValEvents -gt 0) {
        $LimitArgs += @("--max-val-events", $MaxValEvents)
    }
    if ($MaxTestEvents -gt 0) {
        $LimitArgs += @("--max-test-events", $MaxTestEvents)
    }
    $TrainArgs = @(
        (Join-Path $Root "scripts\train_forecast_model.py"),
        "--package-dir", $PackageDir,
        "--output-dir", $OutputDir,
        "--protocol", $Protocol,
        "--forecast-start", "60",
        "--forecast-horizon", "50"
    )
    $TrainArgs += $LimitArgs
    $TrainArgs += @(
        "--epochs", $Epochs,
        "--batch-size", $BatchSize,
        "--num-workers", $NumWorkers,
        "--hidden-channels", $HiddenChannels,
        "--model-type", "segmented_residual",
        "--device", "cuda",
        "--lr", $LearningRate,
        "--active-weight", "1.0",
        "--m0-loss-weight", $M0LossWeight,
        "--train-eval-max-batches", $TrainEvalMaxBatches,
        "--amp",
        "--tensorboard-dir", "off",
        "--log-every", "1"
    )
    "=== $(Get-Date -Format s) starting $Protocol full training ===" | Tee-Object -FilePath $LogPath
    $OldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @TrainArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ErrorActionPreference = $OldErrorActionPreference
    if ($LASTEXITCODE -ne 0) {
        throw "$Protocol training failed with exit code $LASTEXITCODE"
    }

    $OldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $RunDir = Join-Path $OutputDir $Protocol
    $PlotArgs = @(
        (Join-Path $Root "scripts\plot_forecast_examples.py"),
        "--run-dir", $RunDir,
        "--package-dir", $PackageDir,
        "--output-dir", (Join-Path $RunDir "figures"),
        "--split", "test",
        "--max-events", $PlotExamples,
        "--device", "cuda"
    )
    & $Python @PlotArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ErrorActionPreference = $OldErrorActionPreference
    if ($LASTEXITCODE -ne 0) {
        throw "$Protocol plotting failed with exit code $LASTEXITCODE"
    }
    "=== $(Get-Date -Format s) finished $Protocol ===" | Tee-Object -FilePath $LogPath -Append
}

Invoke-Protocol -Protocol random
Invoke-Protocol -Protocol blocked

& $Python (Join-Path $Root "scripts\summarize_training_results.py") `
    --small-overfit-dir (Join-Path $Root "small_overfit_5070ti_persist_init_16e240") `
    --training-dir $OutputDir `
    --output (Join-Path $OutputDir "training_summary.md")
if ($LASTEXITCODE -ne 0) {
    throw "summary failed with exit code $LASTEXITCODE"
}

Write-Host "Full local 5070 Ti training complete: $OutputDir"
