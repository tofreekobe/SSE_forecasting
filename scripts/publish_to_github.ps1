param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryUrl,

    [string]$RemoteName = "origin",

    [string]$Branch = "",

    [switch]$DryRun,

    [switch]$ForceOriginUpdate,

    [switch]$SkipRemoteProbe
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host "==> $Label"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Resolve-Python {
    $venvPython = Join-Path $PSScriptRoot "..\.venv-cu128\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return (Resolve-Path $venvPython).Path
    }
    return "python"
}

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

if (-not $Branch) {
    $Branch = (git branch --show-current).Trim()
}
if (-not $Branch) {
    throw "Could not resolve current git branch."
}

$python = Resolve-Python

Invoke-Step "Verify clean release-ready repository" {
    & $python scripts\check_release_ready.py
}

$existingRemote = ""
try {
    $existingRemote = (git remote get-url $RemoteName 2>$null).Trim()
} catch {
    $existingRemote = ""
}

if ($existingRemote) {
    if ($existingRemote -ne $RepositoryUrl) {
        if (-not $ForceOriginUpdate) {
            throw "Remote '$RemoteName' already points to '$existingRemote'. Use -ForceOriginUpdate to replace it with '$RepositoryUrl'."
        }
        if ($DryRun) {
            Write-Host "Would update $RemoteName remote URL from '$existingRemote' to '$RepositoryUrl'."
        } else {
            Invoke-Step "Update $RemoteName remote URL" {
                & git remote set-url $RemoteName $RepositoryUrl
            }
        }
    } elseif ($DryRun) {
        Write-Host "Remote '$RemoteName' already points to '$RepositoryUrl'."
    }
} else {
    if ($DryRun) {
        Write-Host "Would add $RemoteName remote: $RepositoryUrl"
    } else {
        Invoke-Step "Add $RemoteName remote" {
            & git remote add $RemoteName $RepositoryUrl
        }
    }
}

if (-not $SkipRemoteProbe) {
    Invoke-Step "Probe remote repository" {
        & git ls-remote $RepositoryUrl
    }
}

if ($DryRun) {
    Write-Host "Would push with: git push -u $RemoteName $Branch"
    Write-Host "Dry run complete. No remote or branch changes were made."
} else {
    Invoke-Step "Push $Branch to $RemoteName" {
        & git push -u $RemoteName $Branch
    }
    Write-Host "Published $Branch to $RepositoryUrl"
}
