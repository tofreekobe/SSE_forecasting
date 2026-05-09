$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Out = Join-Path $Root "pai_sse_code_bundle.zip"

if (Test-Path $Out) {
    Remove-Item -LiteralPath $Out
}

$Items = @(
    "src",
    "scripts",
    "tests",
    "docs",
    "requirements-pai.txt",
    "requirements-diagnostics.txt",
    "docs_hf_diagnostics.md"
)

$Paths = foreach ($Item in $Items) {
    $Path = Join-Path $Root $Item
    if (Test-Path $Path) {
        $Path
    }
}

Compress-Archive -Path $Paths -DestinationPath $Out -Force
Write-Host "Created $Out"
