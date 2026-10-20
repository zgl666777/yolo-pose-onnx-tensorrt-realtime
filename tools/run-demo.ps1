param([switch]$Headless)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtime)) {
    throw 'Install the project in .venv first; see README.md.'
}
$runName = 'runs/demo-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
Push-Location -LiteralPath $projectRoot
try {
    $arguments = @('-m', 'poseguard', 'demo', '--save-video', '--output', $runName)
    if ($Headless) { $arguments += '--headless' }
    & $runtime @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
