param([string]$BuildDirectory = 'build-gpu')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'gpu-env.ps1')
if (-not (Test-Path -LiteralPath (Join-Path $env:OpenCV_DIR 'OpenCVConfig.cmake'))) {
    throw 'OpenCV C++ SDK is not installed.'
}
Push-Location -LiteralPath $projectRoot
try {
    & cmake -S . -B $BuildDirectory -G 'Visual Studio 17 2022' -A x64 `
        "-DCUDAToolkit_ROOT=$env:CUDA_PATH" "-DTENSORRT_ROOT=$env:TENSORRT_ROOT" "-DOpenCV_DIR=$env:OpenCV_DIR"
    if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed' }
    & cmake --build $BuildDirectory --config Release --parallel 4
    if ($LASTEXITCODE -ne 0) { throw 'C++ build failed' }
} finally { Pop-Location }

