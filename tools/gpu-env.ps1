param(
    [string]$SdkRoot = (Join-Path $env:USERPROFILE '.local\poseguard-sdk')
)
$ErrorActionPreference = 'Stop'
$env:CUDA_PATH = Join-Path $SdkRoot 'cuda-12.8'
$env:CUDAToolkit_ROOT = $env:CUDA_PATH
$env:TENSORRT_ROOT = Join-Path $SdkRoot 'TensorRT-10.8.0.43'
$env:OpenCV_DIR = Join-Path $SdkRoot 'opencv\build'
$directories = @(
    (Join-Path $env:CUDA_PATH 'bin'),
    (Join-Path $env:TENSORRT_ROOT 'lib'),
    (Join-Path $env:TENSORRT_ROOT 'bin')
)
foreach ($directory in $directories) {
    if (-not (Test-Path -LiteralPath $directory)) { throw ('SDK directory missing: ' + $directory) }
}
$opencvBin = Join-Path $env:OpenCV_DIR 'x64\vc16\bin'
if (Test-Path -LiteralPath $opencvBin) { $directories += $opencvBin }
$entries = @($env:PATH -split ';')
foreach ($directory in $directories) {
    if ($entries -notcontains $directory) { $entries = @($directory) + $entries }
}
$env:PATH = $entries -join ';'
Write-Output ('CUDA_PATH=' + $env:CUDA_PATH)
Write-Output ('TENSORRT_ROOT=' + $env:TENSORRT_ROOT)
Write-Output ('OpenCV_DIR=' + $env:OpenCV_DIR)
