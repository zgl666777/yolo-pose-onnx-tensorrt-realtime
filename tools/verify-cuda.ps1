$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'gpu-env.ps1')
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsRoot) { throw 'MSVC C++ tools are missing' }
Import-Module (Join-Path $vsRoot 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vsRoot -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
Push-Location -LiteralPath $projectRoot
try {
    New-Item -ItemType Directory -Path 'build-gpu' -Force | Out-Null
    & (Join-Path $env:CUDA_PATH 'bin\nvcc.exe') -std=c++17 -arch=sm_89 tools/cuda_smoke.cu -o build-gpu/cuda_smoke.exe
    if ($LASTEXITCODE -ne 0) { throw 'nvcc kernel compilation failed' }
    & .\build-gpu\cuda_smoke.exe
    if ($LASTEXITCODE -ne 0) { throw 'CUDA kernel execution failed' }
} finally { Pop-Location }

