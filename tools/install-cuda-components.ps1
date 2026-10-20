param(
    [string]$SdkRoot = (Join-Path $env:USERPROFILE '.local\poseguard-sdk')
)
$ErrorActionPreference = 'Stop'
$release = '12.8.1'
$base = 'https://developer.download.nvidia.com/compute/cuda/redist/'
$manifest = Invoke-RestMethod -Uri ($base + 'redistrib_' + $release + '.json')
$downloads = Join-Path $SdkRoot 'downloads'
$staging = Join-Path $SdkRoot 'cuda-staging'
$destination = Join-Path $SdkRoot 'cuda-12.8'
New-Item -ItemType Directory -Path $downloads, $staging, $destination -Force | Out-Null
foreach ($component in @('cuda_nvcc', 'cuda_cudart', 'cuda_cccl', 'cuda_nvrtc')) {
    $entry = $manifest.$component.'windows-x86_64'
    $archive = Join-Path $downloads (Split-Path -Leaf $entry.relative_path)
    $valid = (Test-Path -LiteralPath $archive) -and
        ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -eq $entry.sha256)
    if (-not $valid) {
        Write-Output ('Downloading ' + $component + ' (' + $entry.size + ' bytes)')
        & curl.exe --silent --show-error --location --fail --retry 2 --connect-timeout 30 `
            --max-time 1800 --output $archive ($base + $entry.relative_path)
        if ($LASTEXITCODE -ne 0) { throw ('Download failed: ' + $component) }
    }
    $hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $entry.sha256) { throw ('SHA256 mismatch: ' + $component) }
    $componentDir = Join-Path $staging $component
    New-Item -ItemType Directory -Path $componentDir -Force | Out-Null
    & tar.exe -xf $archive -C $componentDir
    if ($LASTEXITCODE -ne 0) { throw ('Extraction failed: ' + $component) }
    $root = Get-ChildItem -LiteralPath $componentDir -Directory | Select-Object -First 1
    Copy-Item -Path (Join-Path $root.FullName '*') -Destination $destination -Recurse -Force
    Write-Output ('Installed ' + $component + ' with verified SHA256')
}
$manifest | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath (Join-Path $destination 'redistribution.json') -Encoding utf8
& (Join-Path $destination 'bin\nvcc.exe') --version
if ($LASTEXITCODE -ne 0) { throw 'nvcc verification failed' }
Write-Output ('CUDA_ROOT=' + $destination)

