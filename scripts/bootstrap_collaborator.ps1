[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$WithPerformanceMetrics
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$venvDirectory = Join-Path $projectRoot '.venv'
$python = Join-Path $venvDirectory 'Scripts\python.exe'
$previousLocation = Get-Location

try {
    Set-Location $projectRoot
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        $uv = Get-Command uv -ErrorAction SilentlyContinue
        if ($uv) {
            & $uv.Source venv --python 3.10 $venvDirectory
            if ($LASTEXITCODE -ne 0) { throw 'uv could not create the Python 3.10 environment.' }
        } else {
            $py = Get-Command py -ErrorAction SilentlyContinue
            if (-not $py) {
                throw 'Install uv or Python 3.10 first. See docs/collaboration-setup.md.'
            }
            & $py.Source -3.10 -m venv $venvDirectory
            if ($LASTEXITCODE -ne 0) { throw 'Python 3.10 could not create the virtual environment.' }
        }
    }

    $extras = if ($WithPerformanceMetrics) {
        '.[dev,dashboard,performance]'
    } else {
        '.[dev,dashboard]'
    }
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) {
        & $uv.Source pip install --python $python -e $extras
    } else {
        & $python -m pip install -e $extras
    }
    if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }

    & $python -c "import torch, wukong_rl; print('Python environment OK; torch=' + torch.__version__ + '; cuda=' + str(torch.cuda.is_available()))"
    if ($LASTEXITCODE -ne 0) { throw 'Package import smoke test failed.' }
    if (-not $SkipTests) {
        & $python -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw 'Repository tests failed.' }
    }

    $dataset = Join-Path $projectRoot 'artifacts\datasets-telemetry-clean\yinhu'
    $onlineCheckpoint = Join-Path $projectRoot 'artifacts\checkpoints\latest.pt'
    $bcCheckpoint = Join-Path $projectRoot 'artifacts\checkpoints\bc-branched-v3.pt'
    Write-Host ''
    Write-Host 'Bootstrap complete.'
    Write-Host "Dataset included: $([bool](Test-Path -LiteralPath $dataset))"
    Write-Host "Online checkpoint included: $([bool](Test-Path -LiteralPath $onlineCheckpoint))"
    Write-Host "BC checkpoint included: $([bool](Test-Path -LiteralPath $bcCheckpoint))"
    if (-not $WithPerformanceMetrics) {
        Write-Host 'Optional process/GPU resource metrics were skipped. Re-run with -WithPerformanceMetrics if needed.'
    }
    Write-Host 'Next: read START-HERE-COLLABORATOR.md, then calibrate against your own game window.'
} finally {
    Set-Location $previousLocation
}
