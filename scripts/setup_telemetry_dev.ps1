[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$vendorRoot = Join-Path $projectRoot 'artifacts\vendor'
$loaderSource = Join-Path $vendorRoot 'B1CSharpLoader-current'
$loaderRepository = 'https://github.com/game-a11y/B1CSharpLoader.git'
$loaderCommit = '5d607c9d32a14e03608a26e1913dc085bcaaa911'

New-Item -ItemType Directory -Force -Path $vendorRoot | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $loaderSource '.git'))) {
    if (Test-Path -LiteralPath $loaderSource) {
        throw "Vendor path exists but is not a Git checkout: $loaderSource"
    }
    git clone --no-checkout $loaderRepository $loaderSource
    if ($LASTEXITCODE -ne 0) { throw 'Could not clone B1CSharpLoader.' }
}

git -C $loaderSource fetch --depth 1 origin $loaderCommit
if ($LASTEXITCODE -ne 0) { throw 'Could not fetch the pinned B1CSharpLoader commit.' }
git -C $loaderSource -c advice.detachedHead=false checkout --detach $loaderCommit
if ($LASTEXITCODE -ne 0) { throw 'Could not check out the pinned B1CSharpLoader commit.' }
$actualCommit = (git -C $loaderSource rev-parse HEAD).Trim()
if ($actualCommit -ne $loaderCommit) {
    throw "Unexpected B1CSharpLoader commit: $actualCommit"
}

Write-Host "Prepared read-only telemetry SDK at $loaderSource"
Write-Host "Pinned commit: $actualCommit"
