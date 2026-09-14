[CmdletBinding()]
param(
    [string]$OutputDirectory = 'dist',
    [switch]$SkipDataset
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$previousLocation = Get-Location
$temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    'wukong-rl-export-' + [guid]::NewGuid().ToString('N')
)

try {
    Set-Location $projectRoot
    $dirty = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the Git worktree.' }
    if ($dirty.Count -ne 0) {
        throw 'Commit or stash the worktree before exporting a collaboration bundle.'
    }

    $commit = (git rev-parse --short=12 HEAD).Trim()
    $branch = (git branch --show-current).Trim()
    $remote = (git remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Could not resolve Git metadata.' }

    New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
    $staging = Join-Path $temporaryRoot 'WukongRL'
    $sourceArchive = Join-Path $temporaryRoot 'source.zip'
    git archive --format=zip --output=$sourceArchive HEAD
    if ($LASTEXITCODE -ne 0) { throw 'git archive failed.' }
    Expand-Archive -LiteralPath $sourceArchive -DestinationPath $staging

    $artifactRoot = Join-Path $staging 'artifacts'
    $checkpointTarget = Join-Path $artifactRoot 'checkpoints'
    New-Item -ItemType Directory -Force -Path $checkpointTarget | Out-Null
    foreach ($name in ('latest.pt', 'bc-branched-v3.pt')) {
        $source = Join-Path $projectRoot "artifacts\checkpoints\$name"
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Required checkpoint is missing: $source"
        }
        Copy-Item -LiteralPath $source -Destination $checkpointTarget
    }

    $modSource = Join-Path $projectRoot 'artifacts\mods\WukongTelemetry'
    if (-not (Test-Path -LiteralPath (Join-Path $modSource 'WukongTelemetry.dll') -PathType Leaf)) {
        throw 'Prebuilt WukongTelemetry.dll is missing. Run scripts/build_telemetry_mod.ps1 first.'
    }
    $modTarget = Join-Path $artifactRoot 'mods\WukongTelemetry'
    New-Item -ItemType Directory -Force -Path $modTarget | Out-Null
    Copy-Item -LiteralPath (Join-Path $modSource 'WukongTelemetry.dll') -Destination $modTarget
    Copy-Item -LiteralPath (Join-Path $modSource 'skill_ids.txt') -Destination $modTarget

    if (-not $SkipDataset) {
        $datasetSource = Join-Path $projectRoot 'artifacts\datasets-telemetry-clean\yinhu'
        if (-not (Test-Path -LiteralPath $datasetSource -PathType Container)) {
            throw "Required demonstration dataset is missing: $datasetSource"
        }
        $datasetTarget = Join-Path $artifactRoot 'datasets-telemetry-clean\yinhu'
        New-Item -ItemType Directory -Force -Path (Split-Path $datasetTarget) | Out-Null
        Copy-Item -LiteralPath $datasetSource -Destination $datasetTarget -Recurse
    }

    $artifactFiles = Get-ChildItem -LiteralPath $artifactRoot -Recurse -File
    $manifestFiles = foreach ($file in $artifactFiles) {
        [ordered]@{
            path = [System.IO.Path]::GetRelativePath($staging, $file.FullName).Replace('\', '/')
            bytes = $file.Length
            sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        }
    }
    $manifest = [ordered]@{
        schema_version = 1
        created_at = (Get-Date).ToUniversalTime().ToString('o')
        git_remote = $remote
        git_branch = $branch
        git_commit = $commit
        converged_model = $false
        excluded = @('artifacts/replay', 'artifacts/metrics', 'artifacts/profiles', 'artifacts/evaluations', 'old checkpoints', 'third-party loader source/binaries')
        files = @($manifestFiles)
    }
    $manifestPath = Join-Path $staging 'COLLABORATION-MANIFEST.json'
    [System.IO.File]::WriteAllText(
        $manifestPath,
        ($manifest | ConvertTo-Json -Depth 5),
        [System.Text.UTF8Encoding]::new($false)
    )

    $outputRoot = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
        [System.IO.Path]::GetFullPath($OutputDirectory)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
    }
    New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
    $suffix = if ($SkipDataset) { '-code-model' } else { '-starter' }
    $destination = Join-Path $outputRoot "WukongRL-$commit$suffix.zip"
    if (Test-Path -LiteralPath $destination) {
        throw "Refusing to overwrite an existing bundle: $destination"
    }
    Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $destination -CompressionLevel Optimal
    $bundleHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
    Write-Host "Bundle: $destination"
    Write-Host "Bytes: $((Get-Item -LiteralPath $destination).Length)"
    Write-Host "SHA256: $bundleHash"
} finally {
    Set-Location $previousLocation
    if (Test-Path -LiteralPath $temporaryRoot) {
        $resolvedTemporary = [System.IO.Path]::GetFullPath($temporaryRoot)
        $resolvedSystemTemporary = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
        if (-not $resolvedTemporary.StartsWith($resolvedSystemTemporary, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove an unexpected temporary path: $resolvedTemporary"
        }
        Remove-Item -LiteralPath $resolvedTemporary -Recurse -Force
    }
}
