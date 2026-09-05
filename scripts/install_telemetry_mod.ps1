[CmdletBinding()]
param(
    [string]$GameDirectory = 'C:\Program Files (x86)\Steam\steamapps\common\BlackMythWukong',
    [switch]$EnableJit
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$gameRoot = [System.IO.Path]::GetFullPath($GameDirectory)
$gameBinaryDirectory = Join-Path $gameRoot 'b1\Binaries\Win64'
$gameExecutable = Join-Path $gameBinaryDirectory 'b1-Win64-Shipping.exe'
$loaderArchive = Join-Path $projectRoot 'artifacts\vendor\B1CSharpLoader-0.0.8.zip'
$loaderStaging = Join-Path $projectRoot 'artifacts\vendor\B1CSharpLoader-release-0.0.8'
$loaderUrl = 'https://github.com/czastack/B1CSharpLoader/releases/download/v0.0.8/B1CSharpLoader-0.0.8.zip'
$loaderSha256 = '721E8C34174060AD988B91100A63D013AF361109E2A173A754C7FEDA1847D4F0'
$modPackage = Join-Path $projectRoot 'artifacts\mods\WukongTelemetry'

if (Get-Process -Name 'b1-Win64-Shipping' -ErrorAction SilentlyContinue) {
    throw 'Black Myth: Wukong is running. Close the game before installing telemetry.'
}
if (-not (Test-Path -LiteralPath $gameExecutable -PathType Leaf)) {
    throw "Game executable was not found at the validated target: $gameExecutable"
}

& (Join-Path $PSScriptRoot 'build_telemetry_mod.ps1')
if (-not (Test-Path -LiteralPath $loaderArchive -PathType Leaf)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $loaderArchive) | Out-Null
    Invoke-WebRequest -Headers @{ 'User-Agent' = 'wukong-rl-telemetry-installer' } -Uri $loaderUrl -OutFile $loaderArchive
}
$actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $loaderArchive).Hash
if ($actualHash -ne $loaderSha256) {
    throw "B1CSharpLoader archive hash mismatch. Expected $loaderSha256, got $actualHash"
}
if (-not (Test-Path -LiteralPath (Join-Path $loaderStaging 'b1\Binaries\Win64\version.dll'))) {
    New-Item -ItemType Directory -Force -Path $loaderStaging | Out-Null
    Expand-Archive -LiteralPath $loaderArchive -DestinationPath $loaderStaging
}

$sourceBinaryDirectory = Join-Path $loaderStaging 'b1\Binaries\Win64'
$sourceVersionDll = Join-Path $sourceBinaryDirectory 'version.dll'
$sourceLoaderDirectory = Join-Path $sourceBinaryDirectory 'CSharpLoader'
$targetVersionDll = Join-Path $gameBinaryDirectory 'version.dll'
$targetLoaderDirectory = Join-Path $gameBinaryDirectory 'CSharpLoader'
if (Test-Path -LiteralPath $targetVersionDll) {
    $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceVersionDll).Hash
    $targetHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $targetVersionDll).Hash
    if ($sourceHash -ne $targetHash) {
        throw "A different version.dll already exists. It was not overwritten: $targetVersionDll"
    }
}
if (Test-Path -LiteralPath $targetLoaderDirectory) {
    foreach ($sourceFile in Get-ChildItem -LiteralPath $sourceLoaderDirectory -File) {
        if ($sourceFile.Name -eq 'b1cs.ini') { continue }
        $targetFile = Join-Path $targetLoaderDirectory $sourceFile.Name
        if (Test-Path -LiteralPath $targetFile) {
            $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceFile.FullName).Hash
            $targetHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $targetFile).Hash
            if ($sourceHash -ne $targetHash) {
                throw "A different loader file already exists. Nothing was overwritten: $targetFile"
            }
        }
    }
}

if (-not (Test-Path -LiteralPath $targetVersionDll)) {
    Copy-Item -LiteralPath $sourceVersionDll -Destination $targetVersionDll
}
New-Item -ItemType Directory -Force -Path $targetLoaderDirectory | Out-Null
foreach ($sourceFile in Get-ChildItem -LiteralPath $sourceLoaderDirectory -File) {
    if ($sourceFile.Name -eq 'b1cs.ini') { continue }
    $targetFile = Join-Path $targetLoaderDirectory $sourceFile.Name
    if (-not (Test-Path -LiteralPath $targetFile)) {
        Copy-Item -LiteralPath $sourceFile.FullName -Destination $targetFile
    }
}
$targetLoaderConfig = Join-Path $targetLoaderDirectory 'b1cs.ini'
if (-not (Test-Path -LiteralPath $targetLoaderConfig)) {
    Copy-Item -LiteralPath (Join-Path $projectRoot 'mods\WukongTelemetry\b1cs.ini') -Destination $targetLoaderConfig
}

$targetModDirectory = Join-Path $targetLoaderDirectory 'Mods\WukongTelemetry'
New-Item -ItemType Directory -Force -Path $targetModDirectory | Out-Null
$existingModDll = Join-Path $targetModDirectory 'WukongTelemetry.dll'
$shouldBackupConfig = $EnableJit -and (Test-Path -LiteralPath $targetLoaderConfig -PathType Leaf)
if ((Test-Path -LiteralPath $existingModDll) -or $shouldBackupConfig) {
    $backupDirectory = Join-Path $projectRoot ('artifacts\backups\telemetry\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
    New-Item -ItemType Directory -Force -Path $backupDirectory | Out-Null
    if (Test-Path -LiteralPath $existingModDll) {
        Copy-Item -LiteralPath $existingModDll -Destination $backupDirectory
    }
    if ($shouldBackupConfig) {
        Copy-Item -LiteralPath $targetLoaderConfig -Destination $backupDirectory
    }
    Write-Host "Backed up previous telemetry files to $backupDirectory"
}
if ($EnableJit) {
    $loaderConfigText = [System.IO.File]::ReadAllText($targetLoaderConfig)
    if ($loaderConfigText -notmatch '(?m)^EnableJit=[01]\s*$') {
        throw "Loader config does not contain a recognized EnableJit setting: $targetLoaderConfig"
    }
    $loaderConfigText = [regex]::Replace(
        $loaderConfigText,
        '(?m)^EnableJit=[01]\s*$',
        'EnableJit=1')
    [System.IO.File]::WriteAllText($targetLoaderConfig, $loaderConfigText)
    Write-Host 'CSharpLoader JIT explicitly enabled for the persistent telemetry ticker.'
}
Copy-Item -LiteralPath (Join-Path $modPackage 'WukongTelemetry.dll') -Destination $targetModDirectory -Force
$targetSkillIds = Join-Path $targetModDirectory 'skill_ids.txt'
if (-not (Test-Path -LiteralPath $targetSkillIds)) {
    Copy-Item -LiteralPath (Join-Path $modPackage 'skill_ids.txt') -Destination $targetSkillIds
}

Write-Host 'Read-only telemetry installed. No game/save files were modified.'
Write-Host "Mod path: $targetModDirectory"
Write-Host 'Start the game, enter combat, then run:'
Write-Host '.\.venv\Scripts\python.exe -m wukong_rl telemetry-probe --seconds 20'
