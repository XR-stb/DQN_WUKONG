[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$loaderSource = Join-Path $projectRoot 'artifacts\vendor\B1CSharpLoader-current'
$modProject = Join-Path $projectRoot 'mods\WukongTelemetry\WukongTelemetry.csproj'
$outputDirectory = Join-Path $projectRoot 'artifacts\mods\WukongTelemetry'

if (-not (Test-Path -LiteralPath (Join-Path $loaderSource 'GameDll\BtlSvr.Main.dll'))) {
    throw 'Telemetry SDK is missing. Run .\scripts\setup_telemetry_dev.ps1 first.'
}

dotnet build $modProject -c Release "-p:LoaderSourceDir=$loaderSource"
if ($LASTEXITCODE -ne 0) { throw 'WukongTelemetry build failed.' }

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$builtDll = Join-Path $projectRoot 'mods\WukongTelemetry\bin\Release\net472\WukongTelemetry.dll'
Copy-Item -LiteralPath $builtDll -Destination $outputDirectory -Force
Copy-Item -LiteralPath (Join-Path $projectRoot 'mods\WukongTelemetry\skill_ids.txt') -Destination $outputDirectory -Force

Write-Host "Telemetry mod package: $outputDirectory"
Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $outputDirectory 'WukongTelemetry.dll')
