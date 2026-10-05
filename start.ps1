param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$nextcheckPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $nextcheckPython)) {
    $nextcheckPython = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../work/nextcheck-venv/Scripts/python.exe'))
}
if (-not (Test-Path -LiteralPath $nextcheckPython)) { throw 'Run setup.ps1 first.' }
& $nextcheckPython scripts/launch.py --port $Port
exit $LASTEXITCODE
