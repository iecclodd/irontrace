param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$irontracePython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $irontracePython)) { throw 'Run .\setup.ps1 from the repository first.' }
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'frontend/dist/index.html'))) {
    throw 'The console is not built. Run npm.cmd --prefix frontend run build.'
}
& $irontracePython scripts/launch.py --port $Port
exit $LASTEXITCODE
