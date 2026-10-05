$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
foreach ($irontraceTool in @('uv', 'node', 'npm.cmd')) {
    if (-not (Get-Command $irontraceTool -ErrorAction SilentlyContinue)) {
        throw "Missing prerequisite: $irontraceTool. See README.md, then reopen PowerShell."
    }
}
$irontraceNode = [version](& node -p 'process.versions.node')
if ($irontraceNode -lt [version]'22.12.0') { throw 'Install Node.js 22.12 or newer, then reopen PowerShell.' }
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    uv venv --python 3.11 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed' }
} else {
    & ./.venv/Scripts/python.exe -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)'
    if ($LASTEXITCODE -ne 0) { throw 'Existing .venv must use Python 3.11. Rename it, then rerun setup.ps1.' }
}
uv pip install --python .venv/Scripts/python.exe --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match -r backend/requirements.lock
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
uv pip install --python .venv/Scripts/python.exe --no-deps -e backend
if ($LASTEXITCODE -ne 0) { throw 'Editable package installation failed' }
npm.cmd --prefix frontend ci
if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed' }
npm.cmd --prefix frontend run build
if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed' }
Write-Output 'IRONTRACE setup complete. Follow README.md to prepare the model, prepare data, run the doctor, and start the console.'
