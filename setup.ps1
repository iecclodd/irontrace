$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
uv venv --python 3.11 .venv
if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed' }
uv pip install --python .venv/Scripts/python.exe --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match -r backend/requirements.lock
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
uv pip install --python .venv/Scripts/python.exe --no-deps -e backend
if ($LASTEXITCODE -ne 0) { throw 'Editable package installation failed' }
npm.cmd --prefix frontend ci
if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed' }
npm.cmd --prefix frontend run build
if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed' }
Write-Output 'Setup complete. Prepare data and run the model doctor as documented in README.md.'
