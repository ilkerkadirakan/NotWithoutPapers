<#
.SYNOPSIS
Runs environment smoke test and then pytest (if installed).

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\run_smoke.ps1
#>

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (!(Test-Path $python)) {
  throw "Python not found at $python. Create .venv first."
}

Set-Location $root
& $python scripts\smoke_test.py
if ($LASTEXITCODE -ne 0) {
  throw "smoke_test.py failed with exit code $LASTEXITCODE"
}

& $python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pytest') else 1)"
if ($LASTEXITCODE -ne 0) {
  Write-Host "pytest is not installed in .venv."
  Write-Host "Install dependencies with:"
  Write-Host "  .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
  exit 1
}

& $python -m pytest
if ($LASTEXITCODE -ne 0) {
  throw "pytest failed with exit code $LASTEXITCODE"
}
