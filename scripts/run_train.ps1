<#
.SYNOPSIS
Runs PPO training from project root using `.venv` Python.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\run_train.ps1 -TotalTimesteps 200000
#>

param(
  [int]$TotalTimesteps = 200000,
  [int]$NEnvs = 8,
  [int]$Seed = 42,
  [int]$EvalEpisodes = 100,
  [string]$SavePath = "artifacts/ppo_papers_please.zip",
  [int]$PrintEvery = 50
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (!(Test-Path $python)) {
  throw "Python not found at $python. Create .venv first."
}

Set-Location $root
& $python -m train.train_ppo `
  --total-timesteps $TotalTimesteps `
  --n-envs $NEnvs `
  --seed $Seed `
  --eval-episodes $EvalEpisodes `
  --save-path $SavePath `
  --print-every $PrintEvery
