<#
.SYNOPSIS
Loads a saved PPO model and prints deterministic evaluation metrics.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\run_eval.ps1 -ModelPath artifacts/ppo_papers_please.zip
#>

param(
  [string]$ModelPath = "artifacts/ppo_papers_please.zip",
  [int]$Episodes = 100,
  [int]$Seed = 10042
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (!(Test-Path $python)) {
  throw "Python not found at $python. Create .venv first."
}

Set-Location $root
& $python -c "from pathlib import Path; from stable_baselines3 import PPO; from eval.evaluate import evaluate_model; p=Path(r'$ModelPath'); model_path_str=str(p); load_path=str(p.with_suffix('')) if p.suffix=='.zip' and p.exists() else model_path_str; model=PPO.load(load_path); s=evaluate_model(model, episodes=$Episodes, seed=$Seed); print('[eval] deterministic policy summary'); print(f'model_path            : {model_path_str}'); print(f'episodes              : $Episodes'); print(f'episode reward        : {s.mean_reward:.3f}'); print(f'decision accuracy     : {s.decision_accuracy:.3f}'); print(f'decision coverage     : {s.decision_coverage:.3f}'); print(f'false accept rate     : {s.false_accept_rate:.3f}'); print(f'false reject rate     : {s.false_reject_rate:.3f}'); print(f'inspection frequency  : {s.inspection_frequency:.3f}')"
