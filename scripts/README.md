# `scripts/` Folder

This folder contains helper scripts for local development.

## Files

- `smoke_test.py`: basic env import/reset/step health check.
- `run_smoke.ps1`: runs smoke test and pytest (if available).
- `run_train.ps1`: starts PPO training with configurable args.
- `run_eval.ps1`: evaluates a saved PPO model and prints metrics.

## Examples

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_train.ps1 -TotalTimesteps 200000
powershell -ExecutionPolicy Bypass -File scripts\run_eval.ps1 -ModelPath artifacts/ppo_papers_please.zip
```
