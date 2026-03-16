# `tests/` Folder

This folder contains contract tests for environment stability.

## Files

- `test_env_contract.py`: observation shape and action-space checks.
- `test_episode_flags.py`: validates episode termination/truncation behavior.
- `test_determinism.py`: checks deterministic reset behavior with same seed.
- `test_reinspect_penalty.py`: verifies redundant inspect penalty on repeated field checks.

## Run

```powershell
python -m pytest
```
