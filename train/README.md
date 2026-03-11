# `train/` Folder

This folder contains PPO training logic.

## Files

- `train_ppo.py`: main training entrypoint for SB3 PPO.
- `callbacks.py`: custom callback for episode metric logging.
- `__init__.py`: package export for callback class.

## Run

```powershell
python -m train.train_ppo --total-timesteps 200000
```
