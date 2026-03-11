# NotWithoutPapers

A Papers Please-inspired reinforcement learning environment where an agent learns border decisions under **partial observability**.

Primary algorithm: **PPO (Stable-Baselines3)**

Turkish detailed documentation is available at [README.tr.md](README.tr.md).

## What This Project Models

Each episode is one workday:

- Daily rules are sampled (allowed countries, permit requirement).
- Applicants are processed in a queue.
- Document fields start hidden.
- The agent can inspect fields (costs time + small penalty).
- The agent decides `APPROVE` or `DENY`.

This creates the core RL trade-off:

- More inspection -> better information, higher time/cost.
- Less inspection -> faster decisions, higher error risk.

## Project Structure

```text
env/      # Gymnasium environment + domain logic
train/    # PPO training entrypoint + callback
eval/     # Evaluation loop + metrics aggregation
tests/    # Contract and determinism tests
scripts/  # Helper scripts (PowerShell, optional)
main.py   # Cross-platform CLI entrypoint
```

## Quick Start

## 1) Install

```bash
python -m venv .venv
python -m pip install -U pip
python -m pip install -r requirements.txt
```

On Windows, prefer:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2) Smoke Check

```bash
python main.py smoke --skip-pytest
```

## 3) Run Tests

```bash
python -m pytest
```

## 4) Train PPO

```bash
python main.py train --total-timesteps 200000 --n-envs 8 --eval-episodes 100
```

## 5) Evaluate Saved Model

```bash
python main.py eval --model-path artifacts/ppo_papers_please.zip --episodes 100
```

## Environment Contract (Important)

Action IDs (do not reorder):

- `0`: APPROVE
- `1`: DENY
- `2`: INSPECT_COUNTRY_ALLOWED
- `3`: INSPECT_HAS_PERMIT
- `4`: INSPECT_EXPIRY_VALID
- `5`: INSPECT_NAME_MATCH

Observation vector contains:

- daily rule state
- current applicant country (one-hot)
- reveal status per field (unknown / true / false)
- normalized `time_left`
- normalized remaining applicants

If this contract changes, update training/evaluation code in the same commit.

## Metrics

The project tracks:

- decision accuracy
- false accept rate
- false reject rate
- inspection frequency
- episode reward

## Why `main.py` as the Entry Point?

`main.py` is the recommended, platform-independent interface:

- `python main.py smoke`
- `python main.py train`
- `python main.py eval`

PowerShell scripts in `scripts/` are convenience wrappers for Windows, not required.

## Suggested Reading Order (to Understand the Code)

1. `main.py`
2. `train/train_ppo.py`
3. `env/papers_env.py`
4. `env/sampling.py`
5. `eval/metrics.py`
6. `tests/`

## Current Status

- Modularized architecture (`env/train/eval/tests/scripts`) is in place.
- Determinism and contract checks exist in tests.
- End-to-end train/eval flow is runnable from `main.py`.
