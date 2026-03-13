# Papers Please RL Project Architecture

## 1. Goal

Build a Gymnasium environment inspired by Papers Please where an RL agent learns border decisions under partial information.

Primary algorithm: **PPO (Stable-Baselines3)**

Core problem characteristics:

- partial observability
- information gathering actions
- decision making under uncertainty
- reward shaping

---

## 2. Runtime Model

### Episode Model

- One episode = one workday
- Applicants are processed sequentially
- The agent has a limited inspection time budget
- Optional mid-day rule update can change policy constraints once during an episode
- Inspect actions can return noisy observations (missing/incorrect reveal)`r`n- Time-out applies unresolved-applicant penalty to avoid inspect-only loops

### Observation Model

Flat vector includes:

- daily rule state (allowed countries + permit/id card/work-pass requirements)
- current applicant country (one-hot)
- reveal status per field (`unknown / true / false`)
- normalized `time_left`
- normalized remaining applicants

### Action Model

Action IDs (stable contract):

- `0` APPROVE
- `1` DENY
- `2` INSPECT_COUNTRY_ALLOWED
- `3` INSPECT_HAS_PERMIT
- `4` INSPECT_EXPIRY_VALID
- `5` INSPECT_NAME_MATCH
- `6` INSPECT_HAS_ID_CARD
- `7` INSPECT_IS_WORKER
- `8` INSPECT_HAS_WORK_PASS
- `9` INSPECT_PURPOSE_MATCH
- `10` INSPECT_SEAL_VALID`r`n- `11` INSPECT_BIOMETRIC_MATCH

### Reward Philosophy

Reward design balances:

- correct decisions
- false accept / false reject penalties
- inspection cost vs information gain

---

## 3. Codebase Layout

### Core Packages

- `env/`
- `env/constants.py`: action IDs + shared constants
- `env/domain.py`: `Rules`, `Applicant`, `oracle_is_legal`
- `env/sampling.py`: rule/applicant sampling + deny-ratio balancing
- `env/papers_env.py`: Gymnasium environment implementation

- `train/`
- `train/train_ppo.py`: PPO training entrypoint
- `train/callbacks.py`: episode metric logging callback

- `eval/`
- `eval/evaluate.py`: deterministic evaluation loop
- `eval/metrics.py`: metric aggregation (`accuracy`, FAR/FRR, inspection freq, reward)

### Support

- `tests/`: contract + determinism tests
- `scripts/`: helper scripts (`.ps1`, Windows convenience)
- `artifacts/`: trained model files
- `main.py`: platform-independent CLI entrypoint
- `README.md`: GitHub-facing English docs
- `README.tr.md`: detailed Turkish docs

---

## 4. Entry Points

Recommended (platform-independent):

```bash
python main.py smoke --skip-pytest
python main.py train --total-timesteps 200000
python main.py eval --model-path artifacts/ppo_papers_please.zip
```

Windows convenience wrappers (optional):

- `scripts/run_smoke.ps1`
- `scripts/run_train.ps1`
- `scripts/run_eval.ps1`

---

## 5. Quality Gates

Minimum expected checks after environment changes:

1. Environment import works (`from env import PapersPleaseEnv`)
2. Reset/step loop runs at least one episode
3. Observation shape matches `observation_space`
4. `terminated` / `truncated` semantics remain correct
5. `episode_stats` is preserved at episode end

Tests currently cover:

- contract checks
- termination/truncation checks
- deterministic reset behavior

---

## 6. Evaluation Metrics

Reported metrics:

- decision accuracy
- false accept rate
- false reject rate
- inspection frequency
- episode reward

---

## 7. Planned Extensions

- additional document types
- richer fraud patterns
- more country/rule variety
- dynamic in-episode rule updates
- lightweight visualization layer (Tkinter / PyGame / Streamlit)

Extension priority rule:

1. keep minimal environment stable
2. then increase complexity





