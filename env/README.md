# `env/` Folder

This folder contains the Gymnasium environment and domain logic.

## Files

- `constants.py`: action IDs and shared constants (countries, fields).
- `domain.py`: `Rules`, `Applicant`, and `oracle_is_legal`.
- `sampling.py`: random daily rule/applicant generation utilities.
- `papers_env.py`: `PapersPleaseEnv` implementation.
- `__init__.py`: package export (`from env import PapersPleaseEnv`).
