from __future__ import annotations

"""Inspect noise behavior tests."""

from env import PapersPleaseEnv
from env.constants import ACTION_INSPECT_HAS_PERMIT


def test_inspect_without_noise_reveals_true_value() -> None:
    env = PapersPleaseEnv(seed=42, inspect_error_prob=0.0, inspect_miss_prob=0.0, time_budget=200)
    env.reset(seed=42)

    true_val = int(env.queue[env.idx].has_permit)
    env.step(ACTION_INSPECT_HAS_PERMIT)
    assert env.revealed["has_permit"] == true_val


def test_inspect_with_full_miss_keeps_unknown() -> None:
    env = PapersPleaseEnv(seed=42, inspect_error_prob=0.0, inspect_miss_prob=1.0, time_budget=200)
    env.reset(seed=42)

    env.step(ACTION_INSPECT_HAS_PERMIT)
    assert env.revealed["has_permit"] == -1
