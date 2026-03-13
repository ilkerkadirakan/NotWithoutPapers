from __future__ import annotations

"""Undecided penalty behavior tests."""

from env import PapersPleaseEnv
from env.constants import ACTION_INSPECT_HAS_PERMIT


def test_time_out_adds_undecided_stats() -> None:
    env = PapersPleaseEnv(seed=7, day_len=10, time_budget=1, inspect_error_prob=0.0, inspect_miss_prob=0.0)
    env.reset(seed=7)

    # Consume the only time unit via inspect; this should force truncation and undecided penalty.
    _, _, terminated, truncated, info = env.step(ACTION_INSPECT_HAS_PERMIT)
    assert not terminated
    assert truncated
    assert "episode_stats" in info
    assert info["episode_stats"]["undecided"] >= 1
