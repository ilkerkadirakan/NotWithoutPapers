from __future__ import annotations

"""Undecided penalty behavior tests."""

from env import PapersPleaseEnv
from env.constants import ACTION_INSPECT_HAS_PERMIT


def test_time_out_has_no_undecided_penalty_by_default() -> None:
    env = PapersPleaseEnv(seed=7, day_len=10, time_budget=1, inspect_error_prob=0.0, inspect_miss_prob=0.0)
    env.reset(seed=7)

    # Consume the only time unit via inspect; this should force truncation.
    _, reward, terminated, truncated, info = env.step(ACTION_INSPECT_HAS_PERMIT)
    assert not terminated
    assert truncated
    assert "episode_stats" in info
    # Counter can still be tracked, but default penalty is disabled.
    assert info["episode_stats"]["undecided"] == 10
    # With zero undecided penalty, only inspect cost is applied.
    assert abs(reward - env.c_inspect) < 1e-9
