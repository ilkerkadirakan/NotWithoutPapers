from __future__ import annotations

"""Termination/truncation behavior tests."""

from env import PapersPleaseEnv


def test_terminated_or_truncated_eventually_true() -> None:
    env = PapersPleaseEnv(seed=7)
    env.reset(seed=7)
    done = False
    last_info = {}
    while not done:
        _, _, terminated, truncated, last_info = env.step(env.action_space.sample())
        done = terminated or truncated

    assert done
    assert "episode_stats" in last_info
