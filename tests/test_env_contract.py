from __future__ import annotations

"""Environment interface contract tests."""

from env import PapersPleaseEnv


def test_observation_shape_matches_space() -> None:
    env = PapersPleaseEnv(seed=123)
    obs, _ = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape


def test_action_space_size() -> None:
    env = PapersPleaseEnv(seed=123)
    assert env.action_space.n == 11
