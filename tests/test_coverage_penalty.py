from __future__ import annotations

"""Decision coverage penalty behavior tests."""

from env import PapersPleaseEnv
from env.constants import ACTION_INSPECT_HAS_PERMIT


def test_coverage_penalty_applies_on_timeout_when_no_decisions() -> None:
    env = PapersPleaseEnv(
        seed=7,
        day_len=10,
        time_budget=1,
        inspect_error_prob=0.0,
        inspect_miss_prob=0.0,
        decision_coverage_target=0.8,
        coverage_shortfall_penalty=-6.0,
    )
    env.reset(seed=7)

    # Consume the only time unit via inspect; this should force truncation.
    _, reward, terminated, truncated, info = env.step(ACTION_INSPECT_HAS_PERMIT)
    assert not terminated
    assert truncated
    assert "episode_stats" in info

    # target=ceil(10*0.8)=8, decisions=0 => shortfall=8
    assert info["episode_stats"]["coverage_shortfall"] == 8
    expected = env.c_inspect + (env.coverage_shortfall_penalty * 8)
    assert abs(reward - expected) < 1e-9
