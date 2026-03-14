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
        coverage_hard_threshold=0.9,
        coverage_hard_penalty=-50.0,
    )
    env.reset(seed=7)

    # Consume the only time unit via inspect; this should force truncation.
    _, reward, terminated, truncated, info = env.step(ACTION_INSPECT_HAS_PERMIT)
    assert not terminated
    assert truncated
    assert "episode_stats" in info

    # target=ceil(10*0.8)=8, decisions=0 => shortfall=8 + hard-coverage violation
    assert info["episode_stats"]["coverage_shortfall"] == 8
    assert info["episode_stats"]["coverage_hard_violations"] == 1
    assert abs(info["episode_stats"]["decision_coverage"] - 0.0) < 1e-12

    expected = env.c_inspect + (env.coverage_shortfall_penalty * 8) + env.coverage_hard_penalty
    assert abs(reward - expected) < 1e-9
