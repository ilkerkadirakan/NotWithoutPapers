from __future__ import annotations

"""Redundant inspect penalty behavior tests."""

from env import PapersPleaseEnv
from env.constants import ACTION_INSPECT_HAS_PERMIT


def test_reinspect_penalty_applies_on_same_field() -> None:
    env = PapersPleaseEnv(
        seed=21,
        day_len=5,
        time_budget=20,
        max_inspects_per_applicant=3,
        p_reinspect=-1.5,
        decision_coverage_target=0.0,
        coverage_hard_threshold=0.0,
    )
    env.reset(seed=21)

    # First inspect reveals field and charges only inspect cost.
    _, reward1, terminated, truncated, info1 = env.step(ACTION_INSPECT_HAS_PERMIT)
    assert not terminated
    assert not truncated
    assert info1["idx"] == 0
    assert abs(reward1 - env.c_inspect) < 1e-9

    # Re-inspecting same field adds redundant inspect penalty.
    _, reward2, terminated, truncated, info2 = env.step(ACTION_INSPECT_HAS_PERMIT)
    assert not terminated
    assert not truncated
    assert info2["idx"] == 0
    assert info2["time_left"] == 18
    assert abs(reward2 - (env.c_inspect + env.p_reinspect)) < 1e-9
    assert env.stats["reinspect"] == 1
    assert env.stats["inspects"] == 2
