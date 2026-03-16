from __future__ import annotations

"""Per-applicant inspect cap behavior tests."""

from env import PapersPleaseEnv
from env.constants import ACTION_INSPECT_HAS_PERMIT, ACTION_INSPECT_NAME_MATCH
from env.domain import oracle_is_legal


def test_overinspect_forces_deny_and_advances_queue() -> None:
    env = PapersPleaseEnv(
        seed=11,
        day_len=5,
        time_budget=20,
        max_inspects_per_applicant=1,
        decision_coverage_target=0.0,
        coverage_hard_threshold=0.0,
    )
    env.reset(seed=11)

    # First inspect is allowed on applicant 0.
    _, _, terminated, truncated, info = env.step(ACTION_INSPECT_HAS_PERMIT)
    assert not terminated
    assert not truncated
    assert info["idx"] == 0

    legal = oracle_is_legal(env.rules, env.queue[env.idx])

    # Second inspect on same applicant exceeds cap -> forced DENY decision.
    _, reward, terminated, truncated, info = env.step(ACTION_INSPECT_NAME_MATCH)
    assert not terminated
    assert not truncated
    assert info["idx"] == 1
    assert info["time_left"] == 18

    expected_decision = env.p_false_reject if legal else env.r_correct
    expected = env.p_overinspect + expected_decision
    assert abs(reward - expected) < 1e-9

    assert env.stats["undecided"] == 0
    assert env.stats["overinspect"] == 1
    assert env.stats["denies"] == 1
