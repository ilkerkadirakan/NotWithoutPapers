from __future__ import annotations

"""Approve-without-inspect penalty behavior tests."""

from env import PapersPleaseEnv
from env.constants import ACTION_APPROVE
from env.domain import oracle_is_legal


def test_approve_without_inspect_penalty_applies() -> None:
    env = PapersPleaseEnv(
        seed=55,
        day_len=5,
        time_budget=20,
        p_approve_without_inspect=-2.5,
        decision_coverage_target=0.0,
        coverage_hard_threshold=0.0,
    )
    env.reset(seed=55)

    legal = oracle_is_legal(env.rules, env.queue[env.idx])
    decision_reward = env.r_correct if legal else env.p_false_accept

    _, reward, terminated, truncated, info = env.step(ACTION_APPROVE)
    assert not terminated
    assert not truncated
    assert info["idx"] == 1
    assert abs(reward - (decision_reward + env.p_approve_without_inspect)) < 1e-9
    assert env.stats["blind_approve"] == 1
