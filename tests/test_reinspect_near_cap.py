from __future__ import annotations

"""Near-cap redundant inspect should force a reveal-based decision."""

from env import PapersPleaseEnv
from env.constants import ACTION_APPROVE, ACTION_DENY, ACTION_INSPECT_HAS_PERMIT
from env.constants import COUNTRIES
from env.domain import oracle_is_legal


def test_reinspect_near_cap_forces_decision_and_advances() -> None:
    env = PapersPleaseEnv(
        seed=31,
        day_len=5,
        time_budget=20,
        max_inspects_per_applicant=2,
        p_reinspect=-0.5,
        decision_coverage_target=0.0,
        coverage_hard_threshold=0.0,
    )
    env.reset(seed=31)

    # First inspect on applicant 0.
    _, _, terminated, truncated, info = env.step(ACTION_INSPECT_HAS_PERMIT)
    assert not terminated
    assert not truncated
    assert info["idx"] == 0

    app = env.queue[env.idx]
    rules = env.rules
    legal = oracle_is_legal(rules, app)
    citizen = int(app.country_idx == COUNTRIES.index("ARSTOTZKA"))
    deny_known = (env.revealed["has_permit"] == 0) and (citizen == 0) and (rules.permit_required == 1)
    forced_decision = ACTION_DENY if deny_known else ACTION_APPROVE

    if forced_decision == ACTION_DENY:
        decision_reward = env.r_correct if not legal else env.p_false_reject
    else:
        decision_reward = env.r_correct if legal else env.p_false_accept

    # Redundant inspect at near-cap should force a decision in same step.
    _, reward, terminated, truncated, info = env.step(ACTION_INSPECT_HAS_PERMIT)
    assert not terminated
    assert not truncated
    assert info["idx"] == 1
    assert info["time_left"] == 18
    assert abs(reward - (env.p_reinspect + decision_reward)) < 1e-9

    assert env.stats["reinspect"] == 1
    assert env.stats["inspects"] == 1
