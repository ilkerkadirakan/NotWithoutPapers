from __future__ import annotations

"""Mid-day rule update behavior tests."""

from env import PapersPleaseEnv
from env.constants import ACTION_APPROVE


def _run_all_decisions(env: PapersPleaseEnv) -> int:
    obs, _ = env.reset(seed=123)
    _ = obs
    updates = 0
    done = False
    while not done:
        _, _, term, trunc, info = env.step(ACTION_APPROVE)
        if "rule_update" in info:
            updates += 1
        done = bool(term or trunc)
    return updates


def test_mid_day_update_happens_once_when_enabled() -> None:
    env = PapersPleaseEnv(seed=123, day_len=20, time_budget=200, mid_day_update_prob=1.0)
    updates = _run_all_decisions(env)
    assert updates == 1


def test_mid_day_update_never_happens_when_disabled() -> None:
    env = PapersPleaseEnv(seed=123, day_len=20, time_budget=200, mid_day_update_prob=0.0)
    updates = _run_all_decisions(env)
    assert updates == 0
