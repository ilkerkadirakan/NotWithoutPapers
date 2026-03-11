from __future__ import annotations

"""Determinism tests for reset seeding behavior."""

import numpy as np

from env import PapersPleaseEnv


def test_reset_is_deterministic_for_same_seed() -> None:
    env = PapersPleaseEnv(seed=1)
    obs1, info1 = env.reset(seed=999)
    obs2, info2 = env.reset(seed=999)

    assert np.allclose(obs1, obs2)
    assert info1["fraud_rate"] == info2["fraud_rate"]
