from __future__ import annotations

"""Quick runtime check for environment import/reset/step contract."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import PapersPleaseEnv


def main() -> None:
    """Run one random-policy episode and print basic health signals."""
    env = PapersPleaseEnv(seed=42)
    obs, _ = env.reset(seed=42)
    print("obs_shape:", obs.shape)

    done = False
    steps = 0
    info = {}
    while not done:
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1

    print("steps:", steps)
    print("has_episode_stats:", "episode_stats" in info)


if __name__ == "__main__":
    main()
