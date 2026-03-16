from __future__ import annotations

"""Deterministic policy evaluation loop for saved/active PPO models."""

from typing import Dict, List, Optional

from stable_baselines3 import PPO

from env import PapersPleaseEnv
from eval.metrics import EvalSummary, summarize_episode_stats


def evaluate_model(
    model: PPO,
    episodes: int,
    seed: int,
    env_kwargs: Optional[Dict[str, object]] = None,
) -> EvalSummary:
    """Run policy for N episodes and return aggregate metrics."""
    kwargs = dict(env_kwargs or {})
    kwargs.setdefault("seed", seed)
    env = PapersPleaseEnv(**kwargs)
    all_rewards: List[float] = []
    all_stats: List[Dict[str, float]] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += float(reward)
            done = bool(terminated or truncated)

        all_rewards.append(ep_reward)
        if "episode_stats" in info:
            all_stats.append(info["episode_stats"])

    return summarize_episode_stats(all_rewards, all_stats)
