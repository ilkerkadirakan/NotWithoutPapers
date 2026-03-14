from __future__ import annotations

"""Training callbacks for logging project-specific episode metrics."""

from typing import Dict, List

from stable_baselines3.common.callbacks import BaseCallback

from eval.metrics import summarize_episode_stats


class EpisodeStatsCallback(BaseCallback):
    """Collect episode stats from vectorized envs and print rolling summaries."""

    def __init__(self, print_every: int = 50, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.print_every = max(1, int(print_every))
        self.episode_rewards: List[float] = []
        self.episode_stats: List[Dict[str, float]] = []
        self._current_rewards: List[float] = []
        self._last_reported_episodes = 0

    def _on_training_start(self) -> None:
        self._current_rewards = [0.0 for _ in range(self.training_env.num_envs)]

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for i in range(len(rewards)):
            self._current_rewards[i] += float(rewards[i])
            if dones[i]:
                self.episode_rewards.append(self._current_rewards[i])
                self._current_rewards[i] = 0.0
                ep_stats = infos[i].get("episode_stats")
                if ep_stats:
                    self.episode_stats.append(ep_stats)

        n_episodes = len(self.episode_stats)
        if (
            self.verbose
            and n_episodes > 0
            and n_episodes != self._last_reported_episodes
            and n_episodes % self.print_every == 0
        ):
            summary = summarize_episode_stats(self.episode_rewards, self.episode_stats[-self.print_every :])
            print(
                f"[train] episodes={n_episodes} "
                f"avg_reward={summary.mean_reward:.3f} "
                f"acc={summary.decision_accuracy:.3f} "
                f"cov={summary.decision_coverage:.3f} "
                f"fa={summary.false_accept_rate:.3f} "
                f"fr={summary.false_reject_rate:.3f} "
                f"inspect={summary.inspection_frequency:.3f}"
            )
            self._last_reported_episodes = n_episodes
        return True
