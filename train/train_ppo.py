from __future__ import annotations

"""PPO training entrypoint for PapersPleaseEnv.

This module trains a baseline PPO policy, saves the model, and runs
 deterministic evaluation at the end.
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env import PapersPleaseEnv
from eval.evaluate import evaluate_model
from train.callbacks import EpisodeStatsCallback


def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments."""
    parser = argparse.ArgumentParser(description="Train PPO on PapersPleaseEnv.")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--save-path", type=Path, default=Path("artifacts/ppo_papers_please.zip"))
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Create env/model, train PPO, save artifacts, then evaluate policy."""
    args = parse_args()
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(day_len=25, time_budget=60, seed=args.seed)
    vec_env = make_vec_env(PapersPleaseEnv, n_envs=args.n_envs, seed=args.seed, env_kwargs=env_kwargs)

    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=args.seed,
        verbose=1,
        n_steps=256,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.01,
    )

    callback = EpisodeStatsCallback(print_every=args.print_every, verbose=1)
    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=args.progress_bar)
    model.save(str(args.save_path))
    vec_env.close()

    summary = evaluate_model(model=model, episodes=args.eval_episodes, seed=args.seed + 10_000)
    print("\n[eval] deterministic policy summary")
    print(f"model_path            : {args.save_path}")
    print(f"episodes              : {args.eval_episodes}")
    print(f"episode reward        : {summary.mean_reward:.3f}")
    print(f"decision accuracy     : {summary.decision_accuracy:.3f}")
    print(f"decision coverage     : {summary.decision_coverage:.3f}")
    print(f"false accept rate     : {summary.false_accept_rate:.3f}")
    print(f"false reject rate     : {summary.false_reject_rate:.3f}")
    print(f"inspection frequency  : {summary.inspection_frequency:.3f}")


if __name__ == "__main__":
    main()
