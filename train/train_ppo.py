from __future__ import annotations

"""PPO training entrypoint for PapersPleaseEnv."""

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch as th
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
    parser.add_argument("--ent-coef", type=float, default=0.0)

    # Environment params
    parser.add_argument("--day-len", type=int, default=25)
    parser.add_argument("--time-budget", type=int, default=60)
    parser.add_argument("--fraud-rate-min", type=float, default=0.15)
    parser.add_argument("--fraud-rate-max", type=float, default=0.35)
    parser.add_argument("--mid-day-update-prob", type=float, default=0.6)
    parser.add_argument("--inspect-error-prob", type=float, default=0.0)
    parser.add_argument("--inspect-miss-prob", type=float, default=0.0)
    parser.add_argument("--max-inspects-per-applicant", type=int, default=3)
    parser.add_argument("--decision-coverage-target", type=float, default=0.9)
    parser.add_argument("--coverage-shortfall-penalty", type=float, default=-20.0)
    parser.add_argument("--coverage-hard-threshold", type=float, default=0.9)
    parser.add_argument("--coverage-hard-penalty", type=float, default=-120.0)

    # Reward params
    parser.add_argument("--r-correct", type=float, default=6.0)
    parser.add_argument("--p-false-accept", type=float, default=-15.0)
    parser.add_argument("--p-false-reject", type=float, default=-15.0)
    parser.add_argument("--c-inspect", type=float, default=-0.1)
    parser.add_argument("--p-overinspect", type=float, default=-2.0)
    parser.add_argument("--p-reinspect", type=float, default=-0.5)
    parser.add_argument("--p-undecided", type=float, default=0.0)

    return parser.parse_args()


def _build_env_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return dict(
        day_len=args.day_len,
        time_budget=args.time_budget,
        fraud_rate_range=(args.fraud_rate_min, args.fraud_rate_max),
        mid_day_update_prob=args.mid_day_update_prob,
        inspect_error_prob=args.inspect_error_prob,
        inspect_miss_prob=args.inspect_miss_prob,
        max_inspects_per_applicant=args.max_inspects_per_applicant,
        decision_coverage_target=args.decision_coverage_target,
        coverage_shortfall_penalty=args.coverage_shortfall_penalty,
        coverage_hard_threshold=args.coverage_hard_threshold,
        coverage_hard_penalty=args.coverage_hard_penalty,
        r_correct=args.r_correct,
        p_false_accept=args.p_false_accept,
        p_false_reject=args.p_false_reject,
        c_inspect=args.c_inspect,
        p_overinspect=args.p_overinspect,
        p_reinspect=args.p_reinspect,
        p_undecided=args.p_undecided,
        seed=args.seed,
    )


def main() -> None:
    """Create env/model, train PPO, then run deterministic eval."""
    args = parse_args()
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    env_kwargs = _build_env_kwargs(args)
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
        ent_coef=args.ent_coef,
    )

    callback = EpisodeStatsCallback(print_every=args.print_every, verbose=1)
    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=args.progress_bar)
    model.save(str(args.save_path))
    vec_env.close()

    summary = evaluate_model(model=model, episodes=args.eval_episodes, seed=args.seed + 10_000, env_kwargs=env_kwargs)
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
