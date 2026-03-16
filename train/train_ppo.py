from __future__ import annotations

"""PPO training entrypoint for PapersPleaseEnv.

This module optionally runs behavior cloning warm-start with a heuristic expert,
then trains PPO and evaluates the final policy.
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env import PapersPleaseEnv
from env.constants import (
    ACTION_APPROVE,
    ACTION_DENY,
    ACTION_INSPECT_BIOMETRIC_MATCH,
    ACTION_INSPECT_COUNTRY_ALLOWED,
    ACTION_INSPECT_EXPIRY_VALID,
    ACTION_INSPECT_HAS_ID_CARD,
    ACTION_INSPECT_HAS_PERMIT,
    ACTION_INSPECT_HAS_WORK_PASS,
    ACTION_INSPECT_IS_WORKER,
    ACTION_INSPECT_NAME_MATCH,
    ACTION_INSPECT_PURPOSE_MATCH,
    ACTION_INSPECT_SEAL_VALID,
    COUNTRIES,
)
from env.domain import oracle_is_legal
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
    parser.add_argument("--ent-coef", type=float, default=0.01)

    # Behavior cloning warm-start params
    parser.add_argument("--bc-pretrain-episodes", type=int, default=0)
    parser.add_argument("--bc-epochs", type=int, default=5)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--bc-lr", type=float, default=1e-3)

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
        p_undecided=args.p_undecided,
        seed=args.seed,
    )


def _expert_action(env: PapersPleaseEnv) -> int:
    """Heuristic expert policy for BC data generation."""
    assert env.rules is not None
    app = env.queue[env.idx]
    rev = env.revealed
    citizen = int(app.country_idx == COUNTRIES.index("ARSTOTZKA"))

    # Avoid inspect loops in expert trajectories: force decision near inspect budget.
    if env.current_inspects >= max(1, env.max_inspects_per_applicant - 1):
        return ACTION_APPROVE if oracle_is_legal(env.rules, app) else ACTION_DENY

    # High-value global checks first.
    if rev["country_allowed"] == -1:
        return ACTION_INSPECT_COUNTRY_ALLOWED
    if rev["expiry_valid"] == -1:
        return ACTION_INSPECT_EXPIRY_VALID
    if rev["purpose_match"] == -1:
        return ACTION_INSPECT_PURPOSE_MATCH
    if rev["biometric_match"] == -1:
        return ACTION_INSPECT_BIOMETRIC_MATCH

    # Rule-dependent document checks.
    if citizen == 0 and env.rules.permit_required == 1 and rev["has_permit"] == -1:
        return ACTION_INSPECT_HAS_PERMIT
    if citizen == 1 and env.rules.id_card_required_for_citizens == 1 and rev["has_id_card"] == -1:
        return ACTION_INSPECT_HAS_ID_CARD

    if rev["is_worker"] == -1:
        return ACTION_INSPECT_IS_WORKER
    if rev["is_worker"] == 1 and env.rules.work_pass_required == 1 and rev["has_work_pass"] == -1:
        return ACTION_INSPECT_HAS_WORK_PASS

    if rev["has_permit"] == 1 and rev["name_match"] == -1:
        return ACTION_INSPECT_NAME_MATCH
    if rev["has_permit"] == 1 and rev["seal_valid"] == -1:
        return ACTION_INSPECT_SEAL_VALID

    deny_flags = [
        rev["country_allowed"] == 0,
        rev["expiry_valid"] == 0,
        rev["purpose_match"] == 0,
        rev["biometric_match"] == 0,
        rev["has_permit"] == 0 and citizen == 0 and env.rules.permit_required == 1,
        rev["has_id_card"] == 0 and citizen == 1 and env.rules.id_card_required_for_citizens == 1,
        rev["is_worker"] == 1 and rev["has_work_pass"] == 0 and env.rules.work_pass_required == 1,
        rev["has_permit"] == 1 and rev["name_match"] == 0,
        rev["has_permit"] == 1 and rev["seal_valid"] == 0,
    ]
    if any(deny_flags):
        return ACTION_DENY

    required_known = [
        rev["country_allowed"] != -1,
        rev["expiry_valid"] != -1,
        rev["purpose_match"] != -1,
        rev["biometric_match"] != -1,
    ]
    if citizen == 0 and env.rules.permit_required == 1:
        required_known.append(rev["has_permit"] != -1)
    if citizen == 1 and env.rules.id_card_required_for_citizens == 1:
        required_known.append(rev["has_id_card"] != -1)
    if rev["is_worker"] != -1 and rev["is_worker"] == 1 and env.rules.work_pass_required == 1:
        required_known.append(rev["has_work_pass"] != -1)
    if rev["has_permit"] == 1:
        required_known.append(rev["name_match"] != -1)
        required_known.append(rev["seal_valid"] != -1)

    if all(required_known):
        return ACTION_APPROVE

    for field, act in (
        ("has_permit", ACTION_INSPECT_HAS_PERMIT),
        ("has_id_card", ACTION_INSPECT_HAS_ID_CARD),
        ("has_work_pass", ACTION_INSPECT_HAS_WORK_PASS),
        ("name_match", ACTION_INSPECT_NAME_MATCH),
        ("seal_valid", ACTION_INSPECT_SEAL_VALID),
    ):
        if rev[field] == -1:
            return act

    return ACTION_DENY


def _collect_bc_dataset(env_kwargs: Dict[str, object], episodes: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    env = PapersPleaseEnv(**env_kwargs)
    obs_buf: List[np.ndarray] = []
    act_buf: List[int] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        step_guard = 0
        max_steps = env.day_len * (env.max_inspects_per_applicant + 2)

        while not done and step_guard < max_steps:
            action = _expert_action(env)
            obs_buf.append(np.asarray(obs, dtype=np.float32))
            act_buf.append(int(action))
            obs, _, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            step_guard += 1

        if (not done) and env.idx < len(env.queue):
            while env.idx < len(env.queue) and env.time_left > 0:
                forced = ACTION_APPROVE if oracle_is_legal(env.rules, env.queue[env.idx]) else ACTION_DENY
                obs_buf.append(np.asarray(obs, dtype=np.float32))
                act_buf.append(int(forced))
                obs, _, term, trunc, _ = env.step(forced)
                if term or trunc:
                    break

    return np.asarray(obs_buf, dtype=np.float32), np.asarray(act_buf, dtype=np.int64)


def _rebalance_bc_dataset(obs_np: np.ndarray, act_np: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Limit inspect dominance and upweight decision samples in BC dataset."""
    decision_mask = (act_np == ACTION_APPROVE) | (act_np == ACTION_DENY)
    inspect_mask = ~decision_mask

    decision_idx = np.flatnonzero(decision_mask)
    inspect_idx = np.flatnonzero(inspect_mask)
    if decision_idx.size == 0 or inspect_idx.size == 0:
        return obs_np, act_np

    max_inspects = min(inspect_idx.size, max(decision_idx.size, 2 * decision_idx.size))
    rng = np.random.default_rng(seed)
    keep_inspect = rng.choice(inspect_idx, size=max_inspects, replace=False)
    keep_idx = np.concatenate([decision_idx, keep_inspect])
    rng.shuffle(keep_idx)

    return obs_np[keep_idx], act_np[keep_idx]


def _run_bc_pretrain(
    model: PPO,
    env_kwargs: Dict[str, object],
    episodes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    """Behavior cloning warm-start on PPO policy network."""
    obs_np, act_np = _collect_bc_dataset(env_kwargs=env_kwargs, episodes=episodes, seed=seed)
    obs_np, act_np = _rebalance_bc_dataset(obs_np=obs_np, act_np=act_np, seed=seed + 7)

    if obs_np.shape[0] == 0:
        print("[bc] dataset is empty, skipping pretrain")
        return

    decision_count = int(((act_np == ACTION_APPROVE) | (act_np == ACTION_DENY)).sum())
    inspect_count = int(act_np.shape[0] - decision_count)
    print(f"[bc] dataset samples={act_np.shape[0]} decisions={decision_count} inspects={inspect_count}")

    device = model.device
    obs_t = th.as_tensor(obs_np, dtype=th.float32, device=device)
    act_t = th.as_tensor(act_np, dtype=th.long, device=device)

    optimizer = th.optim.Adam(model.policy.parameters(), lr=lr)
    n = obs_t.shape[0]

    model.policy.train()
    for ep in range(max(1, epochs)):
        idx = th.randperm(n, device=device)
        total_loss = 0.0
        batches = 0

        for start in range(0, n, max(1, batch_size)):
            bidx = idx[start : start + max(1, batch_size)]
            b_obs = obs_t[bidx]
            b_act = act_t[bidx]

            dist = model.policy.get_distribution(b_obs)
            log_prob = dist.log_prob(b_act)
            is_decision = ((b_act == ACTION_APPROVE) | (b_act == ACTION_DENY)).float()
            sample_weight = 1.0 + 2.0 * is_decision
            loss = -(log_prob * sample_weight).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.detach().item())
            batches += 1

        avg_loss = total_loss / max(1, batches)
        print(f"[bc] epoch={ep + 1}/{epochs} loss={avg_loss:.4f} samples={n}")


def main() -> None:
    """Create env/model, optional BC warm-start, then PPO training and eval."""
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

    if args.bc_pretrain_episodes > 0:
        print(
            f"[bc] pretrain start episodes={args.bc_pretrain_episodes} "
            f"epochs={args.bc_epochs} batch_size={args.bc_batch_size} lr={args.bc_lr}"
        )
        _run_bc_pretrain(
            model=model,
            env_kwargs=env_kwargs,
            episodes=args.bc_pretrain_episodes,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            lr=args.bc_lr,
            seed=args.seed + 50_000,
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






