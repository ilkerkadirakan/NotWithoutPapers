from __future__ import annotations

"""No-inspect leakage/difficulty audit for PapersPleaseEnv.

Estimates how much performance is achievable without any inspect action,
using only rule + country priors.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

from env import PapersPleaseEnv
from env.domain import oracle_is_legal


Key = Tuple[int, ...]


@dataclass
class Stats:
    total: int = 0
    legal: int = 0
    illegal: int = 0
    correct: int = 0
    false_accept: int = 0
    false_reject: int = 0

    def update_truth(self, is_legal: bool) -> None:
        self.total += 1
        if is_legal:
            self.legal += 1
        else:
            self.illegal += 1

    def update_pred(self, pred_approve: bool, is_legal: bool) -> None:
        if pred_approve and is_legal:
            self.correct += 1
        elif pred_approve and (not is_legal):
            self.false_accept += 1
        elif (not pred_approve) and is_legal:
            self.false_reject += 1
        else:
            self.correct += 1

    def summary(self) -> Dict[str, float]:
        denom = max(1, self.total)
        return {
            "n": float(self.total),
            "accuracy": self.correct / denom,
            "false_accept_rate": self.false_accept / denom,
            "false_reject_rate": self.false_reject / denom,
            "legal_ratio": self.legal / denom,
            "illegal_ratio": self.illegal / denom,
        }


def make_key(env: PapersPleaseEnv, idx: int) -> Key:
    """State key available without inspect actions."""
    assert env.rules is not None
    app = env.queue[idx]
    return tuple(int(x) for x in env.rules.allowed_countries_mask.tolist()) + (
        int(env.rules.permit_required),
        int(env.rules.id_card_required_for_citizens),
        int(env.rules.work_pass_required),
        int(app.country_idx),
    )


def collect_priors(env: PapersPleaseEnv, episodes: int, seed: int) -> Dict[Key, Tuple[int, int]]:
    """Return mapping key -> (legal_count, total_count)."""
    priors: Dict[Key, Tuple[int, int]] = {}
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        assert env.rules is not None
        for i, app in enumerate(env.queue):
            key = make_key(env, i)
            legal = int(oracle_is_legal(env.rules, app))
            pos, tot = priors.get(key, (0, 0))
            priors[key] = (pos + legal, tot + 1)
    return priors


def evaluate_no_inspect_policy(
    env: PapersPleaseEnv,
    episodes: int,
    seed: int,
    priors: Dict[Key, Tuple[int, int]],
) -> Tuple[Stats, int]:
    stats = Stats()
    unseen = 0

    global_pos = sum(v[0] for v in priors.values())
    global_tot = sum(v[1] for v in priors.values())
    global_p = (global_pos / global_tot) if global_tot > 0 else 0.5

    for ep in range(episodes):
        env.reset(seed=seed + ep)
        assert env.rules is not None
        for i, app in enumerate(env.queue):
            key = make_key(env, i)
            legal = bool(oracle_is_legal(env.rules, app))
            stats.update_truth(legal)

            if key in priors:
                p_legal = priors[key][0] / max(1, priors[key][1])
            else:
                unseen += 1
                p_legal = global_p

            pred_approve = p_legal >= 0.5
            stats.update_pred(pred_approve, legal)

    return stats, unseen


def evaluate_constant_policy(env: PapersPleaseEnv, episodes: int, seed: int, approve: bool) -> Stats:
    stats = Stats()
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        assert env.rules is not None
        for app in env.queue:
            legal = bool(oracle_is_legal(env.rules, app))
            stats.update_truth(legal)
            stats.update_pred(approve, legal)
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit no-inspect predictability in PapersPleaseEnv.")
    p.add_argument("--train-episodes", type=int, default=3000)
    p.add_argument("--test-episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--day-len", type=int, default=25)
    p.add_argument("--time-budget", type=int, default=60)
    p.add_argument("--fraud-rate-min", type=float, default=0.15)
    p.add_argument("--fraud-rate-max", type=float, default=0.35)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = PapersPleaseEnv(
        day_len=args.day_len,
        time_budget=args.time_budget,
        fraud_rate_range=(args.fraud_rate_min, args.fraud_rate_max),
        seed=args.seed,
    )

    priors = collect_priors(env, episodes=args.train_episodes, seed=args.seed)
    no_inspect_stats, unseen = evaluate_no_inspect_policy(
        env,
        episodes=args.test_episodes,
        seed=args.seed + 100_000,
        priors=priors,
    )
    always_approve = evaluate_constant_policy(env, episodes=args.test_episodes, seed=args.seed + 200_000, approve=True)
    always_deny = evaluate_constant_policy(env, episodes=args.test_episodes, seed=args.seed + 300_000, approve=False)

    s0 = no_inspect_stats.summary()
    s1 = always_approve.summary()
    s2 = always_deny.summary()

    print("[audit] no-inspect predictability")
    print(f"train_episodes: {args.train_episodes}")
    print(f"test_episodes : {args.test_episodes}")
    print(f"state_keys    : {len(priors)}")
    print(f"unseen_keys   : {unseen}")

    print("\n[policy] empirical no-inspect prior")
    print(f"accuracy          : {s0['accuracy']:.3f}")
    print(f"false_accept_rate : {s0['false_accept_rate']:.3f}")
    print(f"false_reject_rate : {s0['false_reject_rate']:.3f}")

    print("\n[baseline] always APPROVE")
    print(f"accuracy          : {s1['accuracy']:.3f}")
    print(f"false_accept_rate : {s1['false_accept_rate']:.3f}")
    print(f"false_reject_rate : {s1['false_reject_rate']:.3f}")

    print("\n[baseline] always DENY")
    print(f"accuracy          : {s2['accuracy']:.3f}")
    print(f"false_accept_rate : {s2['false_accept_rate']:.3f}")
    print(f"false_reject_rate : {s2['false_reject_rate']:.3f}")


if __name__ == "__main__":
    main()
