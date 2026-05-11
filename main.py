from __future__ import annotations

"""Single CLI entrypoint for smoke, train and evaluation workflows.

Examples:
- python main.py smoke --skip-pytest
- python main.py train --total-timesteps 200000
- python main.py eval --model-path artifacts/ppo_papers_please.zip
- python main.py trace --model-path artifacts/sweep_s42_200k_cont.zip
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def _run_train(args: argparse.Namespace) -> int:
    """Delegate training to `train.train_ppo` module with the same CLI args."""
    cmd = [
        sys.executable,
        "-m",
        "train.train_ppo",
        "--total-timesteps",
        str(args.total_timesteps),
        "--n-envs",
        str(args.n_envs),
        "--seed",
        str(args.seed),
        "--eval-episodes",
        str(args.eval_episodes),
        "--save-path",
        str(args.save_path),
        "--print-every",
        str(args.print_every),
        "--ent-coef",
        str(args.ent_coef),
        "--day-len",
        str(args.day_len),
        "--time-budget",
        str(args.time_budget),
        "--fraud-rate-min",
        str(args.fraud_rate_min),
        "--fraud-rate-max",
        str(args.fraud_rate_max),
        "--mid-day-update-prob",
        str(args.mid_day_update_prob),
        "--inspect-error-prob",
        str(args.inspect_error_prob),
        "--inspect-miss-prob",
        str(args.inspect_miss_prob),
        "--max-inspects-per-applicant",
        str(args.max_inspects_per_applicant),
        "--decision-coverage-target",
        str(args.decision_coverage_target),
        "--coverage-shortfall-penalty",
        str(args.coverage_shortfall_penalty),
        "--coverage-hard-threshold",
        str(args.coverage_hard_threshold),
        "--coverage-hard-penalty",
        str(args.coverage_hard_penalty),
        "--r-correct",
        str(args.r_correct),
        "--p-false-accept",
        str(args.p_false_accept),
        "--p-false-reject",
        str(args.p_false_reject),
        "--c-inspect",
        str(args.c_inspect),
        "--p-overinspect",
        str(args.p_overinspect),
        "--p-reinspect",
        str(args.p_reinspect),
        "--p-approve-without-inspect",
        str(args.p_approve_without_inspect),
        "--p-undecided",
        str(args.p_undecided),
    ]
    if args.use_curriculum:
        cmd.append("--use-curriculum")
    cmd.extend(
        [
            "--stage-a-timesteps",
            str(args.stage_a_timesteps),
        ]
    )
    if args.progress_bar:
        cmd.append("--progress-bar")
    result = subprocess.run(cmd, check=False)
    return int(result.returncode)


def _run_eval(args: argparse.Namespace) -> int:
    """Load a saved PPO model and print deterministic evaluation metrics."""
    from stable_baselines3 import PPO

    from eval.evaluate import evaluate_model

    model_path = args.model_path
    if model_path.suffix == ".zip":
        if model_path.exists():
            load_path = str(model_path.with_suffix(""))
        else:
            load_path = str(model_path)
    else:
        load_path = str(model_path)

    model = PPO.load(load_path)
    summary = evaluate_model(model=model, episodes=args.episodes, seed=args.seed)

    print("[eval] deterministic policy summary")
    print(f"model_path            : {args.model_path}")
    print(f"episodes              : {args.episodes}")
    print(f"episode reward        : {summary.mean_reward:.3f}")
    print(f"decision accuracy     : {summary.decision_accuracy:.3f}")
    print(f"decision coverage     : {summary.decision_coverage:.3f}")
    print(f"false accept rate     : {summary.false_accept_rate:.3f}")
    print(f"false reject rate     : {summary.false_reject_rate:.3f}")
    print(f"inspection frequency  : {summary.inspection_frequency:.3f}")
    return 0


def _run_trace(args: argparse.Namespace) -> int:
    """Export deterministic rollout traces for replay UI."""
    from stable_baselines3 import PPO

    from eval.trace_export import export_replay_traces, write_traces_manifest

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = _build_env_kwargs_from_args(args)
    model_path = args.model_path

    if model_path.suffix == ".zip":
        load_path = str(model_path.with_suffix("")) if model_path.exists() else str(model_path)
        model_exists = model_path.exists() or model_path.with_suffix("").exists()
    else:
        load_path = str(model_path)
        model_exists = model_path.exists() or model_path.with_suffix(".zip").exists()

    if not model_exists:
        if args.allow_missing_model:
            manifest = write_traces_manifest(output_dir=output_dir)
            print("[trace] model not found, skipped trace export.")
            print(f"[trace] output_dir          : {output_dir}")
            print(f"[trace] manifest entries    : {len(manifest.get('traces', []))}")
            return 0
        print(f"[trace] model not found: {model_path}")
        return 1

    model = PPO.load(load_path)
    manifest = export_replay_traces(
        model=model,
        episodes=args.episodes,
        seed=args.seed,
        output_dir=output_dir,
        model_path=str(model_path),
        trace_prefix=args.trace_prefix,
        env_kwargs=env_kwargs,
    )
    print("[trace] replay traces exported")
    print(f"model_path            : {model_path}")
    print(f"episodes              : {args.episodes}")
    print(f"output_dir            : {output_dir}")
    print(f"manifest entries      : {len(manifest.get('traces', []))}")
    return 0


def _run_smoke(args: argparse.Namespace) -> int:
    """Run env smoke test and optionally run pytest."""
    from scripts.smoke_test import main as smoke_main

    smoke_main()

    if args.skip_pytest:
        return 0

    if importlib.util.find_spec("pytest") is None:
        print("pytest is not installed in this environment.")
        print("Install dependencies with:")
        print("  python -m pip install -r requirements.txt")
        return 1

    result = subprocess.run([sys.executable, "-m", "pytest"], check=False)
    return int(result.returncode)


def _build_env_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Build environment kwargs from shared CLI parameters."""
    return {
        "day_len": args.day_len,
        "time_budget": args.time_budget,
        "fraud_rate_range": (args.fraud_rate_min, args.fraud_rate_max),
        "mid_day_update_prob": args.mid_day_update_prob,
        "inspect_error_prob": args.inspect_error_prob,
        "inspect_miss_prob": args.inspect_miss_prob,
        "max_inspects_per_applicant": args.max_inspects_per_applicant,
        "decision_coverage_target": args.decision_coverage_target,
        "coverage_shortfall_penalty": args.coverage_shortfall_penalty,
        "coverage_hard_threshold": args.coverage_hard_threshold,
        "coverage_hard_penalty": args.coverage_hard_penalty,
        "r_correct": args.r_correct,
        "p_false_accept": args.p_false_accept,
        "p_false_reject": args.p_false_reject,
        "c_inspect": args.c_inspect,
        "p_overinspect": args.p_overinspect,
        "p_reinspect": args.p_reinspect,
        "p_approve_without_inspect": args.p_approve_without_inspect,
        "p_undecided": args.p_undecided,
        "seed": args.seed,
    }


def _add_train_env_args(train_parser: argparse.ArgumentParser) -> None:
    """Add environment and reward parameter overrides for training."""
    train_parser.add_argument("--day-len", type=int, default=25)
    train_parser.add_argument("--time-budget", type=int, default=60)
    train_parser.add_argument("--fraud-rate-min", type=float, default=0.15)
    train_parser.add_argument("--fraud-rate-max", type=float, default=0.35)
    train_parser.add_argument("--mid-day-update-prob", type=float, default=0.6)
    train_parser.add_argument("--inspect-error-prob", type=float, default=0.0)
    train_parser.add_argument("--inspect-miss-prob", type=float, default=0.0)
    train_parser.add_argument("--max-inspects-per-applicant", type=int, default=3)
    train_parser.add_argument("--decision-coverage-target", type=float, default=0.9)
    train_parser.add_argument("--coverage-shortfall-penalty", type=float, default=-20.0)
    train_parser.add_argument("--coverage-hard-threshold", type=float, default=0.9)
    train_parser.add_argument("--coverage-hard-penalty", type=float, default=-120.0)
    train_parser.add_argument("--r-correct", type=float, default=6.0)
    train_parser.add_argument("--p-false-accept", type=float, default=-15.0)
    train_parser.add_argument("--p-false-reject", type=float, default=-15.0)
    train_parser.add_argument("--c-inspect", type=float, default=-0.1)
    train_parser.add_argument("--p-overinspect", type=float, default=-2.0)
    train_parser.add_argument("--p-reinspect", type=float, default=-0.5)
    train_parser.add_argument("--p-approve-without-inspect", type=float, default=-2.0)
    train_parser.add_argument("--p-undecided", type=float, default=0.0)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser with `train`, `eval`, and `smoke` subcommands."""
    parser = argparse.ArgumentParser(description="Project entrypoint for smoke/train/eval workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train PPO baseline.")
    train_parser.add_argument("--total-timesteps", type=int, default=200_000)
    train_parser.add_argument("--n-envs", type=int, default=8)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--eval-episodes", type=int, default=100)
    train_parser.add_argument("--save-path", type=Path, default=Path("artifacts/ppo_papers_please.zip"))
    train_parser.add_argument("--print-every", type=int, default=50)
    train_parser.add_argument("--progress-bar", action="store_true")
    train_parser.add_argument("--ent-coef", type=float, default=0.0)
    train_parser.add_argument("--use-curriculum", action="store_true")
    train_parser.add_argument("--stage-a-timesteps", type=int, default=80_000)
    _add_train_env_args(train_parser)
    train_parser.set_defaults(func=_run_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a saved PPO model.")
    eval_parser.add_argument("--model-path", type=Path, default=Path("artifacts/ppo_papers_please.zip"))
    eval_parser.add_argument("--episodes", type=int, default=100)
    eval_parser.add_argument("--seed", type=int, default=10_042)
    eval_parser.set_defaults(func=_run_eval)

    trace_parser = subparsers.add_parser("trace", help="Export deterministic replay traces for UI.")
    trace_parser.add_argument("--model-path", type=Path, default=Path("artifacts/sweep_s42_200k_cont.zip"))
    trace_parser.add_argument("--episodes", type=int, default=1)
    trace_parser.add_argument("--seed", type=int, default=10_042)
    trace_parser.add_argument("--output-dir", type=Path, default=Path("ui/public/traces"))
    trace_parser.add_argument("--trace-prefix", type=str, default="trace")
    trace_parser.add_argument("--allow-missing-model", action="store_true")
    _add_train_env_args(trace_parser)
    trace_parser.set_defaults(func=_run_trace)

    smoke_parser = subparsers.add_parser("smoke", help="Run smoke checks.")
    smoke_parser.add_argument("--skip-pytest", action="store_true", help="Run only environment smoke test.")
    smoke_parser.set_defaults(func=_run_smoke)

    return parser


def main() -> int:
    """Parse CLI args and execute selected subcommand."""
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
