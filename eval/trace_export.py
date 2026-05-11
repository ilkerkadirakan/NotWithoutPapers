from __future__ import annotations

"""Replay trace export utilities for animated UI consumption."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

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
from env.domain import Applicant, Rules

TRACE_VERSION = "1.0"
MANIFEST_FILENAME = "traces_manifest.json"

ACTION_NAMES = {
    ACTION_APPROVE: "APPROVE",
    ACTION_DENY: "DENY",
    ACTION_INSPECT_COUNTRY_ALLOWED: "INSPECT_COUNTRY_ALLOWED",
    ACTION_INSPECT_HAS_PERMIT: "INSPECT_HAS_PERMIT",
    ACTION_INSPECT_EXPIRY_VALID: "INSPECT_EXPIRY_VALID",
    ACTION_INSPECT_NAME_MATCH: "INSPECT_NAME_MATCH",
    ACTION_INSPECT_HAS_ID_CARD: "INSPECT_HAS_ID_CARD",
    ACTION_INSPECT_IS_WORKER: "INSPECT_IS_WORKER",
    ACTION_INSPECT_HAS_WORK_PASS: "INSPECT_HAS_WORK_PASS",
    ACTION_INSPECT_PURPOSE_MATCH: "INSPECT_PURPOSE_MATCH",
    ACTION_INSPECT_SEAL_VALID: "INSPECT_SEAL_VALID",
    ACTION_INSPECT_BIOMETRIC_MATCH: "INSPECT_BIOMETRIC_MATCH",
}

INSPECT_ACTIONS = {
    ACTION_INSPECT_COUNTRY_ALLOWED,
    ACTION_INSPECT_HAS_PERMIT,
    ACTION_INSPECT_EXPIRY_VALID,
    ACTION_INSPECT_NAME_MATCH,
    ACTION_INSPECT_HAS_ID_CARD,
    ACTION_INSPECT_IS_WORKER,
    ACTION_INSPECT_HAS_WORK_PASS,
    ACTION_INSPECT_PURPOSE_MATCH,
    ACTION_INSPECT_SEAL_VALID,
    ACTION_INSPECT_BIOMETRIC_MATCH,
}


class PredictModel(Protocol):
    """Minimal protocol required from SB3-like models."""

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[Any, Any]:
        ...


def action_id_to_name(action_id: int) -> str:
    """Return stable action name for a numeric action id."""
    return ACTION_NAMES.get(int(action_id), f"UNKNOWN_{int(action_id)}")


def action_type(action_id: int) -> str:
    """Return high-level action category for UI styling."""
    aid = int(action_id)
    if aid == ACTION_APPROVE:
        return "approve"
    if aid == ACTION_DENY:
        return "deny"
    if aid in INSPECT_ACTIONS:
        return "inspect"
    return "unknown"


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _snapshot_rules(rules: Optional[Rules]) -> Optional[Dict[str, Any]]:
    if rules is None:
        return None

    allowed_mask = [int(x) for x in rules.allowed_countries_mask.tolist()]
    allowed_by_country = {COUNTRIES[i]: allowed_mask[i] for i in range(len(COUNTRIES))}
    return {
        "allowed_countries_mask": allowed_mask,
        "allowed_countries_by_name": allowed_by_country,
        "permit_required": int(rules.permit_required),
        "id_card_required_for_citizens": int(rules.id_card_required_for_citizens),
        "work_pass_required": int(rules.work_pass_required),
    }


def _snapshot_revealed(revealed: Dict[str, int]) -> Dict[str, int]:
    return {k: int(v) for k, v in revealed.items()}


def _snapshot_applicant(app: Applicant) -> Dict[str, Any]:
    return {
        "country_idx": int(app.country_idx),
        "country_name": COUNTRIES[int(app.country_idx)],
    }


def _stats_delta_decision_result(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    if int(after.get("false_accept", 0)) > int(before.get("false_accept", 0)):
        return "false_accept"
    if int(after.get("false_reject", 0)) > int(before.get("false_reject", 0)):
        return "false_reject"

    before_decisions = int(before.get("approves", 0)) + int(before.get("denies", 0))
    after_decisions = int(after.get("approves", 0)) + int(after.get("denies", 0))
    if after_decisions > before_decisions:
        return "correct"
    return "none"


def _trace_filename(trace_prefix: str, seed: int, episode_id: int) -> str:
    safe_prefix = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in trace_prefix)
    return f"{safe_prefix}_seed{seed}_ep{episode_id:03d}.json"


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_episode_trace(model: PredictModel, env: PapersPleaseEnv, seed: int, episode_id: int) -> Dict[str, Any]:
    """
    Roll one deterministic episode and return step-by-step trace payload.

    This function contains no file I/O and is suitable for testing.
    """
    obs, _ = env.reset(seed=seed)
    done = False
    step_idx = 0
    total_reward = 0.0
    steps: List[Dict[str, Any]] = []
    terminated = False
    truncated = False
    final_info: Dict[str, Any] = {}

    while not done:
        assert env.rules is not None
        assert 0 <= env.idx < len(env.queue)

        rules_before = _snapshot_rules(env.rules)
        revealed_before = _snapshot_revealed(env.revealed)
        applicant_before = _snapshot_applicant(env.queue[env.idx])
        stats_before = dict(env.stats)
        time_left_before = int(env.time_left)
        applicant_idx_before = int(env.idx)

        predicted_action, _ = model.predict(obs, deterministic=True)
        action_id = int(predicted_action)

        obs, reward, terminated, truncated, info = env.step(action_id)
        final_info = info
        total_reward += float(reward)
        done = bool(terminated or truncated)

        rules_after = _snapshot_rules(env.rules)
        revealed_after = _snapshot_revealed(env.revealed)
        stats_after = dict(env.stats)
        applicant_idx_after = int(info.get("idx", env.idx))
        time_left_after = int(info.get("time_left", env.time_left))
        decision = _stats_delta_decision_result(stats_before, stats_after)

        if terminated:
            done_state: str | bool = "terminated"
            applicant_after = None
        elif truncated:
            done_state = "truncated"
            applicant_after = None
        else:
            done_state = False
            applicant_after = _snapshot_applicant(env.queue[env.idx])

        steps.append(
            {
                "step_idx": step_idx,
                "time_left_before": time_left_before,
                "time_left_after": time_left_after,
                "applicant_idx_before": applicant_idx_before,
                "applicant_idx_after": applicant_idx_after,
                "action_id": action_id,
                "action_name": action_id_to_name(action_id),
                "action_type": action_type(action_id),
                "reward": float(reward),
                "revealed_before": revealed_before,
                "revealed_after": revealed_after,
                "rules_before": rules_before,
                "rules_after": rules_after,
                "rule_update_event": info.get("rule_update"),
                "decision_result": decision,
                "done": done_state,
                "applicant_before": applicant_before,
                "applicant_after": applicant_after,
            }
        )
        step_idx += 1

    episode_stats = final_info.get("episode_stats", dict(env.stats))
    stats = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in episode_stats.items()}
    return {
        "episode": {
            "episode_id": int(episode_id),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "total_reward": float(total_reward),
            "stats": stats,
        },
        "steps": steps,
    }


def _manifest_entry_from_trace_payload(filename: str, trace: Dict[str, Any]) -> Dict[str, Any]:
    meta = trace.get("meta", {})
    episode = trace.get("episode", {})
    return {
        "id": Path(filename).stem,
        "file": filename,
        "seed": int(meta.get("seed", 0)),
        "episode_id": int(episode.get("episode_id", 0)),
        "model_path": str(meta.get("model_path", "")),
        "total_reward": float(episode.get("total_reward", 0.0)),
        "terminated": bool(episode.get("terminated", False)),
        "truncated": bool(episode.get("truncated", False)),
    }


def write_traces_manifest(output_dir: Path) -> Dict[str, Any]:
    """Scan trace directory and rewrite `traces_manifest.json`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME

    entries: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name == MANIFEST_FILENAME:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict) or "episode" not in payload or "meta" not in payload:
            continue
        entries.append(_manifest_entry_from_trace_payload(path.name, payload))

    manifest = {
        "trace_version": TRACE_VERSION,
        "generated_at": _iso_utc_now(),
        "traces": entries,
    }
    _json_dump(manifest_path, manifest)
    return manifest


def export_replay_traces(
    model: PredictModel,
    *,
    episodes: int,
    seed: int,
    output_dir: Path,
    model_path: str,
    trace_prefix: str = "trace",
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Export one or more deterministic episode traces and update directory manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = PapersPleaseEnv(**dict(env_kwargs or {}))
    env_config = {k: v for k, v in dict(env_kwargs or {}).items()}

    for episode_id in range(int(episodes)):
        run_seed = int(seed) + episode_id
        episode_payload = build_episode_trace(model=model, env=env, seed=run_seed, episode_id=episode_id)
        trace_payload = {
            "meta": {
                "trace_version": TRACE_VERSION,
                "model_path": model_path,
                "seed": run_seed,
                "generated_at": _iso_utc_now(),
                "env_config": env_config,
            },
            "episode": episode_payload["episode"],
            "steps": episode_payload["steps"],
        }
        filename = _trace_filename(trace_prefix=trace_prefix, seed=run_seed, episode_id=episode_id)
        _json_dump(output_dir / filename, trace_payload)

    return write_traces_manifest(output_dir)
