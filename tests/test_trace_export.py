from __future__ import annotations

"""Trace export contract tests for replay UI integration."""

import json
from typing import Any

from eval.trace_export import export_replay_traces, write_traces_manifest


class AlwaysApproveModel:
    """Simple deterministic model stub for exporter tests."""

    def predict(self, observation: Any, deterministic: bool = True):
        return 0, None


def _env_kwargs(seed: int) -> dict[str, object]:
    return {
        "day_len": 6,
        "time_budget": 20,
        "fraud_rate_range": (0.15, 0.35),
        "mid_day_update_prob": 0.0,
        "inspect_error_prob": 0.0,
        "inspect_miss_prob": 0.0,
        "max_inspects_per_applicant": 3,
        "decision_coverage_target": 0.0,
        "coverage_shortfall_penalty": -20.0,
        "coverage_hard_threshold": 0.0,
        "coverage_hard_penalty": -120.0,
        "r_correct": 6.0,
        "p_false_accept": -15.0,
        "p_false_reject": -15.0,
        "c_inspect": -0.1,
        "p_overinspect": -2.0,
        "p_reinspect": -0.5,
        "p_approve_without_inspect": -2.0,
        "p_undecided": 0.0,
        "seed": seed,
    }


def _read_trace_file(output_dir, manifest: dict[str, object]) -> dict[str, object]:
    traces = manifest.get("traces", [])
    assert isinstance(traces, list)
    assert traces
    trace_file = output_dir / traces[0]["file"]
    return json.loads(trace_file.read_text(encoding="utf-8"))


def test_trace_export_schema_fields(tmp_path) -> None:
    output_dir = tmp_path / "trace_schema"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = export_replay_traces(
        model=AlwaysApproveModel(),
        episodes=1,
        seed=123,
        output_dir=output_dir,
        model_path="dummy_model.zip",
        trace_prefix="schema",
        env_kwargs=_env_kwargs(seed=123),
    )
    payload = _read_trace_file(output_dir=output_dir, manifest=manifest)

    assert set(payload.keys()) == {"meta", "episode", "steps"}
    assert payload["meta"]["trace_version"] == "1.0"
    assert isinstance(payload["steps"], list)
    assert len(payload["steps"]) > 0

    step0 = payload["steps"][0]
    required = {
        "step_idx",
        "time_left_before",
        "time_left_after",
        "applicant_idx_before",
        "applicant_idx_after",
        "action_id",
        "action_name",
        "action_type",
        "reward",
        "revealed_before",
        "revealed_after",
        "rules_before",
        "rules_after",
        "rule_update_event",
        "decision_result",
        "done",
    }
    assert required.issubset(step0.keys())
    assert step0["action_id"] == 0
    assert step0["action_name"] == "APPROVE"
    assert step0["action_type"] == "approve"


def test_trace_export_is_deterministic_for_same_seed(tmp_path) -> None:
    dir_a = tmp_path / "trace_det_a"
    dir_b = tmp_path / "trace_det_b"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    common = dict(
        model=AlwaysApproveModel(),
        episodes=1,
        seed=777,
        model_path="dummy_model.zip",
        trace_prefix="det",
        env_kwargs=_env_kwargs(seed=777),
    )
    manifest_a = export_replay_traces(output_dir=dir_a, **common)
    manifest_b = export_replay_traces(output_dir=dir_b, **common)
    trace_a = _read_trace_file(output_dir=dir_a, manifest=manifest_a)
    trace_b = _read_trace_file(output_dir=dir_b, manifest=manifest_b)

    actions_a = [int(s["action_id"]) for s in trace_a["steps"]]
    actions_b = [int(s["action_id"]) for s in trace_b["steps"]]
    rewards_a = [float(s["reward"]) for s in trace_a["steps"]]
    rewards_b = [float(s["reward"]) for s in trace_b["steps"]]

    assert actions_a == actions_b
    assert rewards_a == rewards_b
    assert float(trace_a["episode"]["total_reward"]) == float(trace_b["episode"]["total_reward"])
    assert len(trace_a["steps"]) == len(trace_b["steps"])


def test_manifest_lists_exported_trace_files(tmp_path) -> None:
    output_dir = tmp_path / "trace_manifest"
    output_dir.mkdir(parents=True, exist_ok=True)

    export_replay_traces(
        model=AlwaysApproveModel(),
        episodes=2,
        seed=901,
        output_dir=output_dir,
        model_path="dummy_model.zip",
        trace_prefix="manifest",
        env_kwargs=_env_kwargs(seed=901),
    )
    manifest = write_traces_manifest(output_dir)

    assert manifest["trace_version"] == "1.0"
    traces = manifest["traces"]
    assert isinstance(traces, list)
    assert len(traces) == 2
    files = [t["file"] for t in traces]
    assert any("manifest_seed901_ep000.json" == f for f in files)
    assert any("manifest_seed902_ep001.json" == f for f in files)

    manifest_path = output_dir / "traces_manifest.json"
    assert manifest_path.exists()
