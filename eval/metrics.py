from __future__ import annotations

"""Metric aggregation helpers used by training and evaluation flows."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class EvalSummary:
    """Compact metric set for comparing policy runs."""

    mean_reward: float
    decision_accuracy: float
    false_accept_rate: float
    false_reject_rate: float
    inspection_frequency: float


def summarize_episode_stats(rewards: List[float], stats_list: List[Dict[str, float]]) -> EvalSummary:
    """Aggregate raw episode counters into project KPI metrics."""
    if not stats_list:
        return EvalSummary(0.0, 0.0, 0.0, 0.0, 0.0)

    total_approves = sum(int(s.get("approves", 0)) for s in stats_list)
    total_denies = sum(int(s.get("denies", 0)) for s in stats_list)
    total_decisions = max(1, total_approves + total_denies)

    total_false_accept = sum(int(s.get("false_accept", 0)) for s in stats_list)
    total_false_reject = sum(int(s.get("false_reject", 0)) for s in stats_list)
    total_inspects = sum(int(s.get("inspects", 0)) for s in stats_list)

    correct_decisions = total_decisions - total_false_accept - total_false_reject
    decision_accuracy = correct_decisions / total_decisions
    false_accept_rate = total_false_accept / total_decisions
    false_reject_rate = total_false_reject / total_decisions
    inspection_frequency = total_inspects / total_decisions
    mean_reward = float(np.mean(rewards)) if rewards else 0.0

    return EvalSummary(
        mean_reward=mean_reward,
        decision_accuracy=decision_accuracy,
        false_accept_rate=false_accept_rate,
        false_reject_rate=false_reject_rate,
        inspection_frequency=inspection_frequency,
    )
