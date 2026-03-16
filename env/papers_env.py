from __future__ import annotations

"""Gymnasium environment implementation for the Papers Please RL task."""

import math
import random
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .constants import (
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
    FIELDS,
    N_ACTIONS,
)
from .domain import Applicant, Rules, oracle_is_legal
from .sampling import build_queue_with_deny_band, sample_rules


class PapersPleaseEnv(gym.Env):
    """
    Episode = 1 workday with a queue.
    Partial observability: fields are UNKNOWN until inspected.

    Action space (Discrete):
      0 APPROVE
      1 DENY
      2 INSPECT_COUNTRY_ALLOWED
      3 INSPECT_HAS_PERMIT
      4 INSPECT_EXPIRY_VALID
      5 INSPECT_NAME_MATCH
      6 INSPECT_HAS_ID_CARD
      7 INSPECT_IS_WORKER
      8 INSPECT_HAS_WORK_PASS
      9 INSPECT_PURPOSE_MATCH
     10 INSPECT_SEAL_VALID
     11 INSPECT_BIOMETRIC_MATCH
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        day_len: int = 25,
        time_budget: int = 60,
        fraud_rate_range: Tuple[float, float] = (0.15, 0.35),
        mid_day_update_prob: float = 0.6,
        inspect_error_prob: float = 0.0,
        inspect_miss_prob: float = 0.0,
        max_inspects_per_applicant: int = 3,
        decision_coverage_target: float = 0.9,
        coverage_shortfall_penalty: float = -20.0,
        coverage_hard_threshold: float = 0.9,
        coverage_hard_penalty: float = -120.0,
        r_correct: float = 6.0,
        p_false_accept: float = -15.0,
        p_false_reject: float = -15.0,
        c_inspect: float = -0.1,
        p_overinspect: float = -2.0,
        p_reinspect: float = -0.5,
        p_undecided: float = 0.0,
        debug: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if int(day_len) < 1:
            raise ValueError("day_len must be >= 1")
        if int(time_budget) < 1:
            raise ValueError("time_budget must be >= 1")
        fr_min, fr_max = fraud_rate_range
        if not (0.0 <= fr_min <= fr_max <= 1.0):
            raise ValueError("fraud_rate_range must satisfy 0.0 <= min <= max <= 1.0")
        if not (0.0 <= float(mid_day_update_prob) <= 1.0):
            raise ValueError("mid_day_update_prob must be in [0.0, 1.0]")
        if not (0.0 <= float(inspect_error_prob) <= 1.0):
            raise ValueError("inspect_error_prob must be in [0.0, 1.0]")
        if not (0.0 <= float(inspect_miss_prob) <= 1.0):
            raise ValueError("inspect_miss_prob must be in [0.0, 1.0]")
        if float(inspect_error_prob) + float(inspect_miss_prob) > 1.0:
            raise ValueError("inspect_error_prob + inspect_miss_prob must be <= 1.0")
        if int(max_inspects_per_applicant) < 1:
            raise ValueError("max_inspects_per_applicant must be >= 1")
        if not (0.0 <= float(decision_coverage_target) <= 1.0):
            raise ValueError("decision_coverage_target must be in [0.0, 1.0]")
        if float(coverage_shortfall_penalty) > 0.0:
            raise ValueError("coverage_shortfall_penalty must be <= 0.0")
        if not (0.0 <= float(coverage_hard_threshold) <= 1.0):
            raise ValueError("coverage_hard_threshold must be in [0.0, 1.0]")
        if float(coverage_hard_penalty) > 0.0:
            raise ValueError("coverage_hard_penalty must be <= 0.0")
        if float(p_false_accept) > 0.0:
            raise ValueError("p_false_accept must be <= 0.0")
        if float(p_false_reject) > 0.0:
            raise ValueError("p_false_reject must be <= 0.0")
        if float(c_inspect) > 0.0:
            raise ValueError("c_inspect must be <= 0.0")
        if float(p_overinspect) > 0.0:
            raise ValueError("p_overinspect must be <= 0.0")
        if float(p_reinspect) > 0.0:
            raise ValueError("p_reinspect must be <= 0.0")
        if float(p_undecided) > 0.0:
            raise ValueError("p_undecided must be <= 0.0")

        self.day_len = int(day_len)
        self.time_budget = int(time_budget)
        self.fraud_rate_range = fraud_rate_range
        self.mid_day_update_prob = float(mid_day_update_prob)
        self.inspect_error_prob = float(inspect_error_prob)
        self.inspect_miss_prob = float(inspect_miss_prob)
        self.max_inspects_per_applicant = int(max_inspects_per_applicant)
        self.decision_coverage_target = float(decision_coverage_target)
        self.coverage_shortfall_penalty = float(coverage_shortfall_penalty)
        self.coverage_hard_threshold = float(coverage_hard_threshold)
        self.coverage_hard_penalty = float(coverage_hard_penalty)
        self.debug = bool(debug)
        self._rng = random.Random(seed)

        # rules vector: allowed countries + permit_required + id_card_required + work_pass_required
        obs_dim = len(COUNTRIES) + 3 + len(COUNTRIES) + 3 * len(FIELDS) + 3
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.rules: Optional[Rules] = None
        self.queue: List[Applicant] = []
        self.idx: int = 0
        self.time_left: int = 0
        self.revealed: Dict[str, int] = {}
        self.current_inspects: int = 0

        self.mid_day_update_fired: bool = False
        self.mid_day_update_idx: int = 0
        self.last_rule_update: Optional[str] = None

        self.r_correct = float(r_correct)
        self.p_false_accept = float(p_false_accept)
        self.p_false_reject = float(p_false_reject)
        self.c_inspect = float(c_inspect)
        self.p_overinspect = float(p_overinspect)
        self.p_reinspect = float(p_reinspect)
        self.p_undecided = float(p_undecided)

        self.stats = {
            "approves": 0,
            "denies": 0,
            "false_accept": 0,
            "false_reject": 0,
            "inspects": 0,
            "reinspect": 0,
            "inspect_noise_error": 0,
            "inspect_noise_miss": 0,
            "undecided": 0,
            "overinspect": 0,
            "coverage_shortfall": 0,
            "coverage_hard_violations": 0,
            "decision_coverage": 0.0,
        }

    def _reset_reveals(self) -> None:
        """Set all inspectable fields to unknown for current applicant."""
        self.revealed = {f: -1 for f in FIELDS}

    def _reset_stats(self) -> None:
        """Reset per-episode counters exported as `episode_stats`."""
        for k in self.stats:
            self.stats[k] = 0.0 if k == "decision_coverage" else 0

    def _get_obs(self) -> np.ndarray:
        """Build flat observation vector according to contract in AGENTS.md."""
        assert self.rules is not None
        assert 0 <= self.idx < len(self.queue)

        rules_vec = np.concatenate(
            [
                self.rules.allowed_countries_mask.astype(np.float32),
                np.array(
                    [
                        float(self.rules.permit_required),
                        float(self.rules.id_card_required_for_citizens),
                        float(self.rules.work_pass_required),
                    ],
                    dtype=np.float32,
                ),
            ]
        )

        app = self.queue[self.idx]
        country_oh = np.zeros((len(COUNTRIES),), dtype=np.float32)
        country_oh[app.country_idx] = 1.0

        reveal_parts = []
        for f in FIELDS:
            v = self.revealed[f]
            if v == -1:
                reveal_parts.extend([1.0, 0.0, 0.0])
            elif v == 1:
                reveal_parts.extend([0.0, 1.0, 0.0])
            else:
                reveal_parts.extend([0.0, 0.0, 1.0])
        reveal_vec = np.array(reveal_parts, dtype=np.float32)

        time_norm = np.array([self.time_left / max(1, self.time_budget)], dtype=np.float32)
        inspect_norm = np.array([self.current_inspects / max(1, self.max_inspects_per_applicant)], dtype=np.float32)
        remaining = (len(self.queue) - self.idx) / max(1, len(self.queue))
        remaining_norm = np.array([remaining], dtype=np.float32)

        obs = np.concatenate([rules_vec, country_oh, reveal_vec, time_norm, inspect_norm, remaining_norm]).astype(np.float32)
        return obs

    def _terminal_obs(self) -> np.ndarray:
        """Return valid-shaped terminal observation."""
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _maybe_apply_mid_day_rule_update(self) -> Optional[str]:
        """
        Optionally apply one deterministic rule change once per episode.

        Returns a short event label when update is applied, else None.
        """
        assert self.rules is not None
        if self.mid_day_update_fired:
            return None
        if self.idx < self.mid_day_update_idx:
            return None
        if self._rng.random() > self.mid_day_update_prob:
            self.mid_day_update_fired = True
            return None

        update_type = self._rng.choice(["permit", "id_card", "work_pass", "country_policy"])
        if update_type == "permit":
            self.rules.permit_required = 1 - int(self.rules.permit_required)
            event = f"rule_update:permit_required={self.rules.permit_required}"
        elif update_type == "id_card":
            self.rules.id_card_required_for_citizens = 1 - int(self.rules.id_card_required_for_citizens)
            event = f"rule_update:id_card_required_for_citizens={self.rules.id_card_required_for_citizens}"
        elif update_type == "work_pass":
            self.rules.work_pass_required = 1 - int(self.rules.work_pass_required)
            event = f"rule_update:work_pass_required={self.rules.work_pass_required}"
        else:
            country_idx = self._rng.randrange(len(COUNTRIES))
            self.rules.allowed_countries_mask[country_idx] = 1 - int(self.rules.allowed_countries_mask[country_idx])
            state = int(self.rules.allowed_countries_mask[country_idx])
            event = f"rule_update:country_{COUNTRIES[country_idx]}={state}"

        self.mid_day_update_fired = True
        self.last_rule_update = event
        return event

    def _apply_inspect_noise(self, true_value: int) -> int:
        """
        Return noisy inspect output:
        -1 => missing/unknown, 0/1 => observed boolean.
        """
        r = self._rng.random()
        if r < self.inspect_miss_prob:
            self.stats["inspect_noise_miss"] += 1
            return -1
        if r < (self.inspect_miss_prob + self.inspect_error_prob):
            self.stats["inspect_noise_error"] += 1
            return 1 - int(true_value)
        return int(true_value)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode (workday) with deterministic seed support."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        self.rules = sample_rules(self._rng)
        self.queue, fraud_rate = build_queue_with_deny_band(
            rng=self._rng,
            rules=self.rules,
            day_len=self.day_len,
            fraud_rate_range=self.fraud_rate_range,
            deny_ratio_range=(0.25, 0.55),
        )

        if self.debug:
            deny_needed = sum(not oracle_is_legal(self.rules, a) for a in self.queue)
            print("DENY NEEDED:", deny_needed)

        self.idx = 0
        self.time_left = self.time_budget
        self.current_inspects = 0
        self.mid_day_update_fired = False
        self.mid_day_update_idx = max(1, self.day_len // 2)
        self.last_rule_update = None
        self._reset_reveals()
        self._reset_stats()
        return self._get_obs(), {"fraud_rate": fraud_rate, "mid_day_update_idx": self.mid_day_update_idx}

    def step(self, action: int):
        """Execute decision/inspection action and return Gymnasium transition."""
        assert self.rules is not None

        terminated = False
        truncated = False
        reward = 0.0
        info: Dict[str, object] = {}

        def apply_undecided_penalty() -> None:
            nonlocal reward
            remaining = max(0, len(self.queue) - self.idx)
            if remaining > 0:
                self.stats["undecided"] += int(remaining)
                reward += self.p_undecided * float(remaining)

        def apply_coverage_penalty() -> None:
            nonlocal reward
            decisions = int(self.stats["approves"]) + int(self.stats["denies"])
            coverage = decisions / max(1, len(self.queue))
            self.stats["decision_coverage"] = float(coverage)

            target = int(math.ceil(len(self.queue) * self.decision_coverage_target))
            shortfall = max(0, target - decisions)
            if shortfall > 0:
                self.stats["coverage_shortfall"] += int(shortfall)
                reward += self.coverage_shortfall_penalty * float(shortfall)

            if coverage < self.coverage_hard_threshold:
                self.stats["coverage_hard_violations"] += 1
                reward += self.coverage_hard_penalty

        if self.time_left <= 0:
            truncated = True
            apply_undecided_penalty()
            apply_coverage_penalty()
            info["episode_stats"] = self.stats.copy()
            obs = self._terminal_obs()
            info.update({"time_left": self.time_left, "idx": self.idx})
            return obs, float(reward), terminated, truncated, info

        update_event = self._maybe_apply_mid_day_rule_update()
        if update_event is not None:
            info["rule_update"] = update_event
            if self.debug:
                print(update_event)

        app = self.queue[self.idx]
        inspect_field_map = {
            ACTION_INSPECT_COUNTRY_ALLOWED: "country_allowed",
            ACTION_INSPECT_HAS_PERMIT: "has_permit",
            ACTION_INSPECT_EXPIRY_VALID: "expiry_valid",
            ACTION_INSPECT_NAME_MATCH: "name_match",
            ACTION_INSPECT_HAS_ID_CARD: "has_id_card",
            ACTION_INSPECT_IS_WORKER: "is_worker",
            ACTION_INSPECT_HAS_WORK_PASS: "has_work_pass",
            ACTION_INSPECT_PURPOSE_MATCH: "purpose_match",
            ACTION_INSPECT_SEAL_VALID: "seal_valid",
            ACTION_INSPECT_BIOMETRIC_MATCH: "biometric_match",
        }
        inspect_actions = set(inspect_field_map.keys())

        def decision_from_reveals() -> int:
            citizen = int(app.country_idx == COUNTRIES.index("ARSTOTZKA"))
            deny_flags = [
                self.revealed["country_allowed"] == 0,
                self.revealed["expiry_valid"] == 0,
                self.revealed["purpose_match"] == 0,
                self.revealed["biometric_match"] == 0,
                self.revealed["has_permit"] == 0 and citizen == 0 and self.rules.permit_required == 1,
                self.revealed["has_id_card"] == 0 and citizen == 1 and self.rules.id_card_required_for_citizens == 1,
                self.revealed["is_worker"] == 1 and self.revealed["has_work_pass"] == 0 and self.rules.work_pass_required == 1,
                self.revealed["has_permit"] == 1 and self.revealed["name_match"] == 0,
                self.revealed["has_permit"] == 1 and self.revealed["seal_valid"] == 0,
            ]
            if any(deny_flags):
                return ACTION_DENY
            return ACTION_APPROVE

        # If near inspect cap and selected action is a redundant re-inspect, force a reveal-based decision.
        if action in inspect_actions:
            field = inspect_field_map[int(action)]
            near_cap = self.current_inspects >= max(1, self.max_inspects_per_applicant - 1)
            if self.revealed[field] != -1 and near_cap:
                self.stats["reinspect"] += 1
                self.time_left -= 1
                reward += self.p_reinspect
                action = decision_from_reveals()

        # If inspect cap is exceeded, force a DENY decision to prevent inspect-only deadlocks.
        if action in inspect_actions and self.current_inspects >= self.max_inspects_per_applicant:
            self.stats["overinspect"] += 1
            self.time_left -= 1
            reward += self.p_overinspect
            action = ACTION_DENY

        def reveal(field: str, value: int) -> None:
            self.revealed[field] = int(value)

        def inspect(field: str, true_value: int) -> None:
            nonlocal reward
            if self.revealed[field] != -1:
                self.stats["reinspect"] += 1
                reward += self.p_reinspect

            observed = self._apply_inspect_noise(int(true_value))
            reveal(field, observed)
            reward += self.c_inspect
            self.time_left -= 1
            self.stats["inspects"] += 1
            self.current_inspects += 1

        if action in (ACTION_APPROVE, ACTION_DENY):
            legal = oracle_is_legal(self.rules, app)
            decided_approve = action == ACTION_APPROVE

            if action == ACTION_APPROVE:
                self.stats["approves"] += 1
            else:
                self.stats["denies"] += 1

            if decided_approve and legal:
                reward += self.r_correct
            elif decided_approve and not legal:
                self.stats["false_accept"] += 1
                reward += self.p_false_accept
            elif (not decided_approve) and legal:
                self.stats["false_reject"] += 1
                reward += self.p_false_reject
            else:
                reward += self.r_correct

            self.idx += 1
            self.current_inspects = 0
            self._reset_reveals()
            if self.idx >= len(self.queue):
                terminated = True

        elif action == ACTION_INSPECT_COUNTRY_ALLOWED:
            inspect("country_allowed", int(self.rules.allowed_countries_mask[app.country_idx] == 1))

        elif action == ACTION_INSPECT_HAS_PERMIT:
            inspect("has_permit", int(app.has_permit))

        elif action == ACTION_INSPECT_EXPIRY_VALID:
            inspect("expiry_valid", int(app.expiry_valid))

        elif action == ACTION_INSPECT_NAME_MATCH:
            inspect("name_match", int(app.name_match))

        elif action == ACTION_INSPECT_HAS_ID_CARD:
            inspect("has_id_card", int(app.has_id_card))

        elif action == ACTION_INSPECT_IS_WORKER:
            inspect("is_worker", int(app.is_worker))

        elif action == ACTION_INSPECT_HAS_WORK_PASS:
            inspect("has_work_pass", int(app.has_work_pass))

        elif action == ACTION_INSPECT_PURPOSE_MATCH:
            inspect("purpose_match", int(app.purpose_match))

        elif action == ACTION_INSPECT_SEAL_VALID:
            inspect("seal_valid", int(app.seal_valid))

        elif action == ACTION_INSPECT_BIOMETRIC_MATCH:
            inspect("biometric_match", int(app.biometric_match))

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.idx >= len(self.queue):
            terminated = True

        if (not terminated) and self.time_left <= 0:
            truncated = True
            apply_undecided_penalty()

        if terminated or truncated:
            apply_coverage_penalty()
            info["episode_stats"] = self.stats.copy()

        obs = self._terminal_obs() if (terminated or truncated) else self._get_obs()
        info.update({"time_left": self.time_left, "idx": self.idx})
        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Minimal console renderer for debugging."""
        if self.rules is None or self.idx >= len(self.queue):
            print("[END OF DAY]")
            return

        app = self.queue[self.idx]
        print(f"Time left: {self.time_left} | Remaining: {len(self.queue) - self.idx}")
        print(
            f"Rules: allowed={[(COUNTRIES[i], int(self.rules.allowed_countries_mask[i])) for i in range(len(COUNTRIES))]}, "
            f"permit_required={self.rules.permit_required}, "
            f"id_card_required_for_citizens={self.rules.id_card_required_for_citizens}, "
            f"work_pass_required={self.rules.work_pass_required}"
        )
        print(f"Applicant country={COUNTRIES[app.country_idx]} | revealed={self.revealed}")


















