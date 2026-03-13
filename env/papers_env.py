from __future__ import annotations

"""Gymnasium environment implementation for the Papers Please RL task."""

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

        self.day_len = int(day_len)
        self.time_budget = int(time_budget)
        self.fraud_rate_range = fraud_rate_range
        self.mid_day_update_prob = float(mid_day_update_prob)
        self.inspect_error_prob = float(inspect_error_prob)
        self.inspect_miss_prob = float(inspect_miss_prob)
        self.debug = bool(debug)
        self._rng = random.Random(seed)

        # rules vector: allowed countries + permit_required + id_card_required + work_pass_required
        obs_dim = len(COUNTRIES) + 3 + len(COUNTRIES) + 3 * len(FIELDS) + 2
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.rules: Optional[Rules] = None
        self.queue: List[Applicant] = []
        self.idx: int = 0
        self.time_left: int = 0
        self.revealed: Dict[str, int] = {}

        self.mid_day_update_fired: bool = False
        self.mid_day_update_idx: int = 0
        self.last_rule_update: Optional[str] = None

        self.r_correct = 4.0
        self.p_false_accept = -15.0
        self.p_false_reject = -8.0
        self.c_inspect = -0.1
        # Penalize unresolved applicants at time-out to prevent inspect-only local optimum.
        self.p_undecided = 0.0

        self.stats = {
            "approves": 0,
            "denies": 0,
            "false_accept": 0,
            "false_reject": 0,
            "inspects": 0,
            "inspect_noise_error": 0,
            "inspect_noise_miss": 0,
            "undecided": 0,
        }

    def _reset_reveals(self) -> None:
        """Set all inspectable fields to unknown for current applicant."""
        self.revealed = {f: -1 for f in FIELDS}

    def _reset_stats(self) -> None:
        """Reset per-episode counters exported as `episode_stats`."""
        for k in self.stats:
            self.stats[k] = 0

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
        remaining = (len(self.queue) - self.idx) / max(1, len(self.queue))
        remaining_norm = np.array([remaining], dtype=np.float32)

        obs = np.concatenate([rules_vec, country_oh, reveal_vec, time_norm, remaining_norm]).astype(np.float32)
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

        if self.time_left <= 0:
            truncated = True
            apply_undecided_penalty()
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

        def reveal(field: str, value: int) -> None:
            self.revealed[field] = int(value)

        def inspect(field: str, true_value: int) -> None:
            nonlocal reward
            observed = self._apply_inspect_noise(int(true_value))
            reveal(field, observed)
            reward += self.c_inspect
            self.time_left -= 1
            self.stats["inspects"] += 1

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

        if (not terminated) and self.time_left <= 0:
            truncated = True
            apply_undecided_penalty()

        if terminated or truncated:
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

