from __future__ import annotations

"""Sampling utilities for daily rules and applicant queue generation."""

import random
from typing import Dict, List, Tuple

import numpy as np

from .constants import COUNTRIES
from .domain import Applicant, Rules, oracle_is_legal


def sample_rules(rng: random.Random) -> Rules:
    """Sample one day's policy rules."""
    allowed = np.zeros((len(COUNTRIES),), dtype=np.int32)
    # Keep broad country variety but avoid near-deterministic country-only legality.
    k = rng.randint(max(4, len(COUNTRIES) - 3), len(COUNTRIES))
    for i in rng.sample(range(len(COUNTRIES)), k=k):
        allowed[i] = 1

    return Rules(
        allowed_countries_mask=allowed,
        permit_required=rng.choice([0, 1]),
        id_card_required_for_citizens=rng.choice([0, 1]),
        work_pass_required=rng.choice([0, 1]),
    )


def _is_citizen(country_idx: int) -> int:
    return int(country_idx == COUNTRIES.index("ARSTOTZKA"))


def _legalize_applicant(app: Applicant, rules: Rules, rng: random.Random) -> None:
    """Mutate applicant to a legal state while preserving country identity."""
    citizen = _is_citizen(app.country_idx)

    # Keep occupation latent variable random, then satisfy rule-required docs.
    app.is_worker = rng.choice([0, 1])

    if citizen == 0 and rules.permit_required == 1:
        app.has_permit = 1
    else:
        app.has_permit = rng.choice([0, 1])

    if citizen == 1 and rules.id_card_required_for_citizens == 1:
        app.has_id_card = 1
    else:
        app.has_id_card = rng.choice([0, 1])

    if app.is_worker == 1 and rules.work_pass_required == 1:
        app.has_work_pass = 1
    else:
        app.has_work_pass = rng.choice([0, 1])

    app.expiry_valid = 1
    app.purpose_match = 1
    app.biometric_match = 1
    app.name_match = 1
    app.seal_valid = 1


def _illegalize_applicant(app: Applicant, rules: Rules, rng: random.Random) -> None:
    """Mutate applicant to an illegal state through hidden/inspectable fields."""
    _legalize_applicant(app, rules, rng)

    citizen = _is_citizen(app.country_idx)
    violations: List[str] = ["expiry", "purpose", "biometric"]

    if app.has_permit == 1:
        violations.extend(["name", "seal"])
    if citizen == 0 and rules.permit_required == 1:
        violations.append("missing_permit")
    if citizen == 1 and rules.id_card_required_for_citizens == 1:
        violations.append("missing_id")
    if app.is_worker == 1 and rules.work_pass_required == 1:
        violations.append("missing_work_pass")

    for _ in range(8):
        v = rng.choice(violations)
        if v == "expiry":
            app.expiry_valid = 0
        elif v == "purpose":
            app.purpose_match = 0
        elif v == "biometric":
            app.biometric_match = 0
        elif v == "name":
            app.name_match = 0
        elif v == "seal":
            app.seal_valid = 0
        elif v == "missing_permit":
            app.has_permit = 0
        elif v == "missing_id":
            app.has_id_card = 0
        elif v == "missing_work_pass":
            app.has_work_pass = 0

        if not oracle_is_legal(rules, app):
            return

    # Fallback strong fraud marker.
    app.biometric_match = 0


def _key_for_balance(rules: Rules, app: Applicant) -> Tuple[int, ...]:
    """Rule+country key used to measure no-inspect predictability."""
    return (
        int(app.country_idx),
        int(rules.permit_required),
        int(rules.id_card_required_for_citizens),
        int(rules.work_pass_required),
    )


def sample_applicant(rng: random.Random, rules: Rules, fraud_rate: float) -> Applicant:
    """Sample one applicant with legality mostly controlled by hidden fields."""
    allowed = [i for i in range(len(COUNTRIES)) if rules.allowed_countries_mask[i] == 1]
    if allowed:
        country_idx = rng.choice(allowed)
    else:
        country_idx = rng.randrange(len(COUNTRIES))

    app = Applicant(
        country_idx=country_idx,
        has_permit=1,
        name_match=1,
        expiry_valid=1,
        has_id_card=1,
        is_worker=rng.choice([0, 1]),
        has_work_pass=1,
        purpose_match=1,
        seal_valid=1,
        biometric_match=1,
    )

    if rng.random() < fraud_rate:
        _illegalize_applicant(app, rules, rng)
    else:
        _legalize_applicant(app, rules, rng)

    return app


def _rebalance_queue_by_key(
    rng: random.Random,
    rules: Rules,
    queue: List[Applicant],
    target_ratio_range: Tuple[float, float] = (0.40, 0.60),
) -> None:
    """Balance legal ratio within each (rule, country) bucket."""
    by_key: Dict[Tuple[int, ...], List[int]] = {}
    for i, app in enumerate(queue):
        by_key.setdefault(_key_for_balance(rules, app), []).append(i)

    lo, hi = target_ratio_range
    for indices in by_key.values():
        if len(indices) < 4:
            continue

        legal_count = sum(1 for i in indices if oracle_is_legal(rules, queue[i]))
        ratio = legal_count / len(indices)

        # Bring ratio into target band by flipping hidden legality in-place.
        guard = 0
        while ratio < lo and guard < 200:
            illegal_indices = [i for i in indices if not oracle_is_legal(rules, queue[i])]
            if not illegal_indices:
                break
            idx = rng.choice(illegal_indices)
            _legalize_applicant(queue[idx], rules, rng)
            legal_count = sum(1 for i in indices if oracle_is_legal(rules, queue[i]))
            ratio = legal_count / len(indices)
            guard += 1

        while ratio > hi and guard < 400:
            legal_indices = [i for i in indices if oracle_is_legal(rules, queue[i])]
            if not legal_indices:
                break
            idx = rng.choice(legal_indices)
            _illegalize_applicant(queue[idx], rules, rng)
            legal_count = sum(1 for i in indices if oracle_is_legal(rules, queue[i]))
            ratio = legal_count / len(indices)
            guard += 1


def build_queue_with_deny_band(
    rng: random.Random,
    rules: Rules,
    day_len: int,
    fraud_rate_range: Tuple[float, float],
    deny_ratio_range: Tuple[float, float] = (0.45, 0.55),
) -> Tuple[List[Applicant], float]:
    """Build day queue with global and per-key legality balance for harder inference."""
    fraud_rate = rng.uniform(*fraud_rate_range)
    queue = [sample_applicant(rng=rng, rules=rules, fraud_rate=fraud_rate) for _ in range(day_len)]

    # 1) First enforce per-key (rule+country) balance to reduce no-inspect predictability.
    _rebalance_queue_by_key(rng=rng, rules=rules, queue=queue, target_ratio_range=(0.40, 0.60))

    # 2) Then keep global deny ratio in a stable band for training signal.
    deny_needed = sum(not oracle_is_legal(rules, a) for a in queue)
    target_min = int(day_len * deny_ratio_range[0])
    target_max = int(day_len * deny_ratio_range[1])

    guard = 0
    while deny_needed < target_min and guard < 1000:
        i = rng.randrange(len(queue))
        if oracle_is_legal(rules, queue[i]):
            _illegalize_applicant(queue[i], rules, rng)
            if not oracle_is_legal(rules, queue[i]):
                deny_needed += 1
        guard += 1

    guard = 0
    while deny_needed > target_max and guard < 1000:
        i = rng.randrange(len(queue))
        if not oracle_is_legal(rules, queue[i]):
            _legalize_applicant(queue[i], rules, rng)
            if oracle_is_legal(rules, queue[i]):
                deny_needed -= 1
        guard += 1

    return queue, fraud_rate
