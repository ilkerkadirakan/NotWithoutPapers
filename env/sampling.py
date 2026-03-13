from __future__ import annotations

"""Sampling utilities for daily rules and applicant queue generation."""

import random
from typing import List, Tuple

import numpy as np

from .constants import COUNTRIES
from .domain import Applicant, Rules, oracle_is_legal


def sample_rules(rng: random.Random) -> Rules:
    """Sample one day's policy rules."""
    allowed = np.zeros((len(COUNTRIES),), dtype=np.int32)
    # Allow between 3 and all countries.
    k = rng.randint(3, len(COUNTRIES))
    for i in rng.sample(range(len(COUNTRIES)), k=k):
        allowed[i] = 1

    return Rules(
        allowed_countries_mask=allowed,
        permit_required=rng.choice([0, 1]),
        id_card_required_for_citizens=rng.choice([0, 1]),
        work_pass_required=rng.choice([0, 1]),
    )


def sample_applicant(rng: random.Random, rules: Rules, fraud_rate: float) -> Applicant:
    """Sample one applicant and optionally inject a fraud pattern."""
    country_idx = rng.randrange(len(COUNTRIES))
    is_citizen = int(country_idx == COUNTRIES.index("ARSTOTZKA"))

    is_worker = rng.choice([0, 1])
    has_permit = 1 if (is_citizen == 0 and rules.permit_required == 1) else rng.choice([0, 1])
    has_id_card = 1 if (is_citizen == 1 and rules.id_card_required_for_citizens == 1) else rng.choice([0, 1])
    has_work_pass = 1 if (is_worker == 1 and rules.work_pass_required == 1) else rng.choice([0, 1])

    app = Applicant(
        country_idx=country_idx,
        has_permit=has_permit,
        name_match=1,
        expiry_valid=1,
        has_id_card=has_id_card,
        is_worker=is_worker,
        has_work_pass=has_work_pass,
        purpose_match=1,
        seal_valid=1,
    )

    if rng.random() < fraud_rate:
        fraud_type = rng.choice(
            [
                "bad_country",
                "missing_permit",
                "expired",
                "name_mismatch",
                "missing_id_card",
                "missing_work_pass",
                "purpose_mismatch",
                "fake_seal",
            ]
        )

        if fraud_type == "bad_country":
            disallowed = [i for i in range(len(COUNTRIES)) if rules.allowed_countries_mask[i] == 0]
            if disallowed:
                app.country_idx = rng.choice(disallowed)

        elif fraud_type == "missing_permit":
            app.has_permit = 0

        elif fraud_type == "expired":
            app.expiry_valid = 0

        elif fraud_type == "name_mismatch":
            app.has_permit = 1
            app.name_match = 0

        elif fraud_type == "missing_id_card":
            app.country_idx = COUNTRIES.index("ARSTOTZKA")
            app.has_id_card = 0

        elif fraud_type == "missing_work_pass":
            app.is_worker = 1
            app.has_work_pass = 0

        elif fraud_type == "purpose_mismatch":
            app.purpose_match = 0

        elif fraud_type == "fake_seal":
            app.has_permit = 1
            app.seal_valid = 0

    return app


def build_queue_with_deny_band(
    rng: random.Random,
    rules: Rules,
    day_len: int,
    fraud_rate_range: Tuple[float, float],
    deny_ratio_range: Tuple[float, float] = (0.20, 0.40),
) -> Tuple[List[Applicant], float]:
    """Build day queue and enforce deny ratio band for stable training signal."""
    fraud_rate = rng.uniform(*fraud_rate_range)
    queue = [sample_applicant(rng=rng, rules=rules, fraud_rate=fraud_rate) for _ in range(day_len)]

    deny_needed = sum(not oracle_is_legal(rules, a) for a in queue)
    target_min = int(day_len * deny_ratio_range[0])
    target_max = int(day_len * deny_ratio_range[1])

    while deny_needed < target_min:
        i = rng.randrange(len(queue))
        if oracle_is_legal(rules, queue[i]):
            queue[i].purpose_match = 0
            deny_needed += 1

    while deny_needed > target_max:
        i = rng.randrange(len(queue))
        if not oracle_is_legal(rules, queue[i]):
            a = queue[i]
            a.expiry_valid = 1
            a.name_match = 1
            a.seal_valid = 1
            a.purpose_match = 1
            a.has_permit = 1
            a.has_id_card = 1
            a.has_work_pass = 1
            a.is_worker = 0
            if rules.allowed_countries_mask[a.country_idx] == 0:
                allowed = [j for j in range(len(COUNTRIES)) if rules.allowed_countries_mask[j] == 1]
                if allowed:
                    a.country_idx = rng.choice(allowed)
            if oracle_is_legal(rules, a):
                deny_needed -= 1

    return queue, fraud_rate
