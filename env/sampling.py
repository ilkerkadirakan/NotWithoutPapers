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
    k = rng.randint(2, len(COUNTRIES))
    for i in rng.sample(range(len(COUNTRIES)), k=k):
        allowed[i] = 1
    permit_required = rng.choice([0, 1])
    return Rules(allowed_countries_mask=allowed, permit_required=permit_required)


def sample_applicant(rng: random.Random, rules: Rules, fraud_rate: float) -> Applicant:
    """Sample one applicant and optionally inject a fraud pattern."""
    country_idx = rng.randrange(len(COUNTRIES))
    has_permit = rng.choice([0, 1])
    app = Applicant(
        country_idx=country_idx,
        has_permit=has_permit,
        name_match=1,
        expiry_valid=1,
    )

    if rng.random() < fraud_rate:
        fraud_type = rng.choice(["bad_country", "missing_permit", "expired", "name_mismatch"])
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
            queue[i].expiry_valid = 0
            deny_needed += 1

    while deny_needed > target_max:
        i = rng.randrange(len(queue))
        if not oracle_is_legal(rules, queue[i]):
            a = queue[i]
            a.expiry_valid = 1
            a.name_match = 1
            a.has_permit = 1
            if rules.allowed_countries_mask[a.country_idx] == 0:
                allowed = [j for j in range(len(COUNTRIES)) if rules.allowed_countries_mask[j] == 1]
                if allowed:
                    a.country_idx = rng.choice(allowed)
            if oracle_is_legal(rules, a):
                deny_needed -= 1

    return queue, fraud_rate
