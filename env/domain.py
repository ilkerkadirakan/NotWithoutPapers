from __future__ import annotations

"""Domain data structures and oracle legality rule for applicants."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Rules:
    allowed_countries_mask: np.ndarray
    permit_required: int


@dataclass
class Applicant:
    country_idx: int
    has_permit: int
    name_match: int
    expiry_valid: int


def oracle_is_legal(rules: Rules, app: Applicant) -> bool:
    """Ground-truth legal/illegal decision used for reward and statistics."""
    if rules.allowed_countries_mask[app.country_idx] == 0:
        return False
    if rules.permit_required == 1 and app.has_permit == 0:
        return False
    if app.expiry_valid == 0:
        return False
    if app.has_permit == 1 and app.name_match == 0:
        return False
    return True
