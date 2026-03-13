from __future__ import annotations

"""Domain data structures and oracle legality rule for applicants."""

from dataclasses import dataclass

import numpy as np

from .constants import COUNTRIES


@dataclass
class Rules:
    allowed_countries_mask: np.ndarray
    permit_required: int
    id_card_required_for_citizens: int
    work_pass_required: int


@dataclass
class Applicant:
    country_idx: int
    has_permit: int
    name_match: int
    expiry_valid: int
    has_id_card: int
    is_worker: int
    has_work_pass: int
    purpose_match: int
    seal_valid: int
    biometric_match: int


def oracle_is_legal(rules: Rules, app: Applicant) -> bool:
    """Ground-truth legal/illegal decision used for reward and statistics."""
    if rules.allowed_countries_mask[app.country_idx] == 0:
        return False

    # Permit is required only for foreigners (Arstotzkans are citizens).
    is_citizen = int(app.country_idx == COUNTRIES.index("ARSTOTZKA"))
    if is_citizen == 0 and rules.permit_required == 1 and app.has_permit == 0:
        return False

    # Citizen ID card requirement.
    if is_citizen == 1 and rules.id_card_required_for_citizens == 1 and app.has_id_card == 0:
        return False

    # Worker-specific requirement.
    if app.is_worker == 1 and rules.work_pass_required == 1 and app.has_work_pass == 0:
        return False

    if app.expiry_valid == 0:
        return False

    if app.has_permit == 1 and app.name_match == 0:
        return False

    # Permit seal must be valid if permit is present.
    if app.has_permit == 1 and app.seal_valid == 0:
        return False

    # Declared purpose must be coherent.
    if app.purpose_match == 0:
        return False

    # Biometric/photo mismatch is a strong stolen-identity signal.
    if app.biometric_match == 0:
        return False

    return True
