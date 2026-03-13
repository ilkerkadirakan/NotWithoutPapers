from __future__ import annotations

"""Shared constants for the environment contract.

Important: existing action IDs must stay stable for backward compatibility.
"""

# Core domain constants (expanded toward Papers Please flavor)
COUNTRIES = [
    "ARSTOTZKA",
    "ANTEGRIA",
    "IMPOR",
    "KOLECHIA",
    "OBRISTAN",
    "REPUBLIA",
    "UNITEDFED",
]
DOCS = ["PASSPORT", "PERMIT"]
FIELDS = [
    "name_match",
    "expiry_valid",
    "country_allowed",
    "has_permit",
    "has_id_card",
    "is_worker",
    "has_work_pass",
    "purpose_match",
    "seal_valid",
    "biometric_match",
]

# Action contract (do not reorder existing IDs)
ACTION_APPROVE = 0
ACTION_DENY = 1
ACTION_INSPECT_COUNTRY_ALLOWED = 2
ACTION_INSPECT_HAS_PERMIT = 3
ACTION_INSPECT_EXPIRY_VALID = 4
ACTION_INSPECT_NAME_MATCH = 5
ACTION_INSPECT_HAS_ID_CARD = 6
ACTION_INSPECT_IS_WORKER = 7
ACTION_INSPECT_HAS_WORK_PASS = 8
ACTION_INSPECT_PURPOSE_MATCH = 9
ACTION_INSPECT_SEAL_VALID = 10
ACTION_INSPECT_BIOMETRIC_MATCH = 11

N_ACTIONS = 12
