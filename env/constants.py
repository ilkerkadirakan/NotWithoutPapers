from __future__ import annotations

"""Shared constants for the environment contract.

Important: existing action IDs must stay stable for backward compatibility.
"""

# Core domain constants
COUNTRIES = ["ARSTOTZKA", "KOLECHIA", "OBRISTAN", "ANTEGRIA"]
DOCS = ["PASSPORT", "PERMIT"]
FIELDS = ["name_match", "expiry_valid", "country_allowed", "has_permit"]

# Action contract (do not reorder existing IDs)
ACTION_APPROVE = 0
ACTION_DENY = 1
ACTION_INSPECT_COUNTRY_ALLOWED = 2
ACTION_INSPECT_HAS_PERMIT = 3
ACTION_INSPECT_EXPIRY_VALID = 4
ACTION_INSPECT_NAME_MATCH = 5

N_ACTIONS = 6
