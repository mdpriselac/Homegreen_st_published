"""Helpers for reading boolean flags from the frontend cache."""

from typing import Any


def as_bool(value: Any) -> bool:
    """Interpret a cached flag as a bool.

    Accepts real bools and the legacy "True"/"False" strings that older cache
    files contain (json.dump(default=str) stringified numpy bools). None, NaN
    and unrecognised values are False.
    """
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1')
    if value is None:
        return False
    try:
        return bool(value and value == value)  # NaN != NaN -> False
    except (TypeError, ValueError):
        return False


def significant_mask(series):
    """Boolean Series version of as_bool, for DataFrame filtering."""
    return series.map(as_bool).astype(bool)
