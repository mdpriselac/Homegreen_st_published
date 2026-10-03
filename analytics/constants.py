"""Shared constants and display helpers for the analytics package."""


UNIT_TYPE_PLURALS = {'country': 'countries', 'region': 'regions', 'seller': 'sellers'}


def unit_plural(unit_type: str) -> str:
    """Lower-case plural of a unit type (country -> countries)."""
    return UNIT_TYPE_PLURALS.get(unit_type, f"{unit_type}s")


def unit_label(unit_type: str, unit: str) -> str:
    """Display label for a unit. Region keys ("Country_Subregion") become
    "Country / Subregion"; countries and sellers are returned unchanged."""
    if unit_type == 'region' and isinstance(unit, str) and '_' in unit:
        country, subregion = unit.split('_', 1)
        return f"{country} / {subregion}"
    return unit
