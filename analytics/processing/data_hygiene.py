"""
Shared data-hygiene helpers for the analytics pipeline.

Single place for:
  * placeholder handling ("UNKNOWN", "", "N/A", ... are *missing*, not categories)
  * region identity (a region is always country + subregion)
  * exact process-type normalisation
  * flavor-term extraction (falsy family/genus/species never count)
  * headline overview counts (one definition used by the cache and the pages)

Pure pandas/numpy: no streamlit or supabase imports, so it is cheap to test.
"""

import logging
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lower-cased, stripped spellings that mean "no value".
PLACEHOLDER_STRINGS = frozenset({
    '', 'unknown', 'n/a', 'na', 'n.a.', 'none', 'null', 'nan', 'nil',
    'unspecified', 'not specified', 'not available', 'tbd', '-', '--', '?',
    '[]', "['']", '[""]', "['unknown']", '["unknown"]',
})


def is_placeholder(value: Any) -> bool:
    """True for None/NaN/empty/"UNKNOWN"-style values."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, str):
        return value.replace('\xa0', ' ').strip().lower() in PLACEHOLDER_STRINGS
    return False


def clean_text(value: Any) -> Any:
    """Strip a text value; placeholders become NaN."""
    if is_placeholder(value):
        return np.nan
    if isinstance(value, str):
        return value.replace('\xa0', ' ').strip()
    return value


# -----------------------------------------------------------------------------
# Region identity
# -----------------------------------------------------------------------------

# Known source typos / same-place spelling variants, applied (case-insensitively,
# as whole phrases, so they also fix compound values like "Cerrado, Sao Paolo")
# before a region key is built. The dataset's subregions are plain ASCII with
# title-case ("Espirito Santo", "Sul De Minas"), hence "Sao Paulo", not "São".
# Only unambiguous merges belong here: (country, regex, canonical).
SUBREGION_ALIASES = [
    ('Brazil', r'\bsao paolo\b', 'Sao Paulo'),
    ('Brazil', r'\bsul de minas\b', 'Sul De Minas'),
    ('Papua New Guinea', r'\beastern highlands\b', 'Eastern Highlands'),
]
_COMPILED_ALIASES = [(c, re.compile(p, re.IGNORECASE), r) for c, p, r in SUBREGION_ALIASES]


def normalise_subregion(country: Any, subregion: Any) -> Any:
    """Placeholder -> NaN; otherwise strip and apply SUBREGION_ALIASES."""
    sub = clean_text(subregion)
    if pd.isna(sub):
        return np.nan
    country = clean_text(country)
    for alias_country, pattern, replacement in _COMPILED_ALIASES:
        if country == alias_country:
            sub = pattern.sub(replacement, sub)
    return sub


def make_region_key(country: Any, subregion: Any) -> Any:
    """Region identity = country + subregion. NaN if either part is missing.

    Same-named subregions in different countries stay distinct, and a missing
    subregion never becomes "<Country>_UNKNOWN". Known spelling variants are
    merged first (see SUBREGION_ALIASES).
    """
    country, subregion = clean_text(country), normalise_subregion(country, subregion)
    if pd.isna(country) or pd.isna(subregion):
        return np.nan
    return f"{country}_{subregion}"


def add_region_key(df: pd.DataFrame, country_col: str = 'country_final',
                   subregion_col: str = 'subregion_final') -> pd.Series:
    """Series of region keys aligned to df (NaN where not identifiable)."""
    return pd.Series(
        [make_region_key(c, s) for c, s in zip(df[country_col], df[subregion_col])],
        index=df.index, dtype=object,
    )


# -----------------------------------------------------------------------------
# Process type: exact canonical mapping (no substring matching)
# -----------------------------------------------------------------------------

# Raw values actually present upstream: Washed, Natural, Honey, Wet Hulled,
# Monsoon, Decaf (plus "Unknown"/null placeholders). A few unambiguous
# spelling synonyms are accepted; everything else is NOT guessed at.
PROCESS_MAP = {
    'washed': 'Washed',
    'fully washed': 'Washed',
    'fully-washed': 'Washed',
    'wet process': 'Washed',
    'natural': 'Natural',
    'dry process': 'Natural',
    'honey': 'Honey',
    'honey process': 'Honey',
    'pulped natural': 'Honey',
    'wet hulled': 'Wet Hulled',
    'wet-hulled': 'Wet Hulled',
    'giling basah': 'Wet Hulled',
    'anaerobic': 'Anaerobic',
    'monsoon': 'Other',
    'monsooned': 'Other',
    'decaf': 'Other',
    'decaffeinated': 'Other',
}

_warned_process_values = set()


def normalize_process(value: Any) -> Any:
    """Map a raw process string to a canonical category by exact lookup.

    Placeholders -> NaN. Unrecognised values -> NaN with a one-time warning
    (never silently coerced to another category).
    """
    if is_placeholder(value):
        return np.nan
    key = str(value).strip().lower()
    canonical = PROCESS_MAP.get(key)
    if canonical is None:
        if key not in _warned_process_values:
            _warned_process_values.add(key)
            logger.warning("Unrecognised process type %r treated as missing", value)
        return np.nan
    return canonical


# -----------------------------------------------------------------------------
# Flavors
# -----------------------------------------------------------------------------

FLAVOR_LEVELS = ('family', 'genus', 'species')


def clean_flavors(flavors: Any) -> List[Dict[str, str]]:
    """Drop falsy/placeholder family/genus/species values at their level.

    A flavor dict with no remaining level is dropped entirely.
    """
    if not isinstance(flavors, list):
        return []
    cleaned = []
    for item in flavors:
        if not isinstance(item, dict):
            continue
        new = {}
        for key, val in item.items():
            if key in FLAVOR_LEVELS:
                val = clean_text(val)
                if pd.isna(val):
                    continue
            new[key] = val
        if any(level in new for level in FLAVOR_LEVELS):
            cleaned.append(new)
    return cleaned


def flavor_terms(flavors: Iterable[Dict[str, Any]], level: str) -> List[str]:
    """All truthy terms at a taxonomy level (one per flavor instance)."""
    return [f[level] for f in flavors if isinstance(f, dict) and f.get(level)]


# -----------------------------------------------------------------------------
# Headline counts
# -----------------------------------------------------------------------------

def headline_counts(merged_df: pd.DataFrame) -> Dict[str, int]:
    """Overview counts, defined once.

    regions = distinct country+subregion keys (what the sidebar lists).
    Missing values never count as a unit.
    """
    if merged_df is None or merged_df.empty:
        return {'total_coffees': 0, 'countries_analyzed': 0,
                'regions_analyzed': 0, 'sellers_analyzed': 0,
                'unique_flavor_families': 0}

    if 'region_key' in merged_df.columns:
        region_keys = merged_df['region_key']
    else:
        region_keys = add_region_key(merged_df)

    families = set()
    if 'flavors_parsed' in merged_df.columns:
        for flavors in merged_df['flavors_parsed']:
            families.update(flavor_terms(flavors, 'family'))

    def _n(series):
        return int(series[~series.map(is_placeholder)].nunique())

    return {
        'total_coffees': int(len(merged_df)),
        'countries_analyzed': _n(merged_df['country_final']),
        'regions_analyzed': _n(region_keys),
        'sellers_analyzed': _n(merged_df['seller_name']) if 'seller_name' in merged_df.columns else 0,
        'unique_flavor_families': len(families),
    }
