"""
Shared varietal handling.

normalise_varietal(): canonicalise spelling (case, accents, hyphens, spacing,
known aliases) without merging genuinely different cultivars: "Yellow Catuai"
stays distinct from "Catuai", "Pink Bourbon" from "Bourbon".

expand_varietals(df, mode): the single replacement for the per-module
`_expand_varietals` copies.
  * mode="weighted": one row per (coffee, varietal), weight = 1/k where k is the
    coffee's number of distinct varietals. Use for descriptive counts/shares so
    each coffee counts once in total.
  * mode="single": only coffees with exactly one varietal (weight 1). Use for
    every statistical test (chi-square, Kruskal-Wallis, Mann-Whitney, binomial)
    so observations are independent coffees, not duplicated rows.
"""

import re
import unicodedata
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from analytics.processing.data_hygiene import is_placeholder

# Words that carry no cultivar information.
_GENERIC_WORDS = frozenset({
    'arabica', 'cultivar', 'cultivars', 'variety', 'varieties', 'varietal',
    'varietals', 'type', 'types', 'strain', 'strains', 'local', 'regional',
    'indigenous', 'selective', 'common', 'various', 'other', 'ethiopian',
    'ethiopia', 'yemeni', 'brazilian', 'including', 'and', 'or',
    'of', 'almost', 'entirely', 'primarily', 'majority', 'mixed', 'natural',
    'stock', 'genetics', 'exotic', 'cultivated', 'varies',
    'multiple', 
})
_HEIRLOOM_WORDS = frozenset({
    'heirloom', 'heirlooms', 'landrace', 'landraces', 'wild',
})
# Whole keys (after stripping) that are not a varietal at all.
_NON_VARIETAL = frozenset({'', 'unknown', 'na', 'none', 'null', 'nan', 'blend',
                           'peaberry', 'sumatra', 'arabica', 'sl'})

# Word-level spelling fixes (applied after case/accent folding).
_WORD_FIXES = {
    'gesha': 'geisha', 'geshae': 'geisha', 'geisha': 'geisha',
    'catuay': 'catuai', 'catui': 'catuai', 'catua': 'catuai',
    'catimore': 'catimor', 'tipica': 'typica', 'typical': 'typica',
    'maragogype': 'maragogipe', 'marogogype': 'maragogipe',
    'borbon': 'bourbon', 'bourbonn': 'bourbon',
    'mondo': 'mundo', 'mundu': 'mundo', 'nuovo': 'novo', 'nova': 'novo',
    'riuru': 'ruiru', 'ruiri': 'ruiru',
    'caturro': 'caturra', 'catura': 'caturra',
    'marsallesa': 'marsellesa', 'marcellesa': 'marsellesa', 'marseilles': 'marsellesa',
    'sydra': 'sidra', 'djember': 'jember',
    'kerume': 'kurume', 'kurame': 'kurume', 'kuruma': 'kurume', 'korume': 'kurume',
    'mtn': 'mountain', 'timtim': 'tim tim',
    'villasarchi': 'villa sarchi',
    # Spanish/Portuguese colour words
    'rojo': 'red', 'roja': 'red', 'vermelho': 'red', 'vermelha': 'red',
    'amarillo': 'yellow', 'amarilla': 'yellow', 'amarelo': 'yellow', 'amarela': 'yellow',
    'rosado': 'pink',
}
_COLOUR_WORDS = frozenset({'red', 'yellow', 'pink', 'orange', 'purple'})

# Whole-key aliases (key = lower-case alphanumeric words joined by single spaces).
_ALIASES = {
    'colombia': 'Variedad Colombia',
    'variedad colombia': 'Variedad Colombia',
    'ihcafe 90': 'IHCAFE 90', 'ih 90': 'IHCAFE 90', 'ih90': 'IHCAFE 90',
    'ihcafe90': 'IHCAFE 90', 'ihcafe': 'IHCAFE 90',
    'lini s': 'S795', 'linie s': 'S795', 'line s': 'S795', 'slini': 'S795',
    's lini': 'S795', 'lini': 'S795',
    'tim tim': 'Tim Tim',
    'jember': 'S795',   # Indonesian name for the S795 selection
    'ruiru': 'Ruiru 11', 'ruiru 11': 'Ruiru 11', 'ruiru11': 'Ruiru 11',
    'jadee jaadi': 'Jaadi', 'jadi': 'Jaadi', 'jaadi': 'Jaadi',
    'red yellow catuai': 'Red & Yellow Catuai',
    'red and yellow catuai': 'Red & Yellow Catuai',
    'red yellow caturra': 'Red & Yellow Caturra',
    'red and yellow caturra': 'Red & Yellow Caturra',
    'blue mountain': 'Blue Mountain',
    'typica mejorado': 'Typica Mejorado',
    'hibrido de timor': 'Timor Hybrid', 'timor hybrid': 'Timor Hybrid',
    'timor hybrids': 'Timor Hybrid', 'timor': 'Timor Hybrid',
}

_PERCENT_RE = re.compile(r'(?:less than\s*)?\d+(?:\.\d+)?\s*%')
_PAREN_RE = re.compile(r'\([^()]*\)')
_JARC_RE = re.compile(r'\b(74\d{3})\b')
_SL_RE = re.compile(r'^(?:(?:kenya|kenyan|bourbon)\s+)?sl\s*(\d+)$')
_S795_RE = re.compile(r'^(?:(?:s|lini|linie|line|kenya)\s*)*795$')


def _fold(text: str) -> str:
    """Lower-case, strip accents, keep only alphanumeric words."""
    text = unicodedata.normalize('NFKD', text.replace('\xa0', ' '))
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


_ACRONYMS = frozenset({'usda', 'jarc', 'sl', 'ihcafe', 'tacri'})


def _display(key: str) -> str:
    """Title-case a folded key; words containing digits are upper-cased."""
    return ' '.join(w.upper() if (w in _ACRONYMS or any(c.isdigit() for c in w)) else w.capitalize()
                    for w in key.split())


def normalise_varietal(value: Any) -> Optional[str]:
    """Canonical varietal name, or None when the value is not a varietal.

    Merges (examples): SL-28/SL 28/sl28 -> SL28; Catuai/Catuaí/Catuay -> Catuai;
    Gesha/Geisha -> Geisha; Ruiru-11/Ruiru11 -> Ruiru 11; 74110/JARC 74110/
    "Regional cultivars 74110" -> JARC 74110; generic "Heirloom ..." phrases ->
    Heirloom; "Catuaí Vermelho"/"Catuai Rojo" -> Red Catuai. Colour/lineage
    variants (Yellow Catuai, Pink Bourbon) are intentionally kept distinct.
    """
    if is_placeholder(value):
        return None
    text = str(value).strip()

    # strip percentages, balanced parentheses, then any unbalanced tail/stray ')'
    text = _PERCENT_RE.sub(' ', text)
    prev = None
    while prev != text:
        prev = text
        text = _PAREN_RE.sub(' ', text)
    text = text.split('(')[0].replace(')', ' ')

    key = _fold(text)
    if not key:
        return None

    # robusta / canephora are one species
    if 'robusta' in key.split() or 'canephora' in key.split():
        return 'Robusta'

    jarc = _JARC_RE.search(key)
    if jarc:
        return f'JARC {jarc.group(1)}'

    words = [_WORD_FIXES.get(w, w) for w in key.split()]
    words = ' '.join(words).split()

    has_heirloom = any(w in _HEIRLOOM_WORDS for w in words)
    kept = [w for w in words if w not in _GENERIC_WORDS and w not in _HEIRLOOM_WORDS]
    if not kept:
        return 'Heirloom' if has_heirloom else None
    key = ' '.join(kept)

    if key in _NON_VARIETAL:
        return None

    m = _SL_RE.match(key)
    if m:
        return f'SL{m.group(1)}'
    if _S795_RE.match(key):
        return 'S795'
    if key in _ALIASES:
        return _ALIASES[key]

    # move a trailing colour word to the front: "catuai red" -> "red catuai"
    ws = key.split()
    if len(ws) > 1 and ws[-1] in _COLOUR_WORDS:
        key = ' '.join([ws[-1]] + ws[:-1])
        if key in _ALIASES:
            return _ALIASES[key]

    return _display(key)


def _split_top_level_commas(text: str) -> List[str]:
    """Split on commas that are not inside parentheses; also split "(40%)Next"."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(depth - 1, 0)
        if ch == ',' and depth == 0:
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append(''.join(cur))
    out = []
    for p in parts:
        out.extend(re.split(r'(?<=\))\s*(?=[A-Za-z])', p))
    return out


def split_varietal_string(text: str) -> List[str]:
    """Split a free-text varietal string into raw tokens (paren-aware)."""
    return [t.strip() for t in _split_top_level_commas(text) if t.strip()]


def clean_varietal_list(values: Any) -> List[str]:
    """Normalise each entry, drop non-varietals, de-duplicate, keep order."""
    if values is None or (isinstance(values, float) and np.isnan(values)):
        return []
    if isinstance(values, str):
        values = [values]
    out = []
    for v in values:
        n = normalise_varietal(v)
        if n and n not in out:
            out.append(n)
    return out


def expand_varietals(df: pd.DataFrame, mode: str = 'weighted',
                     varietals_col: str = 'varietals') -> pd.DataFrame:
    """Expand a per-coffee frame into one row per varietal.

    Adds columns: single_varietal, n_varietals (distinct varietals on the
    coffee) and weight.
      mode="weighted": every (coffee, varietal) row, weight = 1/n_varietals,
        so weights sum to 1 per coffee. For descriptive counts/shares ONLY.
      mode="single": only coffees with exactly one varietal, weight = 1.
        Required for any significance test.
    Coffees without a varietal are dropped in both modes.
    """
    if mode not in ('weighted', 'single'):
        raise ValueError("mode must be 'weighted' or 'single'")

    extra = ['single_varietal', 'n_varietals', 'weight']
    if df is None or df.empty or varietals_col not in df.columns:
        base = df.iloc[0:0] if df is not None else pd.DataFrame()
        return base.assign(**{c: pd.Series(dtype=object) for c in extra})

    work = df.copy()
    work['_vlist'] = work[varietals_col].apply(clean_varietal_list)
    work['n_varietals'] = work['_vlist'].apply(len)
    work = work[work['n_varietals'] > 0]
    if mode == 'single':
        work = work[work['n_varietals'] == 1]
    work = work.explode('_vlist').rename(columns={'_vlist': 'single_varietal'})
    work['weight'] = 1.0 / work['n_varietals']
    return work.reset_index(drop=True)
