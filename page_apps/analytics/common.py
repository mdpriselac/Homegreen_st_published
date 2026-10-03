"""Shared helpers for the distinctiveness pages (Explore, By Flavor, Compare, Rankings)."""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from analytics.constants import unit_label, unit_plural
from analytics.frontend.flags import as_bool
from analytics.processing.distinctiveness import (
    MAX_SELLER_SHARE, MIN_SELLERS, format_distinctive_sentence)

LEVELS = ['family', 'genus', 'species']
DEFAULT_LEVEL_INDEX = 1          # genus
UNIT_TYPES = ['country', 'region', 'seller']
UNIT_TYPE_PLURAL_LABELS = {'country': 'Countries', 'region': 'Regions', 'seller': 'Sellers'}
UNIT_TYPE_SINGULAR_LABELS = {'country': 'Country', 'region': 'Region', 'seller': 'Seller'}

LEVELS_NOTE = (
    "Family, genus and species are tested separately: a flavor that stands out at one level "
    "does not mean its parent or child flavors do. Genus is the most informative starting point."
)
SELLER_SUPPORT_NOTE = (
    f"For countries and regions, a flavor is only called distinctive if it is found in coffees "
    f"from at least {MIN_SELLERS} sellers (and no one seller supplies more than "
    f"{MAX_SELLER_SHARE * 100:.0f}% of them), so it isn't just one seller's tasting vocabulary."
)
REGENERATE_MSG = ("The flavor distinctiveness data is not in this version of the analytics cache yet. "
                  "It will appear after the cache is regenerated.")


def level_selector(key: str, label: str = "Flavor level:") -> str:
    """Radio for one taxonomy level at a time (default genus). Returns 'family'|'genus'|'species'."""
    choice = st.radio(label, [lv.title() for lv in LEVELS], index=DEFAULT_LEVEL_INDEX,
                      horizontal=True, key=key)
    st.caption(LEVELS_NOTE)
    return choice.lower()


def fmt_n(x: Any) -> str:
    """Count that may be fractional (weighted): integers without decimals."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{int(round(xf))}" if abs(xf - round(xf)) < 1e-9 else f"{xf:.1f}"


def fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def sentence_for(row: Dict[str, Any], unit_type: str, unit: Optional[str] = None) -> str:
    """Plain-language sentence for a distinctive row, with a display label for the unit.
    Profile rows carry no unit name, so pass it via ``unit``."""
    r = dict(row)
    r['unit'] = unit_label(unit_type, unit if unit is not None else r.get('unit', ''))
    return format_distinctive_sentence(r, unit_type)


def support_text(row: Dict[str, Any]) -> Optional[str]:
    """'found in coffees from N sellers (top seller X% of them)' for origin rows."""
    n = row.get('n_sellers_supporting')
    if n is None:
        return None
    share = row.get('top_seller_share')
    top = row.get('top_seller')
    txt = f"found in coffees from {int(n)} seller{'s' if int(n) != 1 else ''}"
    if share is not None:
        who = f"{top} " if top else "top seller "
        txt += f" ({who}{share * 100:.0f}% of them)"
    return txt


def evidence_frame(rows: List[Dict[str, Any]], unit_type: str, by_unit: bool = False) -> pd.DataFrame:
    """Display table of flavor rows (evidence for each claim). ``by_unit`` puts the unit first
    (By Flavor page); otherwise the flavor first (Explore page)."""
    out = []
    for r in rows:
        rec = {}
        if by_unit:
            rec[UNIT_TYPE_SINGULAR_LABELS.get(unit_type, 'Unit')] = unit_label(unit_type, r['unit'])
        else:
            rec['Flavor'] = r['flavor']
        rec['% of its coffees'] = fmt_pct(r.get('share_in'))
        rec['% of other coffees'] = fmt_pct(r.get('share_rest'))
        lift = r.get('lift')
        rec['Times as often'] = "only here" if lift is None else f"{lift:.1f}x"
        rec['Coffees listing it'] = fmt_n(r.get('a'))
        if r.get('n_sellers_supporting') is not None:
            rec['Sellers'] = int(r['n_sellers_supporting'])
            share = r.get('top_seller_share')
            top = r.get('top_seller')
            rec['Top seller share'] = "n/a" if share is None else (
                f"{share * 100:.0f}%" + (f" ({top})" if top else ""))
        q = r.get('q')
        rec['q-value'] = "n/a" if q is None else f"{q:.1e}" if q < 0.001 else f"{q:.3f}"
        out.append(rec)
    return pd.DataFrame(out)


def params_caption(meta: Dict[str, Any]) -> str:
    p = (meta or {}).get('params', {})
    if not p:
        return ""
    return (f"Only {{types}} with at least {p.get('min_unit_coffees', 10)} coffees are analyzed, and a flavor "
            f"must appear in at least {p.get('min_flavor_coffees', 3)} of a group's coffees to be tested. "
            f"\"Distinctive\" means listed at least {p.get('min_lift', 1.5)}x as often as in other coffees, "
            f"statistically reliable after correcting for multiple comparisons (q < {p.get('q_threshold', 0.05)}).")


# --------------------------------------------------------------------------
# Statistics display helpers (p / q formatting, effect-size wording, chi-square status)
# --------------------------------------------------------------------------

EFFECT_METRIC_LABELS = {
    'cramers_v': "Cramér's V",
    'eta2_h': "Eta-squared (H)",
    'mean_cramers_v': "Mean Cramér's V across flavors",
}
EFFECT_METRIC_LABEL_FALLBACK = {      # older caches: infer the metric from the label text
    "Cramer's V": 'cramers_v', "Eta-squared (H)": 'eta2_h', "Epsilon-squared": 'eta2_h',
    "Avg Cramer's V": 'mean_cramers_v',
}
ETA2_NOTE = ("**Eta-squared (H)** is the effect size of the Kruskal-Wallis test (0-1): "
             "< 0.01 negligible, 0.01-0.06 small, 0.06-0.14 medium, > 0.14 large.")
CRAMERS_NOTE = ("**Cramér's V** is the effect size of the chi-square test (0-1): "
                "< 0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium, > 0.5 large.")


def fmt_p(p: Optional[float], method: Optional[str] = None, n_sim: Optional[int] = None) -> str:
    """'p = 0.012', or 'p < 0.001' (also for a Monte Carlo p-value sitting at its floor)."""
    if p is None:
        return "n/a"
    if method == 'monte_carlo' and n_sim and p <= 1.0 / (n_sim + 1) + 1e-9:
        return "p < 0.001"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def fmt_q(q: Optional[float]) -> str:
    """q-value column text."""
    if q is None:
        return "n/a"
    return f"{q:.1e}" if q < 0.001 else f"{q:.3f}"


def effect_label(es: Optional[float], metric: str = 'eta2_h') -> str:
    """negligible / small / medium / large for the given effect-size metric."""
    if es is None:
        return "unknown"
    cuts = (0.1, 0.3, 0.5) if metric in ('cramers_v', 'mean_cramers_v') else (0.01, 0.06, 0.14)
    return ("negligible" if es < cuts[0] else "small" if es < cuts[1]
            else "medium" if es < cuts[2] else "large")


def effect_metric_of(row: Dict[str, Any]) -> str:
    """effect_size_metric of an association row (falls back to the label for older caches)."""
    return row.get('effect_size_metric') or EFFECT_METRIC_LABEL_FALLBACK.get(row.get('effect_label', ''), 'other')


def collapsed_text(collapsed: Optional[Dict[str, Any]], threshold: Optional[int] = None) -> Optional[str]:
    """What was merged into 'Other' for a collapsed chi-square table (None if nothing)."""
    if not collapsed:
        return None
    rows, cols, dropped = collapsed.get('rows') or [], collapsed.get('cols') or [], collapsed.get('dropped') or []
    if not (rows or cols or dropped):
        return None
    parts = []
    if rows:
        parts.append("rows merged into 'Other': " + ", ".join(rows))
    if cols:
        parts.append("columns merged into 'Other': " + ", ".join(cols))
    if dropped:
        parts.append("dropped (still too small): " + ", ".join(dropped))
    lead = (f"To keep the test valid, categories with fewer than {threshold} coffees were combined. "
            if threshold else "To keep the test valid, small categories were combined. ")
    return lead + "; ".join(parts) + "."


def render_chi_square(chi2: Dict[str, Any], pair_label: str):
    """Chi-square result in an expander, handling every status:
    ok / collapsed / monte_carlo / insufficient (no p-value shown)."""
    if not chi2:
        return
    status = chi2.get('status')
    with st.expander(f"Statistical Test: Association between {pair_label}"):
        if status == 'insufficient' or not chi2.get('has_data'):
            st.info("Insufficient data for a chi-square test on this pair "
                    "(too few coffees or categories after combining small ones).")
            return
        v = chi2.get('cramers_v', 0)
        interp = chi2.get('effect_interpretation', effect_label(v, 'cramers_v'))
        method = chi2.get('p_value_method')
        n_sim = chi2.get('n_simulations')
        p_txt = fmt_p(chi2.get('p_value'), method, n_sim)
        if method == 'monte_carlo':
            p_txt += f" (Monte Carlo p, {n_sim:,} simulations)" if n_sim else " (Monte Carlo p)"
        body = (f"Chi2={chi2.get('chi2', 0):.1f}, {p_txt}, Cramér's V={v:.3f} [{interp}], "
                f"n={fmt_n(chi2.get('n_observations', 0))}")
        if as_bool(chi2.get('is_significant')):
            st.success(f"Statistically significant association ({body})")
        else:
            st.info(f"No significant association ({body})")
        note = collapsed_text(chi2.get('collapsed'), chi2.get('collapse_threshold'))
        if note:
            st.caption(note)
        if method == 'monte_carlo':
            st.caption("The usual chi-square approximation is unreliable for tables with many small counts, "
                       "so the p-value comes from shuffling the data 2,000 times and counting how often "
                       "a result this strong appears by chance." if not n_sim else
                       f"The usual chi-square approximation is unreliable for tables with many small counts, "
                       f"so the p-value comes from shuffling the data {n_sim:,} times and counting how often "
                       "a result this strong appears by chance.")
        st.caption(
            "**Why Chi-square?** This test is designed for comparing two categorical variables "
            "(e.g., country vs process method). It checks whether certain combinations occur more or "
            "less often than you'd expect if the two features were completely independent.  \n"
            + CRAMERS_NOTE
        )



def category_plural(label: str) -> str:
    """Lower-case plural of a category label: Country -> countries, Process Method -> process methods."""
    word = label.lower()
    if word.endswith('y') and word[-2:-1] not in 'aeiou':
        return word[:-1] + 'ies'
    return word + 's'
