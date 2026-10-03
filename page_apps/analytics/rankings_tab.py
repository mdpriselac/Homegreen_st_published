"""
Rankings tab.

Flavor-profile rankings come from the per-coffee distinctiveness analysis and are
computed separately per unit type (country / region / seller) and taxonomy level,
only for groups with enough coffees. Price rankings are country-level. The lifespan
rankings (Fastest Moving / Longest Lasting) are hidden together with the Turnover tab.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from analytics.frontend.cached_data_loader import load_rankings_data
from analytics.constants import unit_plural
from page_apps.analytics.common import (
    LEVELS_NOTE, REGENERATE_MSG, UNIT_TYPE_PLURAL_LABELS, UNIT_TYPES, level_selector,
)
from page_apps.analytics.config import SHOW_TURNOVER_TAB

MIN_COFFEES_FLOOR = 10      # analysis minimum (MIN_UNIT_COFFEES)
MIN_COFFEES_DEFAULT = 20    # default display minimum

PROFILE_CATEGORIES = {
    "Most distinctive flavor profile": 'distinctive_profile',
    "Most varied flavor profile": 'varied_profile',
}
PRICE_CATEGORIES = {
    "Best Value (Lowest Price)": 'best_value',
    "Highest Priced": 'highest_priced',
}
LIFESPAN_CATEGORIES = {
    "Fastest Moving (Shortest Lifespan)": 'fastest_moving',
    "Longest Lasting": 'longest_lasting',
}

JSD_COPY = (
    "This measures how different the overall flavor mix is from the rest of the market. "
    "An origin can rank high through many small shifts even if no single flavor stands out on its own."
)
VARIED_COPY = ("This measures how evenly the group's coffees spread across flavors: "
               "0 means one flavor dominates, 1 means all flavors appear equally.")
LIFESPAN_COPY = ("Ranked by the median time a coffee stays listed, estimated so that coffees still "
                 "listed count as \"at least this long\". Groups where fewer than half of the coffees have "
                 "left yet have no median and are not ranked.")
SELLER_CAPTION = "A seller's profile reflects both which origins they stock and how they write tasting notes."


def ranking_categories() -> List[str]:
    cats = list(PROFILE_CATEGORIES) + list(PRICE_CATEGORIES)
    if SHOW_TURNOVER_TAB:
        cats += list(LIFESPAN_CATEGORIES)
    return cats


def render_rankings_tab():
    """Render the rankings tab"""
    st.header("Rankings & Leaderboards")

    lifespan_line = ("\n    - **Fastest Moving / Longest Lasting**: ranked by median listing lifespan (countries or sellers)."
                     if SHOW_TURNOVER_TAB else "")
    st.markdown(f"""
    **What you'll see here:** Rankings of origins and sellers on different criteria.

    **How to interpret:**
    - **Most distinctive flavor profile**: {JSD_COPY}
    - **Most varied flavor profile**: {VARIED_COPY}
    - **Best Value / Highest Priced**: ranked by median price per lb (countries only).{lifespan_line}
    - Flavor-profile rankings are done separately for countries, regions and sellers, and only include groups with enough coffees.
    """)

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Ranking Category:", ranking_categories())
    with col2:
        min_coffees = st.slider("Minimum Coffees:", MIN_COFFEES_FLOOR, 100, MIN_COFFEES_DEFAULT)

    if category in PROFILE_CATEGORIES:
        c1, c2 = st.columns(2)
        with c1:
            unit_type = st.radio("Rank:", [UNIT_TYPE_PLURAL_LABELS[t] for t in UNIT_TYPES],
                                 horizontal=True, key="rank_unit_type")
        unit_type = {v: k for k, v in UNIT_TYPE_PLURAL_LABELS.items()}[unit_type]
        with c2:
            level = level_selector("rank_level", "Flavor level:")
        rankings_df = generate_profile_rankings(category, unit_type, level, min_coffees)
        if rankings_df is None:
            st.info(REGENERATE_MSG)
            return
        st.caption(JSD_COPY if PROFILE_CATEGORIES[category] == 'distinctive_profile' else VARIED_COPY)
        st.caption(f"Only {unit_plural(unit_type)} with at least {min_coffees} coffees are ranked. Small groups are pulled "
                   "toward the market average so a handful of coffees can't top the list.")
        if unit_type == 'seller':
            st.caption(SELLER_CAPTION)
        display_profile_rankings(rankings_df, category, unit_type, level)
    else:
        unit_type = None
        if category in LIFESPAN_CATEGORIES:
            choice = st.radio("Rank:", ["Countries", "Sellers"], horizontal=True, key="rank_lifespan_type")
            unit_type = 'country' if choice == "Countries" else 'seller'
            st.caption(LIFESPAN_COPY)
        rankings_df = generate_rankings(category, min_coffees, unit_type)
        display_rankings(rankings_df, category)




def generate_profile_rankings(category: str, unit_type: str, level: str, min_coffees: int):
    """Rows for a flavor-profile ranking; None when the cache has no such data."""
    rankings = load_rankings_data() or {}
    key = PROFILE_CATEGORIES.get(category)
    by_type = rankings.get(key)
    if by_type is None:
        return None
    rows = by_type.get(unit_type, {}).get(level, [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df['n_coffees'] >= min_coffees].sort_values('score', ascending=False)
    return df.reset_index(drop=True)


def display_profile_rankings(df: pd.DataFrame, category: str, unit_type: str, level: str):
    st.subheader(f"Rankings: {category} ({UNIT_TYPE_PLURAL_LABELS[unit_type].lower()}, {level} level)")
    if df.empty:
        st.info("No groups meet the selected minimum number of coffees.")
        return
    show = df.head(25).copy()
    show.insert(0, 'Rank', range(1, len(show) + 1))
    show['Standout flavors'] = show['signature'].apply(
        lambda s: ', '.join(s) if isinstance(s, list) and s else 'none stands out')
    show = show.rename(columns={'label': 'Name', 'n_coffees': 'Coffees', 'score': 'Score'})
    st.dataframe(
        show[['Rank', 'Name', 'Coffees', 'Score', 'Standout flavors']],
        column_config={"Score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=float(show['Score'].max() or 1), format="%.3f")},
        hide_index=True,
    )


def generate_rankings(category: str, min_coffees: int, unit_type: Optional[str] = None) -> pd.DataFrame:
    """Price rankings (country-level) and, when enabled, lifespan rankings (country or seller)"""
    rankings_data = load_rankings_data() or {}
    key = {**PRICE_CATEGORIES, **LIFESPAN_CATEGORIES}.get(category)
    rankings = rankings_data.get(key, []) if key else []
    if not rankings:
        return pd.DataFrame()

    rankings_df = pd.DataFrame(rankings)
    if unit_type and 'unit_type' in rankings_df.columns:
        rankings_df = rankings_df[rankings_df['unit_type'] == unit_type]
    coffee_col = 'total_coffees' if 'total_coffees' in rankings_df.columns else 'coffee_count'
    if coffee_col in rankings_df.columns:
        rankings_df = rankings_df[rankings_df[coffee_col] >= min_coffees]
    ascending = category in {"Best Value (Lowest Price)", "Fastest Moving (Shortest Lifespan)"}
    if 'score' in rankings_df.columns:
        rankings_df = rankings_df.sort_values('score', ascending=ascending)
    return rankings_df


def display_rankings(rankings_df: pd.DataFrame, category: str):
    """Display price/lifespan rankings"""
    st.subheader(f"Rankings: {category}")
    if rankings_df.empty:
        st.info("No data available for rankings")
        return

    show = rankings_df.head(25).copy().reset_index(drop=True)
    show.insert(0, 'Rank', [str(i + 1) for i in range(len(show))])
    rename = {'entity_name': 'Name', 'unit': 'Name', 'unit_type': 'Type', 'type': 'Type',
              'total_coffees': 'Coffees', 'coffee_count': 'Coffees'}
    show = show.rename(columns={k: v for k, v in rename.items() if k in show.columns})
    if 'top_flavors' in show.columns:
        show = show.drop(columns=['top_flavors'])
    if category in PRICE_CATEGORIES:
        show = show.rename(columns={'score': 'Median Price ($/lb)', 'mean_price': 'Mean Price ($/lb)'})
    elif category in LIFESPAN_CATEGORIES:
        show = show.rename(columns={'score': 'Median Lifespan (days)'})
        if {'q25_lifespan', 'q75_lifespan'} <= set(show.columns):
            show['Middle half (days)'] = [f"{a:.0f} to {b:.0f}" if pd.notna(a) and pd.notna(b) else ""
                                          for a, b in zip(show['q25_lifespan'], show['q75_lifespan'])]
            show = show.drop(columns=['q25_lifespan', 'q75_lifespan'])
        show = show.drop(columns=[c for c in ('events', 'censored', 'mean_lifespan') if c in show.columns])
    st.dataframe(show, hide_index=True)
