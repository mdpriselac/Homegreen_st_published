"""
Process & Varietal Tab Renderer

Displays process method distributions by origin, varietal distributions by origin,
flavor profiles by process/varietal, and process x varietal associations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

from analytics.constants import unit_label
from page_apps.analytics.common import fmt_n, render_chi_square


def render_process_varietal_tab():
    """Render the Process & Varietal analytics tab"""
    st.header("Process & Varietal Analysis")

    st.markdown("""
    **What you'll see here:** How processing methods and varietals distribute across
    origins, and how they relate to flavor profiles and each other.

    **How to interpret:**
    - **Stacked bars** show the percentage breakdown of process methods within each country
    - **Heatmaps** show which varietals are most common in which origins (darker = more common)
    - **Chi-square test** checks whether two categorical features are associated (e.g., "does the
      distribution of process methods differ significantly across countries?"). It's the standard
      test for categorical-vs-categorical data.
    - **Cramer's V** is the effect size for chi-square — it tells you *how strong* the association is,
      not just whether it exists
    """)

    from analytics.frontend.cached_data_loader import load_cross_feature_data
    data = load_cross_feature_data()

    if not data or not data.get('process_by_origin'):
        st.info("Cross-feature data not available. Please regenerate the analytics cache.")
        return

    section = st.radio(
        "Explore:",
        [
            "Process by Origin",
            "Varietal by Origin",
            "Flavor by Process",
            "Flavor by Varietal",
            "Process x Varietal",
        ],
        horizontal=True,
        key="pv_section",
    )

    if section == "Process by Origin":
        _render_process_by_origin(data.get('process_by_origin', {}))
    elif section == "Varietal by Origin":
        _render_varietal_by_origin(data.get('varietal_by_origin', {}))
    elif section == "Flavor by Process":
        _render_flavor_by_group(
            data, 'flavor_by_process', 'Process Method', 'process'
        )
    elif section == "Flavor by Varietal":
        _render_flavor_by_group(
            data, 'flavor_by_varietal', 'Varietal', 'varietal'
        )
    elif section == "Process x Varietal":
        _render_process_by_varietal(data.get('process_by_varietal', {}))


# --------------------------------------------------------------------------
# Process by Origin
# --------------------------------------------------------------------------

def _render_process_by_origin(data: Dict[str, Any]):
    """Render process method distribution by country/region"""
    view = st.radio("View by:", ["Country", "Region"], horizontal=True, key="pbo_view")
    key = 'by_country' if view == "Country" else 'by_region'
    section_data = data.get(key, {})

    if not section_data or not section_data.get('has_data'):
        st.info(f"Not enough data to show process methods by {view.lower()}.")
        return

    stacked = section_data.get('stacked_data', {})
    groups = stacked.get('groups', [])
    categories = stacked.get('categories', [])
    percentages = stacked.get('percentages', [])
    if view == "Region":
        groups = [unit_label('region', g) for g in groups]   # "Country / Subregion"

    if groups and categories and percentages:
        # Stacked bar chart (percentage)
        fig = go.Figure()
        for j, cat in enumerate(categories):
            fig.add_trace(go.Bar(
                name=cat,
                x=groups,
                y=[row[j] * 100 for row in percentages],
            ))
        fig.update_layout(
            barmode='stack',
            title=f"Process Method Distribution by {view}",
            xaxis_title=view,
            yaxis_title="Percentage",
            xaxis_tickangle=45,
            legend_title="Process Method",
        )
        st.plotly_chart(fig, use_container_width=True)

    _render_chi_square_result(section_data.get('chi_square', {}),
                              f"process method and {view.lower()}")


# --------------------------------------------------------------------------
# Varietal by Origin
# --------------------------------------------------------------------------

def _render_varietal_by_origin(data: Dict[str, Any]):
    """Render varietal distribution by country (heatmap)"""
    if not data or not data.get('has_data'):
        st.info("Not enough data for varietal-by-origin analysis.")
        return

    st.subheader("Varietal Distribution by Country")

    show_pct = st.checkbox("Show percentages (row-normalized)", value=True, key="vbo_pct")
    heatmap_key = 'heatmap_pct' if show_pct else 'heatmap_counts'
    hm = data.get(heatmap_key, {})

    rows = hm.get('rows', [])
    cols = hm.get('columns', [])
    values = hm.get('values', [])

    if rows and cols and values:
        fig = px.imshow(
            values,
            x=cols,
            y=rows,
            title="Varietal by Country" + (" (%)" if show_pct else " (coffee count; multi-varietal coffees are split evenly)"),
            color_continuous_scale='YlGnBu',
            aspect='auto',
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    _render_chi_square_result(data.get('chi_square', {}), "varietal and country")


# --------------------------------------------------------------------------
# Flavor by Process / Varietal
# --------------------------------------------------------------------------

def _render_flavor_by_group(all_data: Dict[str, Any],
                            prefix: str, group_label: str, short_key: str):
    """Render flavor profiles grouped by process method or varietal"""

    level = st.selectbox(
        "Taxonomy level:",
        ["Family", "Genus"],
        key=f"fbg_level_{short_key}",
    )
    data_key = f'{prefix}_{level.lower()}'
    data = all_data.get(data_key, {})

    if not data or not data.get('has_data'):
        st.info(f"Not enough data for flavor-by-{group_label.lower()} at {level.lower()} level.")
        return

    profiles = data.get('profiles', {})
    distinctive = data.get('distinctive', {})

    if not profiles:
        return

    group_names = sorted(profiles.keys())
    selected = st.selectbox(f"Select {group_label}:", group_names, key=f"fbg_sel_{short_key}")

    if selected and selected in profiles:
        profile = profiles[selected]
        n_distinct = profile.get('n_coffees')
        eff = profile.get('total_coffees', 0)
        if n_distinct is not None and abs(float(n_distinct) - float(eff)) > 0.05:
            st.write(f"**{selected}** — {fmt_n(n_distinct)} coffees (counted as {fmt_n(eff)} "
                     "after splitting multi-varietal coffees evenly)")
        else:
            st.write(f"**{selected}** — {fmt_n(n_distinct if n_distinct is not None else eff)} coffees")

        flavor_df = pd.DataFrame(profile['flavors'][:20])
        if not flavor_df.empty:
            fig = px.bar(
                flavor_df,
                x='flavor',
                y='rate',
                title=f"Flavor Profile of {selected} ({level} Level)",
                labels={'rate': 'Proportion of Coffees', 'flavor': f'Flavor {level}'},
                text=flavor_df['count'].apply(lambda x: f'n={fmt_n(x)}'),
            )
            fig.update_layout(xaxis_tickangle=45, yaxis_tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)

    # Distinctive flavors
    if distinctive:
        with st.expander(f"Distinctive flavors for each {group_label.lower()}"):
            st.caption(
                "**Over-represented flavors** appear more often in this group than across all coffees. "
                "The ratio (e.g., 2.0x) means the flavor is that many times more common here "
                "than the global average. A ratio of 2.0x = twice as common; 3.0x = three times as common."
            )
            for group_name in group_names:
                dist_list = distinctive.get(group_name, [])
                if dist_list:
                    top = [d for d in dist_list if d.get('ratio', 0) > 1.2][:5]
                    if top:
                        st.write(f"**{group_name}** — most over-represented flavors:")
                        for d in top:
                            st.write(
                                f"  - {d['flavor']}: "
                                f"{d['local_rate']:.0%} vs {d['global_rate']:.0%} global "
                                f"({d['ratio']:.1f}x)"
                            )


# --------------------------------------------------------------------------
# Process x Varietal
# --------------------------------------------------------------------------

def _render_process_by_varietal(data: Dict[str, Any]):
    """Render process method x varietal association heatmap"""
    if not data or not data.get('has_data'):
        st.info("Not enough data for process x varietal analysis.")
        return

    st.subheader("Process Method by Varietal")

    show_pct = st.checkbox("Show percentages", value=True, key="pxv_pct")
    heatmap_key = 'heatmap_pct' if show_pct else 'heatmap_counts'
    hm = data.get(heatmap_key, {})

    rows = hm.get('rows', [])
    cols = hm.get('columns', [])
    values = hm.get('values', [])

    if rows and cols and values:
        fig = px.imshow(
            values,
            x=cols,
            y=rows,
            title="Process Method by Varietal" + (" (%)" if show_pct else " (coffee count; multi-varietal coffees are split evenly)"),
            color_continuous_scale='YlOrRd',
            aspect='auto',
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    _render_chi_square_result(data.get('chi_square', {}), "process method and varietal")


# --------------------------------------------------------------------------
# Common helper
# --------------------------------------------------------------------------

def _render_chi_square_result(chi2: Dict[str, Any], pair_label: str):
    """Chi-square result: handles ok / collapsed / monte_carlo / insufficient statuses"""
    render_chi_square(chi2, pair_label)
