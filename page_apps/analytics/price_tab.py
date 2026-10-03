"""
Price Analysis Tab Renderer

Displays price distributions, price by category comparisons,
and premium indicator analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

from analytics.frontend.flags import as_bool, significant_mask
from page_apps.analytics.common import ETA2_NOTE, category_plural, effect_label, fmt_p, fmt_q


def render_price_tab():
    """Render the Price Analysis tab"""
    st.header("Price Analysis")

    st.markdown("""
    **What you'll see here:** How coffee prices vary across origins, processing methods,
    varietals, and flavor profiles.

    **How to interpret:**
    - **Price definition:** price per lb is the *best per-lb price offered* for a coffee
      (usually the largest bag size), not an average across bag sizes
    - **Bar charts** show the median price with error bars spanning the interquartile range
      (25th to 75th percentile) for each group
    - **Kruskal-Wallis H test** checks whether price differences across 3+ groups are statistically significant —
      it's a rank-based test chosen because coffee prices are often skewed with outliers
    - **Mann-Whitney U test** compares exactly two groups (coffees WITH vs WITHOUT a specific flavor) —
      like Kruskal-Wallis but specialized for the two-group case
    - **q-value** is the p-value adjusted for the number of comparisons made; q < 0.05 means the difference is
      statistically significant (unlikely to be due to chance)
    - **Eta-squared (H)** is the effect size of the Kruskal-Wallis test: how *large* the difference actually is
      (a significant but tiny effect may not matter in practice)
    - **Varietal** prices use single-varietal coffees only (coffees listing several varietals are left out)
    """)

    from analytics.frontend.cached_data_loader import load_price_analysis_data
    price_data = load_price_analysis_data()

    if not price_data:
        st.info("Price analysis data not available. Please regenerate the analytics cache.")
        return

    _render_price_overview(price_data.get('overview', {}))

    # Sub-section selector
    section = st.radio(
        "Explore prices by:",
        ["Country", "Process Method", "Varietal", "Flavor Profile", "Premium Indicators"],
        horizontal=True
    )

    if section == "Country":
        _render_price_by_category(price_data.get('by_country', {}), 'Country')
    elif section == "Process Method":
        _render_price_by_category(price_data.get('by_process', {}), 'Process Method')
    elif section == "Varietal":
        _render_price_by_category(price_data.get('by_varietal', {}), 'Varietal')
    elif section == "Flavor Profile":
        _render_price_by_flavor(price_data)
    elif section == "Premium Indicators":
        _render_premium_indicators(price_data.get('premium_indicators', []))


def _render_price_overview(data: Dict[str, Any]):
    """Render price overview statistics"""
    if not data or not data.get('has_data'):
        st.warning("No price data available.")
        return

    st.subheader("Price Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Coffees with Price", data.get('total_with_price', 'N/A'))
    with col2:
        coverage = data.get('coverage_rate', 0)
        st.metric("Coverage Rate", f"{coverage:.0%}")
    with col3:
        st.metric("Median Price", f"${data.get('median', 0):.2f}/lb")
    with col4:
        st.metric("Mean Price", f"${data.get('mean', 0):.2f}/lb")

    # Histogram
    histogram = data.get('histogram', {})
    if histogram and histogram.get('counts'):
        fig = go.Figure(data=[go.Bar(
            x=histogram['bin_labels'],
            y=histogram['counts'],
            marker_color='#2E8B57'
        )])
        fig.update_layout(
            title="Price Distribution ($/lb)",
            xaxis_title="Price Range ($/lb, log-spaced bins)" if histogram.get('log_bins') else "Price Range",
            yaxis_title="Number of Coffees",
            xaxis_tickangle=45
        )
        st.plotly_chart(fig, use_container_width=True)

        window = ""
        if data.get('window_start') and data.get('window_end'):
            window = f" first seen {data['window_start']} through last seen {data['window_end']},"
        mix = ""
        if data.get('n_active') is not None and data.get('n_expired') is not None:
            mix = f" ({data['n_active']:,} active, {data['n_expired']:,} expired)"
        st.caption(
            f"Covers coffees{window} including both active and expired listings{mix}. "
            "Price per lb is the best per-lb price offered (usually the largest bag)."
        )

        q25 = data.get('q25', 0)
        q75 = data.get('q75', 0)
        max_price = data.get('max', 0)
        if q25 and q75:
            st.caption(
                f"Most coffees fall in the **${q25:.2f}-${q75:.2f}/lb** range "
                f"(middle 50%). Outliers extend up to ${max_price:.2f}/lb."
            )

    with st.expander("Detailed Statistics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min", f"${data.get('min', 0):.2f}/lb")
            st.metric("25th Percentile", f"${data.get('q25', 0):.2f}/lb")
        with col2:
            st.metric("Median", f"${data.get('median', 0):.2f}/lb")
            st.metric("Std Dev", f"${data.get('std', 0):.2f}")
        with col3:
            st.metric("75th Percentile", f"${data.get('q75', 0):.2f}/lb")
            st.metric("Max", f"${data.get('max', 0):.2f}/lb")


def _render_price_by_category(data: Dict[str, Any], category_label: str):
    """Render price distributions by a categorical variable"""
    if not data or not data.get('has_data'):
        st.info(f"Not enough data to analyze prices by {category_label}.")
        return

    st.subheader(f"Price by {category_label}")

    groups = data.get('groups', [])
    if not groups:
        return

    group_df = pd.DataFrame(groups)

    # Bar of the median with interquartile-range (q25-q75) error bars.
    # Older caches have no q25/q75 columns: show bars without error bars.
    sorted_df = group_df.sort_values('median', ascending=False)
    iqr_kwargs = {}
    if 'q25' in sorted_df.columns and 'q75' in sorted_df.columns:
        sorted_df = sorted_df.assign(
            _err_up=(sorted_df['q75'] - sorted_df['median']).clip(lower=0),
            _err_down=(sorted_df['median'] - sorted_df['q25']).clip(lower=0),
        )
        iqr_kwargs = {'error_y': '_err_up', 'error_y_minus': '_err_down'}
    fig = px.bar(
        sorted_df,
        x='name',
        y='median',
        **iqr_kwargs,
        title=f"Median Price by {category_label}",
        labels={'median': 'Median Price ($/lb)', 'name': category_label},
        text=sorted_df['count'].apply(lambda x: f'n={x}'),
        color='median',
        color_continuous_scale='Viridis',
    )
    fig.update_layout(xaxis_tickangle=45, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Data table
    with st.expander(f"Full {category_label} Price Data"):
        display_df = group_df.rename(columns={
            'name': category_label,
            'count': 'Count',
            'mean': 'Mean ($/lb)',
            'median': 'Median ($/lb)',
            'min': 'Min ($/lb)',
            'max': 'Max ($/lb)',
            'q_value': 'q-value',
        })
        if 'q-value' in display_df.columns:
            display_df['q-value'] = display_df['q-value'].apply(fmt_q)
        for col in ['Mean ($/lb)', 'Median ($/lb)', 'Min ($/lb)', 'Max ($/lb)']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
        cols = [category_label, 'Count', 'Median ($/lb)', 'Mean ($/lb)', 'Min ($/lb)', 'Max ($/lb)']
        if 'q-value' in display_df.columns:
            cols.append('q-value')   # each group vs all other coffees, BH-adjusted
        st.dataframe(display_df[cols], hide_index=True)

    # Statistical test
    kw = data.get('kruskal_wallis', {})
    if kw.get('has_data'):
        groups_word = category_plural(category_label)
        with st.expander(f"Statistical Test: Price differences across {groups_word}"):
            es = kw.get('eta_squared_h', kw.get('effect_size', 0))
            es_label = effect_label(es, 'eta2_h')

            if as_bool(kw.get('is_significant')):
                st.success(
                    f"Statistically significant price differences across {groups_word} "
                    f"(H={kw['h_statistic']:.2f}, {fmt_p(kw['p_value'])}, "
                    f"eta-squared (H)={es:.3f} [{es_label}])"
                )
            else:
                st.info(
                    f"No statistically significant price differences across {groups_word} "
                    f"(H={kw['h_statistic']:.2f}, {fmt_p(kw['p_value'])})"
                )
            st.caption(
                "**Why Kruskal-Wallis?** This test compares a numeric variable (price) across multiple "
                "groups (e.g., countries). It's chosen over the standard ANOVA because coffee prices aren't "
                "normally distributed — Kruskal-Wallis works by comparing *rank order* rather than raw values, "
                "making it robust to outliers and skewed data.  \n"
                + ETA2_NOTE + (
                    "  \nPrices by varietal use single-varietal coffees only." if 'Varietal' in category_label else "")
            )


def _render_price_by_flavor(price_data: Dict[str, Any]):
    """Render price-flavor correlation analysis"""
    st.subheader("Price by Flavor Profile")

    st.markdown("""
    This analysis compares the median price of coffees **with** a given flavor
    vs coffees **without** that flavor using a **Mann-Whitney U test** — a non-parametric
    test designed for comparing two groups. It's chosen here because each flavor splits coffees
    into exactly two groups (has flavor / doesn't have flavor), and prices aren't normally distributed.
    A significant result means the flavor is associated with higher or lower prices.
    """)

    # Level selector
    level = st.selectbox("Taxonomy Level:", ["Family", "Genus", "Species"], key="price_flavor_level")

    data_key = f'by_flavor_{level.lower()}'
    data = price_data.get(data_key, {})

    if not data or not data.get('has_data'):
        st.info(f"Not enough data for price-flavor analysis at the {level.lower()} level.")
        return

    flavors = data.get('flavors', [])
    if not flavors:
        return

    flavor_df = pd.DataFrame(flavors)

    # Bar chart: price difference
    sig_df = flavor_df[significant_mask(flavor_df['is_significant'])].copy()
    if not sig_df.empty:
        st.write(f"**{len(sig_df)} flavors with statistically significant price associations:**")

        fig = px.bar(
            sig_df.sort_values('price_difference', ascending=False),
            x='flavor',
            y='price_difference',
            title=f"Price Premium/Discount by Flavor ({level} Level) - Significant Only",
            labels={'price_difference': 'Median Price Difference ($/lb)', 'flavor': 'Flavor'},
            color='price_difference',
            color_continuous_scale='RdYlGn',
            text=sig_df.sort_values('price_difference', ascending=False)['count_with'].apply(lambda x: f'n={x}'),
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # Full table
    with st.expander("All Flavor-Price Associations"):
        display_df = flavor_df.rename(columns={
            'flavor': 'Flavor',
            'count_with': 'With Flavor',
            'count_without': 'Without Flavor',
            'median_price_with': 'Median $ (with)',
            'median_price_without': 'Median $ (without)',
            'price_difference': 'Difference',
            'q_value': 'q-value',
            'is_significant': 'Significant?',
        })
        if 'q-value' in display_df.columns:
            display_df['q-value'] = display_df['q-value'].apply(fmt_q)
            cols = ['Flavor', 'With Flavor', 'Median $ (with)', 'Median $ (without)', 'Difference', 'q-value', 'Significant?']
        else:   # older cache: raw p-values only
            display_df = display_df.rename(columns={'p_value': 'P-Value'})
            cols = ['Flavor', 'With Flavor', 'Median $ (with)', 'Median $ (without)', 'Difference', 'P-Value', 'Significant?']
        st.dataframe(display_df[cols], hide_index=True)


def _render_premium_indicators(indicators: List[Dict[str, Any]]):
    """Render premium indicator ranking"""
    st.subheader("Premium Indicators")

    st.markdown("""
    Features significantly associated with higher-priced coffees, ranked by **premium**:
    the group's median price per lb minus the overall median. Only positive, statistically
    significant premiums (Benjamini-Hochberg adjusted q < 0.05) are shown. This combines origin, process, and flavor analyses.
    """)

    if not indicators:
        st.info("Not enough data to identify premium indicators.")
        return

    ind_df = pd.DataFrame(indicators)
    if 'price_premium' not in ind_df.columns:
        st.info("Premium data is not in this cache version. Please regenerate the analytics cache.")
        return
    ind_df = ind_df[ind_df['price_premium'] > 0].sort_values('price_premium', ascending=False)
    if ind_df.empty:
        st.info("No significant positive price premiums found.")
        return

    # Color by type
    color_map = {'origin': '#2E8B57', 'process': '#4682B4', 'flavor': '#DAA520'}

    fig = px.bar(
        ind_df.head(15),
        x='value',
        y='price_premium',
        color='feature',
        title="Top Premium Indicators",
        labels={'price_premium': 'Premium over overall median ($/lb)', 'value': 'Feature Value', 'feature': 'Category'},
        text=ind_df.head(15)['count'].apply(lambda x: f'n={x}'),
    )
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # Table
    with st.expander("Full Premium Indicators Table"):
        display_df = ind_df.rename(columns={
            'feature': 'Category',
            'value': 'Feature',
            'median_price': 'Median Price ($/lb)',
            'count': 'Sample Size',
        })
        display_df['Premium ($/lb)'] = ind_df['price_premium'].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else ''
        )
        cols = ['Category', 'Feature', 'Premium ($/lb)', 'Median Price ($/lb)', 'Sample Size']
        if 'q_value' in ind_df.columns:
            display_df['q-value'] = ind_df['q_value'].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else ''
            )
            cols.append('q-value')
        st.dataframe(display_df[cols], hide_index=True)
