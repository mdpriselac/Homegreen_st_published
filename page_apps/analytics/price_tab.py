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


def render_price_tab():
    """Render the Price Analysis tab"""
    st.header("Price Analysis")

    st.markdown("""
    **What you'll see here:** How coffee prices vary across origins, processing methods,
    varietals, and flavor profiles.

    **How to interpret:**
    - **Bar charts** show the median price with error bars (standard deviation) for each group
    - **Kruskal-Wallis H test** checks whether price differences across 3+ groups are statistically significant —
      it's a rank-based test chosen because coffee prices are often skewed with outliers
    - **Mann-Whitney U test** compares exactly two groups (coffees WITH vs WITHOUT a specific flavor) —
      like Kruskal-Wallis but specialized for the two-group case
    - **P-value** < 0.05 means the difference is statistically significant (unlikely to be due to chance)
    - **Effect size** tells you how *large* the difference actually is (a significant but tiny effect may not matter in practice)
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
            xaxis_title="Price Range",
            yaxis_title="Number of Coffees",
            xaxis_tickangle=45
        )
        st.plotly_chart(fig, use_container_width=True)

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

    # Box plot style visualization using bar + error bars
    fig = px.bar(
        group_df.sort_values('median', ascending=False),
        x='name',
        y='median',
        error_y=group_df.sort_values('median', ascending=False)['std'],
        title=f"Median Price by {category_label}",
        labels={'median': 'Median Price ($/lb)', 'name': category_label},
        text=group_df.sort_values('median', ascending=False)['count'].apply(lambda x: f'n={x}'),
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
        })
        for col in ['Mean ($/lb)', 'Median ($/lb)', 'Min ($/lb)', 'Max ($/lb)']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
        st.dataframe(display_df[[category_label, 'Count', 'Median ($/lb)', 'Mean ($/lb)', 'Min ($/lb)', 'Max ($/lb)']], hide_index=True)

    # Statistical test
    kw = data.get('kruskal_wallis', {})
    if kw.get('has_data'):
        with st.expander(f"Statistical Test: Price differences across {category_label.lower()}s"):
            es = kw.get('effect_size', 0)
            if es < 0.01:
                es_label = "negligible"
            elif es < 0.06:
                es_label = "small"
            elif es < 0.14:
                es_label = "medium"
            else:
                es_label = "large"

            if kw.get('is_significant'):
                st.success(
                    f"Statistically significant price differences across {category_label.lower()}s "
                    f"(H={kw['h_statistic']:.2f}, p={kw['p_value']:.4f}, "
                    f"effect size={es:.3f} [{es_label}])"
                )
            else:
                st.info(
                    f"No statistically significant price differences across {category_label.lower()}s "
                    f"(H={kw['h_statistic']:.2f}, p={kw['p_value']:.4f})"
                )
            st.caption(
                "**Why Kruskal-Wallis?** This test compares a numeric variable (price) across multiple "
                "groups (e.g., countries). It's chosen over the standard ANOVA because coffee prices aren't "
                "normally distributed — Kruskal-Wallis works by comparing *rank order* rather than raw values, "
                "making it robust to outliers and skewed data.  \n"
                "**Effect size** (epsilon-squared) measures how much prices actually differ across groups: "
                "< 0.01 negligible, 0.01-0.06 small, 0.06-0.14 medium, > 0.14 large."
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
    sig_df = flavor_df[flavor_df['is_significant'] == True].copy()
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
            'p_value': 'P-Value',
            'is_significant': 'Significant?',
        })
        st.dataframe(display_df[['Flavor', 'With Flavor', 'Median $ (with)', 'Median $ (without)', 'Difference', 'P-Value', 'Significant?']], hide_index=True)


def _render_premium_indicators(indicators: List[Dict[str, Any]]):
    """Render premium indicator ranking"""
    st.subheader("Premium Indicators")

    st.markdown("""
    Features most associated with higher-priced coffees, ranked by median price.
    This combines insights from origin, process, and flavor analyses.
    """)

    if not indicators:
        st.info("Not enough data to identify premium indicators.")
        return

    ind_df = pd.DataFrame(indicators)

    # Color by type
    color_map = {'origin': '#2E8B57', 'process': '#4682B4', 'flavor': '#DAA520'}

    fig = px.bar(
        ind_df.head(15),
        x='value',
        y='median_price',
        color='feature',
        title="Top Premium Indicators",
        labels={'median_price': 'Median Price ($/lb)', 'value': 'Feature Value', 'feature': 'Category'},
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
        cols = ['Category', 'Feature', 'Median Price ($/lb)', 'Sample Size']
        if 'price_premium' in ind_df.columns:
            display_df['Price Premium'] = ind_df['price_premium'].apply(
                lambda x: f"${x:.2f}" if pd.notna(x) else ''
            )
            cols.append('Price Premium')
        st.dataframe(display_df[cols], hide_index=True)
