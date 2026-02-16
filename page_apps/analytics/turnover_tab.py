"""
Seller Turnover Tab Renderer

Displays coffee lifespan distributions, seller turnover metrics,
and temporal patterns.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any


def render_turnover_tab():
    """Render the Seller Turnover analytics tab"""
    st.header("Coffee Lifespan & Seller Turnover")

    st.markdown("""
    **What you'll see here:** How long coffees remain listed before being removed,
    turnover patterns by seller, and how lifespan relates to origin, process, and price.

    **How to interpret:**
    - **Lifespan** = days between first observed and last observed (expired coffees only)
    - **Turnover Rate** = proportion of a seller's coffees that have expired
    - Shorter lifespans may indicate faster-selling or more seasonal coffees
    """)

    from analytics.frontend.cached_data_loader import load_turnover_data
    turnover_data = load_turnover_data()

    if not turnover_data or not turnover_data.get('has_data', False):
        # Check if sub-sections have data
        overview = turnover_data.get('lifespan_overview', {}) if turnover_data else {}
        if not overview.get('has_data', False):
            st.info("Turnover data not available. Please regenerate the analytics cache.")
            return

    # Sub-sections
    _render_lifespan_overview(turnover_data.get('lifespan_overview', {}))
    _render_seller_turnover(turnover_data.get('by_seller', {}), turnover_data.get('seller_summary', []))
    _render_lifespan_by_origin(turnover_data.get('by_origin', {}))
    _render_lifespan_by_process(turnover_data.get('by_process', {}))
    _render_lifespan_by_price(turnover_data.get('by_price', {}))
    _render_seasonal_patterns(turnover_data.get('seasonal_patterns', {}))


def _render_lifespan_overview(data: Dict[str, Any]):
    """Render overall lifespan distribution"""
    if not data or not data.get('has_data'):
        return

    st.subheader("Overall Lifespan Distribution")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Expired Coffees", data.get('total_expired', 'N/A'))
    with col2:
        st.metric("Active Coffees", data.get('total_active', 'N/A'))
    with col3:
        median = data.get('median_days')
        st.metric("Median Lifespan", f"{median:.0f} days" if median is not None else 'N/A')
    with col4:
        mean = data.get('mean_days')
        st.metric("Mean Lifespan", f"{mean:.0f} days" if mean is not None else 'N/A')

    if median is not None:
        st.caption(
            f"A median lifespan of **{median:.0f} days** means half of expired coffees were "
            f"listed for more than {median:.0f} days and half for less. "
            f"The mean ({mean:.0f} days) being {'higher' if mean and mean > median else 'lower'} "
            f"suggests {'some coffees stay listed much longer than typical, pulling the average up' if mean and mean > median else 'the distribution is fairly symmetric'}."
        )

    # Histogram
    histogram = data.get('histogram', {})
    if histogram and histogram.get('counts'):
        fig = go.Figure(data=[go.Bar(
            x=histogram['bin_labels'],
            y=histogram['counts'],
            marker_color='#8B4513'
        )])
        fig.update_layout(
            title="Distribution of Coffee Lifespans",
            xaxis_title="Lifespan (days)",
            yaxis_title="Number of Coffees",
            xaxis_tickangle=45
        )
        st.plotly_chart(fig, use_container_width=True)

    # Additional stats
    with st.expander("Detailed Statistics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min", f"{data.get('min_days', 'N/A')} days")
            st.metric("25th Percentile", f"{data.get('q25_days', 0):.0f} days")
        with col2:
            st.metric("Median", f"{data.get('median_days', 0):.0f} days")
            st.metric("Std Dev", f"{data.get('std_days', 0):.0f} days")
        with col3:
            st.metric("75th Percentile", f"{data.get('q75_days', 0):.0f} days")
            st.metric("Max", f"{data.get('max_days', 'N/A')} days")


def _render_seller_turnover(seller_data: Dict[str, Any], seller_summary: list):
    """Render seller-level turnover analysis"""
    st.subheader("Turnover by Seller")

    if seller_summary:
        summary_df = pd.DataFrame(seller_summary)
        if not summary_df.empty:
            display_df = summary_df.rename(columns={
                'seller': 'Seller',
                'total_coffees': 'Total',
                'active': 'Active',
                'expired': 'Expired',
                'turnover_rate': 'Turnover Rate',
                'median_lifespan_days': 'Median Lifespan (days)',
                'unique_countries': 'Countries',
            })

            st.dataframe(
                display_df,
                column_config={
                    'Turnover Rate': st.column_config.ProgressColumn(
                        "Turnover Rate", min_value=0, max_value=1, format="%.0%%"
                    ),
                },
                hide_index=True,
            )

    sellers = seller_data.get('sellers', [])
    if sellers:
        seller_df = pd.DataFrame(sellers)
        if not seller_df.empty and len(seller_df) > 1:
            fig = px.bar(
                seller_df.sort_values('median_lifespan'),
                x='seller',
                y='median_lifespan',
                error_y='std_lifespan',
                title="Median Coffee Lifespan by Seller",
                labels={'median_lifespan': 'Median Lifespan (days)', 'seller': 'Seller'},
                color='expired_count',
                color_continuous_scale='Viridis',
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        # Statistical test result
        kw = seller_data.get('kruskal_wallis', {})
        if kw.get('has_data'):
            _render_kruskal_wallis_result(kw, "sellers")


def _render_lifespan_by_origin(data: Dict[str, Any]):
    """Render lifespan by origin country"""
    if not data or not data.get('has_data'):
        return

    st.subheader("Lifespan by Origin Country")

    countries = data.get('countries', [])
    if countries:
        country_df = pd.DataFrame(countries)
        if not country_df.empty and len(country_df) > 1:
            fig = px.bar(
                country_df.sort_values('median_lifespan'),
                x='country',
                y='median_lifespan',
                error_y='std_lifespan',
                title="Median Coffee Lifespan by Country",
                labels={'median_lifespan': 'Median Lifespan (days)', 'country': 'Country'},
                color='count',
                color_continuous_scale='Viridis',
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        kw = data.get('kruskal_wallis', {})
        if kw.get('has_data'):
            _render_kruskal_wallis_result(kw, "countries")


def _render_lifespan_by_process(data: Dict[str, Any]):
    """Render lifespan by process type"""
    if not data or not data.get('has_data'):
        return

    st.subheader("Lifespan by Process Method")

    processes = data.get('processes', [])
    if processes:
        process_df = pd.DataFrame(processes)
        if not process_df.empty and len(process_df) > 1:
            fig = px.bar(
                process_df.sort_values('median_lifespan'),
                x='process',
                y='median_lifespan',
                error_y='std_lifespan',
                title="Median Coffee Lifespan by Process Method",
                labels={'median_lifespan': 'Median Lifespan (days)', 'process': 'Process'},
                text='count',
            )
            st.plotly_chart(fig, use_container_width=True)

        kw = data.get('kruskal_wallis', {})
        if kw.get('has_data'):
            _render_kruskal_wallis_result(kw, "process methods")


def _render_lifespan_by_price(data: Dict[str, Any]):
    """Render lifespan vs price analysis"""
    if not data or not data.get('has_data'):
        return

    st.subheader("Lifespan vs Price")

    # Correlation info
    corr = data.get('correlation')
    p_val = data.get('p_value')
    if corr is not None:
        # Interpret direction
        if abs(corr) < 0.1:
            direction = "essentially no relationship"
        elif corr > 0:
            strength = "weak" if abs(corr) < 0.3 else ("moderate" if abs(corr) < 0.6 else "strong")
            direction = f"{strength} positive (higher price tends to mean longer listing)"
        else:
            strength = "weak" if abs(corr) < 0.3 else ("moderate" if abs(corr) < 0.6 else "strong")
            direction = f"{strength} negative (higher price tends to mean shorter listing)"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Spearman Correlation", f"{corr:.3f}")
        with col2:
            st.metric("P-Value", f"{p_val:.4f}" if p_val is not None else 'N/A')
        with col3:
            sig = data.get('is_significant', False)
            st.metric("Significant?", "Yes" if sig else "No")

        st.caption(
            f"**Interpretation:** {direction.capitalize()}.  \n"
            "**Why Spearman?** This correlation is used instead of the standard Pearson correlation because "
            "it measures *monotonic* relationships (\"as one goes up, does the other tend to go up?\") without "
            "assuming a straight-line relationship or normally distributed data. It works on rank order, "
            "making it robust to outlier prices and extreme lifespans.  \n"
            "Range: -1 to +1. Near 0 = unrelated. Positive = higher price → longer listing. "
            "Negative = higher price → shorter listing."
        )

    # Scatter plot
    scatter = data.get('scatter_data', {})
    if scatter.get('prices') and scatter.get('lifespans'):
        fig = px.scatter(
            x=scatter['prices'],
            y=scatter['lifespans'],
            title="Price vs Lifespan",
            labels={'x': 'Average Price ($/lb)', 'y': 'Lifespan (days)'},
            opacity=0.6,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Quartile analysis
    quartiles = data.get('quartile_stats', [])
    if quartiles:
        q_df = pd.DataFrame(quartiles)
        if not q_df.empty:
            fig = px.bar(
                q_df,
                x='quartile',
                y='median_lifespan',
                title="Median Lifespan by Price Quartile",
                labels={'median_lifespan': 'Median Lifespan (days)', 'quartile': 'Price Range'},
                text='count',
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_seasonal_patterns(data: Dict[str, Any]):
    """Render seasonal appearance/disappearance patterns"""
    if not data or not data.get('has_data'):
        return

    st.subheader("Seasonal Patterns")

    appearances = data.get('appearances', [])
    disappearances = data.get('disappearances', [])

    if appearances:
        app_df = pd.DataFrame(appearances)
        if not app_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=app_df['month'],
                y=app_df['count'],
                name='New Listings',
                marker_color='#2E8B57',
            ))

            if disappearances:
                dis_df = pd.DataFrame(disappearances)
                if not dis_df.empty:
                    fig.add_trace(go.Bar(
                        x=dis_df['month'],
                        y=dis_df['count'],
                        name='Removed',
                        marker_color='#CD5C5C',
                    ))

            fig.update_layout(
                title="Monthly Coffee Listings: New vs Removed",
                xaxis_title="Month",
                yaxis_title="Count",
                barmode='group',
                xaxis_tickangle=45,
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_kruskal_wallis_result(kw: Dict[str, Any], group_label: str):
    """Render Kruskal-Wallis test result"""
    with st.expander(f"Statistical Test: Lifespan differences across {group_label}"):
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
                f"Statistically significant difference in lifespans across {group_label} "
                f"(H={kw['h_statistic']:.2f}, p={kw['p_value']:.4f}, "
                f"effect size={es:.3f} [{es_label}])"
            )
        else:
            st.info(
                f"No statistically significant difference in lifespans across {group_label} "
                f"(H={kw['h_statistic']:.2f}, p={kw['p_value']:.4f})"
            )
        st.caption(
            "**Why Kruskal-Wallis?** This test compares a numeric variable (lifespan) across multiple "
            "groups. It's chosen over ANOVA because lifespan data is typically right-skewed (a few coffees "
            "last much longer than most). Kruskal-Wallis uses rank ordering, making it robust to outliers.  \n"
            "**Effect size** (epsilon-squared) tells you how much lifespans actually differ: "
            "< 0.01 negligible, 0.01-0.06 small, 0.06-0.14 medium, > 0.14 large."
        )
