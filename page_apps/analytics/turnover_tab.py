"""
Seller Turnover Tab Renderer

How long coffees stay listed, estimated with survival analysis (Kaplan-Meier):
coffees that are still listed count as "at least this long" instead of being
dropped, and only coffees that appeared after the first scrape's initial
inventory are used. Group differences use log-rank tests with
Benjamini-Hochberg q-values. The tab is hidden by SHOW_TURNOVER_TAB until the
underlying seller data is verified.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics.frontend.flags import as_bool
from page_apps.analytics.common import fmt_n, fmt_p, fmt_q

NOT_REACHED = "not reached"
NOT_REACHED_NOTE = ("Not reached means fewer than half of this group's coffees have left yet, "
                    "so a typical lifespan can't be estimated.")


def fmt_days(x: Optional[float]) -> str:
    """Days with no decimals; None means the median was not reached."""
    return NOT_REACHED if x is None else f"{x:.0f}"


def _interval_phrase(window: Dict[str, Any]) -> str:
    days = window.get('scrape_interval_days')
    if days is None:
        return "regular scrapes"
    return "weekly scrapes" if abs(days - 7) < 0.5 else f"scrapes about every {days:.0f} days"


def render_turnover_tab():
    """Render the Seller Turnover analytics tab"""
    st.header("Coffee Lifespan & Seller Turnover")

    from analytics.frontend.cached_data_loader import load_turnover_data
    turnover_data = load_turnover_data()

    overview = (turnover_data or {}).get('lifespan_overview', {})
    if not overview.get('has_data', False):
        st.info("Turnover data not available. Please regenerate the analytics cache.")
        return

    window = turnover_data.get('observation_window') or overview.get('observation_window') or {}
    _render_method_caption(overview, window)
    _render_method_expander(turnover_data, overview, window)

    _render_lifespan_overview(overview)
    _render_seller_turnover(turnover_data.get('by_seller', {}), turnover_data.get('seller_summary', []),
                            window)
    _render_group_survival(turnover_data.get('by_origin', {}), 'countries', 'country',
                           "Lifespan by Origin Country", "Country")
    _render_group_survival(turnover_data.get('by_process', {}), 'processes', 'process',
                           "Lifespan by Process Method", "Process")
    _render_lifespan_by_price(turnover_data.get('by_price', {}))
    _render_seasonal_patterns(turnover_data.get('seasonal_patterns', {}))


# --------------------------------------------------------------------------
# Method text
# --------------------------------------------------------------------------

def _render_method_caption(overview: Dict[str, Any], window: Dict[str, Any]):
    """Two short sentences: what is measured and the typical listing."""
    cutoff = window.get('initial_inventory_cutoff') or window.get('window_start') or "the start of tracking"
    median = overview.get('median_days')
    typical = (f"{median:.0f} days" if median is not None
               else "not yet known (fewer than half have left yet)")
    st.caption(
        f"Lifespans are estimated from {_interval_phrase(window)} for coffees first listed after {cutoff}; "
        f"coffees still listed count as \"at least this long\". Typical listing: {typical}."
    )


def _render_method_expander(data: Dict[str, Any], overview: Dict[str, Any], window: Dict[str, Any]):
    with st.expander("How we measure this"):
        ex = window.get('exclusions') or {}
        lines = []
        if window:
            lines.append(
                f"**Window.** Scraping started on {window.get('first_scrape', 'n/a')}; coffees already listed "
                f"in the first weeks (up to {window.get('initial_inventory_cutoff', 'n/a')}) are the initial "
                "inventory, and we can't tell how long they had been listed, so they are excluded. "
                f"Removals are only counted from {window.get('tracking_start', 'n/a')}, when removal tracking began. "
                f"The data runs to {window.get('window_end', 'n/a')}.")
        if ex:
            lines.append(
                f"**Coffees used.** {fmt_n(ex.get('eligible', 0))} of {fmt_n(ex.get('total_coffees', 0))} coffees are "
                f"eligible ({fmt_n(ex.get('eligible_expired', 0))} removed, "
                f"{fmt_n(ex.get('eligible_active_censored', 0))} still listed). "
                f"Excluded: {fmt_n(ex.get('left_censored_initial_inventory', 0))} from the initial inventory, "
                f"{fmt_n(ex.get('expired_before_tracking', 0))} removed before removal tracking began, "
                f"{fmt_n(ex.get('invalid_or_missing_dates', 0))} with missing or invalid dates.")
        if window.get('duration_definition'):
            lines.append(f"**Duration.** {window['duration_definition']}")
        lines.append(
            "**Typical listing.** The median comes from a Kaplan-Meier estimate: the time by which half of "
            "coffees have left, counting coffees that are still listed as \"at least this long\" instead of "
            "ignoring them (ignoring them would make listings look shorter than they are). "
            "If fewer than half have left, the median is \"not reached\".")
        inactive = (data.get('by_seller') or {}).get('excluded_inactive_sellers') or []
        if inactive or window.get('seller_inactive_days'):
            days = window.get('seller_inactive_days', 60)
            lines.append(
                f"**Closed sellers.** Sellers with no listings seen in the last {days} days are treated as "
                "no longer tracked and are left out of seller comparisons"
                + (f" ({', '.join(inactive)})." if inactive else "."))
        test = (data.get('by_seller') or {}).get('test')
        lines.append(
            "**Comparing groups.** Each group is compared with all other eligible coffees using a log-rank test "
            "(it compares how fast coffees leave, including those still listed). The p-values are adjusted for the "
            "number of groups compared (Benjamini-Hochberg q-value; q < 0.05 is treated as a real difference)."
            + (f" Test used: {test}." if test else ""))
        for line in lines:
            st.markdown(line)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def _render_lifespan_overview(data: Dict[str, Any]):
    """Overall lifespan: counts, Kaplan-Meier median / quartiles, survival curve"""
    st.subheader("Overall Lifespan")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Coffees analyzed", fmt_n((data.get('total_expired') or 0) + (data.get('total_active') or 0)))
    with col2:
        st.metric("Removed", fmt_n(data.get('total_expired', 'N/A')))
    with col3:
        st.metric("Still listed", fmt_n(data.get('total_active', 'N/A')),
                  help="Still-listed coffees count as 'at least this long'.")
    with col4:
        median = data.get('median_days')
        st.metric("Typical listing (median)", NOT_REACHED if median is None else f"{median:.0f} days")

    q25, q75 = data.get('q25_days'), data.get('q75_days')
    if median is None:
        st.caption(NOT_REACHED_NOTE)
    else:
        text = f"Half of coffees are removed within {median:.0f} days."
        if q25 is not None:
            text += f" A quarter are gone within {fmt_days(q25)} days"
            text += f", and three quarters within {fmt_days(q75)} days." if q75 is not None else "."
        st.caption(text)

    curve = data.get('km_curve') or {}
    times, surv = curve.get('times'), curve.get('survival')
    if times and surv:
        fig = go.Figure(go.Scatter(x=[0] + list(times), y=[1.0] + list(surv), mode='lines',
                                   line_shape='hv', line=dict(color='#8B4513')))
        fig.update_layout(
            title="Share of coffees still listed after N days",
            xaxis_title="Days since first listed",
            yaxis_title="Share still listed",
            yaxis_tickformat='.0%', yaxis_range=[0, 1.02],
        )
        st.plotly_chart(fig, use_container_width=True)


def _group_frame(rows: List[Dict[str, Any]], name_key: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        df['_name'] = df[name_key]
    return df


def _group_table(df: pd.DataFrame, name_label: str) -> pd.DataFrame:
    """Display table for survival groups."""
    def opt(col):
        return df[col] if col in df.columns else pd.Series([None] * len(df), index=df.index)

    def num(x):
        return None if x is None or pd.isna(x) else x

    return pd.DataFrame({
        name_label: df['_name'],
        'Coffees': df['count'].map(fmt_n),
        'Removed': opt('events').map(lambda x: '' if num(x) is None else fmt_n(x)),
        'Still listed': opt('censored').map(lambda x: '' if num(x) is None else fmt_n(x)),
        'Median (days)': opt('median_lifespan').map(lambda x: fmt_days(num(x))),
        'Middle half (days)': [
            f"{fmt_days(num(a))} to {fmt_days(num(b))}"
            for a, b in zip(opt('q25_lifespan'), opt('q75_lifespan'))
        ],
        'q-value': opt('q_value').map(lambda q: fmt_q(num(q))),
        'Different from others?': opt('is_significant').map(lambda v: "Yes" if as_bool(v) else "No"),
    })


def _median_chart(df: pd.DataFrame, name_label: str, title: str):
    """Median lifespan bar chart with middle-half (q25-q75) error bars; groups without a median are skipped."""
    if 'median_lifespan' not in df.columns:
        return
    plot = df[df['median_lifespan'].notna()].copy()
    if len(plot) < 2:
        return
    kwargs = {}
    if 'q25_lifespan' in plot and 'q75_lifespan' in plot:
        plot['_up'] = (plot['q75_lifespan'] - plot['median_lifespan']).clip(lower=0).fillna(0)
        plot['_down'] = (plot['median_lifespan'] - plot['q25_lifespan']).clip(lower=0).fillna(0)
        kwargs = {'error_y': '_up', 'error_y_minus': '_down'}
    fig = px.bar(
        plot.sort_values('median_lifespan'), x='_name', y='median_lifespan', **kwargs,
        title=title,
        labels={'median_lifespan': 'Median lifespan (days)', '_name': name_label},
        color='count', color_continuous_scale='Viridis',
    )
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    n_missing = int(df['median_lifespan'].isna().sum())
    if n_missing:
        st.caption(f"{n_missing} group(s) have no median. " + NOT_REACHED_NOTE)


def _render_seller_turnover(seller_data: Dict[str, Any], seller_summary: list, window: Dict[str, Any]):
    """Seller-level lifespan (Kaplan-Meier) and turnover"""
    st.subheader("Turnover by Seller")

    sellers = (seller_data or {}).get('sellers', [])
    if sellers:
        df = _group_frame(sellers, 'seller')
        _median_chart(df, 'Seller', "Median Coffee Lifespan by Seller (bars show the middle half)")
        st.dataframe(_group_table(df, 'Seller'), hide_index=True)
        st.caption("Each seller is compared with all other eligible coffees (log-rank test, "
                   "q-values adjusted across sellers).")

    inactive = (seller_data or {}).get('excluded_inactive_sellers') or []
    if inactive:
        days = window.get('seller_inactive_days', 60)
        st.caption(f"Left out: sellers with no listings seen in the last {days} days "
                   f"(no longer tracked): {', '.join(inactive)}.")

    if seller_summary:
        summary = pd.DataFrame(seller_summary)
        if not summary.empty:
            show = summary.rename(columns={
                'seller': 'Seller', 'total_coffees': 'Total', 'active': 'Active',
                'expired': 'Removed', 'turnover_rate': 'Share removed',
                'unique_countries': 'Countries',
            })
            keep = [c for c in ['Seller', 'Total', 'Active', 'Removed', 'Share removed', 'Countries']
                    if c in show.columns]
            with st.expander("Coffees listed and removed per seller"):
                config = ({'Share removed': st.column_config.ProgressColumn(
                    "Share removed", min_value=0, max_value=1, format="%.0f%%")}
                    if 'Share removed' in keep else None)
                st.dataframe(show[keep], column_config=config, hide_index=True)


def _render_group_survival(data: Dict[str, Any], rows_key: str, name_key: str, title: str, name_label: str):
    """Lifespan by origin country / process method"""
    if not data or not data.get('has_data'):
        return
    st.subheader(title)
    rows = data.get(rows_key, [])
    if not rows:
        return
    df = _group_frame(rows, name_key)
    _median_chart(df, name_label, f"Median Coffee Lifespan by {name_label} (bars show the middle half)")
    st.dataframe(_group_table(df, name_label), hide_index=True)
    if data.get('test'):
        st.caption("Each group is compared with all other eligible coffees "
                   "(log-rank test, q-values adjusted across groups).")


def _render_lifespan_by_price(data: Dict[str, Any]):
    """Lifespan vs price (price quartiles, Kaplan-Meier)"""
    if not data or not data.get('has_data'):
        return

    st.subheader("Lifespan vs Price")

    quartiles = data.get('quartile_stats', [])
    if quartiles:
        q_df = pd.DataFrame(quartiles)
        if not q_df.empty and 'median_lifespan' in q_df.columns:
            plot = q_df[q_df['median_lifespan'].notna()]
            if not plot.empty:
                fig = px.bar(
                    plot, x='quartile', y='median_lifespan',
                    title="Median Lifespan by Price Quartile",
                    labels={'median_lifespan': 'Median lifespan (days)', 'quartile': 'Price quartile'},
                    text=plot['count'].map(lambda x: f"n={fmt_n(x)}"),
                )
                st.plotly_chart(fig, use_container_width=True)
            if q_df['median_lifespan'].isna().any():
                st.caption("Quartiles without a bar: " + NOT_REACHED_NOTE)

    p_val = data.get('p_value')
    if p_val is not None:
        sig = as_bool(data.get('is_significant'))
        verdict = "differs" if sig else "does not clearly differ"
        st.caption(f"Lifespan {verdict} between the cheapest and most expensive quarter of coffees "
                   f"({fmt_p(p_val)}; {data.get('test', 'log-rank test')}). "
                   "Price is the best per-lb price offered (usually the largest bag).")

    corr = data.get('correlation')
    scatter = data.get('scatter_data', {})
    if corr is not None or (scatter.get('prices') and scatter.get('lifespans')):
        with st.expander("Price vs lifespan: removed coffees only"):
            if corr is not None:
                st.metric("Spearman correlation", f"{corr:.3f}")
                st.caption(data.get('correlation_note',
                                    "Spearman on removed coffees only (biased toward short lifespans)")
                           + ". Treat it as a rough picture only.")
            if scatter.get('prices') and scatter.get('lifespans'):
                fig = px.scatter(
                    x=scatter['prices'], y=scatter['lifespans'], opacity=0.6,
                    title="Price vs lifespan (removed coffees)",
                    labels={'x': 'Price per lb (best per-lb price offered, usually the largest bag)',
                            'y': 'Lifespan (days)'},
                )
                st.plotly_chart(fig, use_container_width=True)


def _render_seasonal_patterns(data: Dict[str, Any]):
    """Monthly new listings and removals"""
    if not data or not data.get('has_data'):
        return

    st.subheader("Seasonal Patterns")

    appearances = data.get('appearances', [])
    disappearances = data.get('disappearances', [])

    if appearances:
        app_df = pd.DataFrame(appearances)
        if not app_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=app_df['month'], y=app_df['count'],
                                 name='New Listings', marker_color='#2E8B57'))
            if disappearances:
                dis_df = pd.DataFrame(disappearances)
                if not dis_df.empty:
                    fig.add_trace(go.Bar(x=dis_df['month'], y=dis_df['count'],
                                         name='Removed', marker_color='#CD5C5C'))
            fig.update_layout(title="Monthly Coffee Listings: New vs Removed", xaxis_title="Month",
                              yaxis_title="Count", barmode='group', xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    notes = []
    if data.get('excluded_first_scrape_month'):
        notes.append(f"The first scrape month ({data['excluded_first_scrape_month']}) is left out of new "
                     "listings, because it mostly shows the existing inventory rather than new arrivals.")
    if data.get('disappearances_from'):
        notes.append(f"Removals are counted from {data['disappearances_from']}, when removal tracking began.")
    if notes:
        st.caption(" ".join(notes))
