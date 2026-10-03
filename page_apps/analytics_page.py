"""
Coffee Flavor Analytics Dashboard

Interactive analysis of what makes coffee from different origins and sellers distinctive,
computed over individual coffees.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List, Any

from analytics.frontend.cached_data_loader import (
    load_overview_data, get_cache_status, show_cache_status_widget,
    load_data_completeness, load_distinctiveness_meta,
)
from page_apps.analytics.config import SHOW_TURNOVER_TAB   # Turnover tab hidden until Phase 4
from page_apps.analytics.flavor_origin_tab import render_flavor_origin_tab
from page_apps.analytics.rankings_tab import render_rankings_tab
from page_apps.analytics.turnover_tab import render_turnover_tab
from page_apps.analytics.price_tab import render_price_tab
from page_apps.analytics.process_varietal_tab import render_process_varietal_tab
from page_apps.analytics.cross_feature_tab import render_cross_feature_tab
from page_apps.analytics.common import support_text


def main():
    """Main analytics dashboard page"""
    st.title("🔬 Coffee Flavor Analytics")
    st.markdown("Discover how coffee flavors relate to their origins through statistical and computational analysis.")
    st.info("Note that this is purely experimental and undergoing a lot of iteration. Some information is fun and useful. Some is not. When we're happy with our flavor analytics, we'll make a clear post explaining the information and simplifying the presentation.")
    # Show cache status in sidebar
    show_cache_status_widget()
    
    # Check cache status
    cache_status = get_cache_status()
    
    if not cache_status['exists']:
        st.error("📊 Analytics data is temporarily unavailable. Please try again later.")
        return

    if not cache_status['fresh']:
        st.warning("⚠️ Analytics data may be out of date (generated more than 24 hours ago).")

    # Create tabs - restructured layout
    tab_specs = [
        ("📊 Overview", render_overview_tab),
        ("🫘 Flavor & Origin", render_flavor_origin_tab),
        ("🏷️ Process & Varietal", render_process_varietal_tab),
        ("💰 Price Analysis", render_price_tab),
        ("📅 Seller Turnover", render_turnover_tab),
        ("🔗 Cross-Feature", render_cross_feature_tab),
        ("🏆 Rankings", render_rankings_tab),
    ]
    if not SHOW_TURNOVER_TAB:
        tab_specs = [t for t in tab_specs if t[1] is not render_turnover_tab]

    for tab, (_, render_fn) in zip(st.tabs([label for label, _ in tab_specs]), tab_specs):
        with tab:
            render_fn()


def render_overview_tab():
    """Render the overview tab with key insights and summary visualizations"""
    st.header("Dataset Overview")
    
    # Add explanation
    st.markdown("""
    **What you'll see here:** A high-level view of our coffee dataset and key discoveries about flavor patterns.
    
    **How to interpret:** The metrics show the scope of our analysis, while geographic distribution reveals 
    which countries and regions contribute most to our understanding of coffee flavors. This gives you 
    context for the more detailed analyses in other tabs.
    """)
    
    # Load overview data from cache
    overview_data = load_overview_data()
    
    if not overview_data:
        st.warning("Overview data not available in cache")
        return
    
    basic_stats = overview_data.get('dataset_stats', {})
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Coffees", 
            value=basic_stats.get('total_coffees', 'N/A')
        )
    with col2:
        st.metric(
            "Countries Analyzed", 
            value=basic_stats.get('countries_analyzed', 'N/A')
        )
    with col3:
        st.metric(
            "Regions Analyzed", 
            value=basic_stats.get('regions_analyzed', 'N/A')
        )
    with col4:
        st.metric(
            "Sellers",
            value=basic_stats.get('sellers_analyzed', 'N/A')
        )
    
    # Data completeness section
    completeness = load_data_completeness()
    if completeness and completeness.get('has_data'):
        st.subheader("📋 Data Completeness")
        fields = completeness.get('fields', {})
        cols = st.columns(4)
        field_labels = [
            ('flavor', 'Flavor Notes'),
            ('country', 'Country'),
            ('process_type', 'Process Method'),
            ('varietal', 'Varietal'),
        ]
        for i, (key, label) in enumerate(field_labels):
            with cols[i]:
                rate = fields.get(key, {}).get('rate', 0)
                count = fields.get(key, {}).get('count', 0)
                st.metric(label, f"{rate:.0%}", help=f"{count} coffees")

        cols2 = st.columns(4)
        field_labels_2 = [
            ('price', 'Price'),
            ('region', 'Region'),
            ('dates', 'Temporal Data'),
        ]
        for i, (key, label) in enumerate(field_labels_2):
            with cols2[i]:
                rate = fields.get(key, {}).get('rate', 0)
                count = fields.get(key, {}).get('count', 0)
                st.metric(label, f"{rate:.0%}", help=f"{count} coffees")

    # Key findings section
    st.subheader("🎯 Key Discoveries")

    # Prefer the per-coffee distinctiveness findings; fall back to legacy overview findings
    meta = load_distinctiveness_meta() or {}
    key_findings = meta.get('key_findings') or overview_data.get('key_findings', [])
    if key_findings:
        for i, finding in enumerate(key_findings[:5]):
            title, body_lines, metrics = format_key_finding(finding)
            with st.expander(f"Discovery {i+1}: {title}"):
                for line in body_lines:
                    st.markdown(line)
                for metric_name, metric_value in metrics.items():
                    st.metric(metric_name, metric_value)
    else:
        # Generate fallback key findings from available data
        fallback_findings = _generate_fallback_findings(overview_data, basic_stats)
        if fallback_findings:
            for line in fallback_findings:
                st.markdown(f"- {line}")
        else:
            st.info("Key discoveries will appear here as analysis completes.")
    
    # Geographic distribution
    st.subheader("🌍 Geographic Distribution")
    
    geographic_data = overview_data.get('geographic_data', [])
    
    if geographic_data:
        # Convert to DataFrame for visualization
        country_df = pd.DataFrame(geographic_data)
        country_df = country_df.rename(columns={
            'country': 'Country',
            'total_coffees': 'Total Coffees',
            'flavor_families': 'Flavor Families',
            'sellers': 'Sellers',
            'regions': 'Regions'
        })
        
        if not country_df.empty:
            # Bar chart of coffee counts by country
            fig_bar = px.bar(
                country_df.sort_values('Total Coffees', ascending=False).head(15),
                x='Country',
                y='Total Coffees',
                title="Coffee Count by Country (Top 15)"
            )
            fig_bar.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Summary table
            st.dataframe(
                country_df.sort_values('Total Coffees', ascending=False),
                column_config={
                    "Total Coffees": st.column_config.NumberColumn("Total Coffees"),
                    "Flavor Families": st.column_config.NumberColumn("Flavor Families"),
                    "Sellers": st.column_config.NumberColumn("Sellers"),
                    "Regions": st.column_config.NumberColumn("Regions")
                },
                hide_index=True
            )
    else:
        st.info("Geographic data will appear here once analysis is complete.")


# Helper functions


def format_key_finding(finding: Dict[str, Any]):
    """Map a key finding to (title, markdown lines, metrics).

    The generator writes {'finding': str, 'examples': [...]} plus, for distinctiveness
    findings, the supporting numbers (coffees, sellers, top seller share). Older shapes
    with title/description/metrics are still accepted.
    """
    title = finding.get('title') or finding.get('finding') or 'Key Finding'
    lines = []
    if finding.get('description'):
        lines.append(finding['description'])
    examples = finding.get('examples') or []
    if examples:
        lines.append("**Examples:**")
        for ex in examples:
            if isinstance(ex, dict):
                methods = ', '.join(ex.get('methods', []))
                suffix = f" ({methods})" if methods else ""
                lines.append(f"- {ex.get('unit', '?')}: {ex.get('flavor', '?')}{suffix}")
            else:
                lines.append(f"- {ex}")
    sup = support_text(finding)
    if sup:
        lines.append(f"Based on {finding.get('a', '?')} coffees at the {finding.get('level', 'genus')} level, "
                     f"{sup}, so it isn't just one seller's tasting vocabulary.")
    return title, lines, finding.get('metrics') or {}


def _generate_fallback_findings(overview_data: Dict, basic_stats: Dict) -> List[str]:
    """Summary bullet points from dataset counts when there are no distinctiveness findings"""
    findings = []

    total = basic_stats.get('total_coffees', 0)
    countries = basic_stats.get('countries_analyzed', 0)
    regions = basic_stats.get('regions_analyzed', 0)
    if total:
        findings.append(
            f"**{total} coffees** analyzed across **{countries} countries** and **{regions} regions**."
        )

    geo = overview_data.get('geographic_data', [])
    if geo:
        sorted_geo = sorted(geo, key=lambda x: x.get('total_coffees', 0), reverse=True)
        top_country = sorted_geo[0]
        findings.append(
            f"**{top_country['country']}** has the most coffees ({top_country['total_coffees']})."
        )

    return findings


if __name__ == "__main__":
    main()
