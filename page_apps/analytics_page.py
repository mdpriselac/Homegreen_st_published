"""
Coffee Flavor Analytics Dashboard

Interactive analysis of coffee flavor distinctiveness by origin using multiple analytical methods.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
from collections import defaultdict

from analytics.frontend.cached_data_loader import (
    load_overview_data, load_unit_profile, get_available_units,
    load_flavor_hierarchies, load_rankings_data, get_cache_status,
    show_cache_status_widget, get_cached_data_loader, clear_all_caches,
    load_turnover_data, load_price_analysis_data, load_cooccurrence_data,
    load_data_completeness
)
from analytics.frontend.data_cache_generator import FrontendDataCacheGenerator
from page_apps.analytics.turnover_tab import render_turnover_tab
from page_apps.analytics.price_tab import render_price_tab
from page_apps.analytics.process_varietal_tab import render_process_varietal_tab
from page_apps.analytics.cross_feature_tab import render_cross_feature_tab


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
        st.error("📊 Analytics data cache not found")
        st.markdown("The analytics system needs to generate a data cache for fast loading.")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚀 Generate Cache", type="primary"):
                generate_cache_with_progress()
                st.rerun()
        with col2:
            st.warning("⏱️ This will take 5-10 minutes to complete")
        return
    
    # Cache management buttons
    if not cache_status['fresh']:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🔄 Update Cache"):
                generate_cache_with_progress()
                st.rerun()
        
        with col2:
            st.warning("⚠️ Data cache may be stale (older than 24 hours)")
    
    # Create tabs - restructured layout
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview",
        "🫘 Flavor & Origin",
        "🏷️ Process & Varietal",
        "💰 Price Analysis",
        "📅 Seller Turnover",
        "🔗 Cross-Feature",
        "🏆 Rankings"
    ])

    with tab1:
        render_overview_tab()

    with tab2:
        render_flavor_origin_tab()

    with tab3:
        render_process_varietal_tab()

    with tab4:
        render_price_tab()

    with tab5:
        render_turnover_tab()

    with tab6:
        render_cross_feature_tab()

    with tab7:
        render_rankings_tab()


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
            "Flavor Parse Rate", 
            value=f"{basic_stats.get('flavor_parse_rate', 0):.1%}" if basic_stats.get('flavor_parse_rate') else 'N/A'
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

    key_findings = overview_data.get('key_findings', [])
    if key_findings:
        for i, finding in enumerate(key_findings[:5]):
            with st.expander(f"Discovery {i+1}: {finding.get('title', 'Key Finding')}"):
                st.markdown(finding.get('description', ''))
                if 'metrics' in finding:
                    for metric_name, metric_value in finding['metrics'].items():
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


def render_flavor_origin_tab():
    """Consolidated tab: Explore + By Flavor + Compare in one tab with sub-sections"""
    st.header("Flavor & Origin Analysis")

    section = st.radio(
        "Analysis mode:",
        ["Explore by Origin", "Explore by Flavor", "Compare Origins"],
        horizontal=True,
        key="flavor_origin_section"
    )

    if section == "Explore by Origin":
        render_explore_section()
    elif section == "Explore by Flavor":
        render_flavor_section()
    elif section == "Compare Origins":
        render_compare_section()


def render_explore_section():
    """Render the explore section for detailed unit analysis (was render_explore_tab)"""
    
    # Add explanation
    st.markdown("""
    **What you'll see here:** Deep-dive analysis for individual countries, regions, or sellers showing their unique flavor characteristics.

    **How to interpret:**
    - **Statistical Significance**: Uses Fisher's exact test (or chi-square for larger samples) to find flavors that
      appear significantly more often in this origin than globally. P-values are corrected for multiple comparisons
      using the Benjamini-Hochberg method (reduces false positives when testing many flavors at once).
    - **Odds Ratio**: How many times more likely a flavor is in this origin (e.g., 3.2x means this origin's coffees
      are 3.2 times more likely to have this flavor than coffees from other origins).
    - **TF-IDF Distinctiveness** ("uniqueness score"): Borrowed from text analysis — it measures how distinctive a
      flavor is by weighing both frequency here AND rarity elsewhere. A flavor that's common everywhere scores low;
      a flavor that's common here but rare elsewhere scores high.
    - **Consensus Findings**: Flavors confirmed by multiple independent methods, giving higher confidence the
      pattern is real and not a statistical artifact.

    Use this to understand what makes each origin's coffee unique.
    """)
    
    # Unit selection interface
    col1, col2 = st.columns(2)
    
    with col1:
        unit_type = st.radio("Explore by:", ["Country", "Region", "Seller"])
    
    # Get available units based on type
    available_units = get_available_units(unit_type.lower())
    
    with col2:
        if available_units:
            selected_unit = st.selectbox(f"Select {unit_type}:", available_units)
        else:
            st.warning(f"No {unit_type.lower()} data available")
            return
    
    if selected_unit:
        # Get comprehensive profile for selected unit
        try:
            profile = load_unit_profile(selected_unit, unit_type.lower())
            
            if profile:
                display_unit_profile(profile)
            else:
                st.warning(f"No analysis results found for {selected_unit}")
                
        except Exception as e:
            st.error(f"Error loading data for {selected_unit}: {str(e)}")


def render_flavor_section():
    """Render the by-flavor section for flavor-first exploration"""
    # Add explanation
    st.markdown("""
    **What you'll see here:** Start with a specific flavor and discover which origins express it most distinctively.
    
    **How to interpret:**
    - **Flavor Hierarchy**: Choose from Family (broad, like "Fruity"), Genus (more specific, like "Berry"), or Species (most specific, like "Blueberry")
    - **Most Distinctive Regions**: Places where this flavor is most characteristic - high TF-IDF scores mean this flavor really defines that origin
    - **Co-occurring Flavors**: Other flavors that commonly appear alongside your selected flavor in the same coffees
    - **Statistical Significance**: Origins where this flavor appears significantly more often than expected by chance
    
    Perfect for exploring "Where can I find the best [specific flavor]?" questions.
    """)
    
    # Get flavor hierarchies from cache
    flavor_data = load_flavor_hierarchies()
    
    if not flavor_data:
        st.warning("Flavor hierarchy data not available")
        return
    
    # Flavor selection interface
    col1, col2, col3 = st.columns(3)
    
    with col1:
        family_options = ["All"] + sorted(flavor_data['families'])
        selected_family = st.selectbox("Flavor Family:", family_options)
    
    with col2:
        if selected_family != "All" and selected_family in flavor_data['genera_by_family']:
            genus_options = ["All"] + sorted(flavor_data['genera_by_family'][selected_family])
        else:
            genus_options = ["All"] + sorted(flavor_data['all_genera'])
        selected_genus = st.selectbox("Flavor Genus:", genus_options)
    
    with col3:
        if selected_genus != "All" and selected_genus in flavor_data['species_by_genus']:
            species_options = ["All"] + sorted(flavor_data['species_by_genus'][selected_genus])
        else:
            species_options = ["All"]
        selected_species = st.selectbox("Flavor Species:", species_options)
    
    # Display results for selected flavor
    if selected_family != "All" or selected_genus != "All" or selected_species != "All":
        display_flavor_analysis(selected_family, selected_genus, selected_species)


def render_compare_section():
    """Render the comparison section"""
    # Add explanation
    st.markdown("""
    **What you'll see here:** Side-by-side comparison of multiple origins to understand their similarities and differences.
    
    **How to interpret:**
    - **Total Coffees**: Sample size - larger numbers give more reliable results
    - **Distinctive Flavors**: Number of flavors that are statistically significant for this origin
    - **Top Distinctive Flavor**: The single most characteristic flavor for each origin with its TF-IDF score
    - **TF-IDF Scores**: Higher scores (closer to 1.0) mean the flavor more strongly defines that origin
    
    Use this to answer questions like "How does Colombian coffee differ from Ethiopian?" or "Which seller has the most unique profile?"
    """)
    
    comparison_type = st.radio("Compare:", ["Countries", "Regions", "Sellers"])
    
    # Get available entities - convert plural to singular properly
    entity_type_mapping = {
        'countries': 'country',
        'regions': 'region', 
        'sellers': 'seller'
    }
    entity_type = entity_type_mapping.get(comparison_type.lower(), comparison_type.lower().rstrip('s'))
    available_entities = get_available_units(entity_type)
    
    if available_entities:
        selected_entities = st.multiselect(
            f"Select {comparison_type.lower()} to compare (max 4):",
            available_entities,
            max_selections=4
        )
        
        if len(selected_entities) >= 2:
            display_comparison(selected_entities, entity_type)
        else:
            st.info("Please select at least 2 entities to compare.")
    else:
        st.warning(f"No {comparison_type.lower()} data available for comparison.")


def render_rankings_tab():
    """Render the rankings tab"""
    st.header("Rankings & Leaderboards")

    st.markdown("""
    **What you'll see here:** Rankings of origins based on different analytical criteria.

    **How to interpret:**
    - **Most Distinctive Overall**: Score = number of flavors confirmed by multiple analysis methods. Higher = more uniquely identifiable flavor identity.
    - **Most Specialized**: Score = concentration index (0-1). Higher values mean the origin focuses on a narrow set of specific flavors rather than a broad mix.
    - **Most Diverse**: Score = Shannon entropy-based diversity. Higher values mean the origin produces a wider variety of flavor profiles.
    - **Best Value / Highest Priced**: Ranked by median price per lb (minimum 5 coffees).
    - **Fastest Moving / Longest Lasting**: Ranked by median listing lifespan in days (expired coffees only, minimum 5).
    """)

    col1, col2 = st.columns(2)

    with col1:
        ranking_category = st.selectbox(
            "Ranking Category:",
            [
                "Most Distinctive Overall",
                "Most Specialized",
                "Most Diverse",
                "Highest Volume",
                "Best Value (Lowest Price)",
                "Highest Priced",
                "Fastest Moving (Shortest Lifespan)",
                "Longest Lasting",
            ],
        )

    with col2:
        min_coffees = st.slider("Minimum Coffees:", 1, 100, 10)

    rankings_data = generate_rankings(ranking_category, "All", min_coffees)

    if not rankings_data.empty:
        display_rankings(rankings_data, ranking_category)
    else:
        st.info("No data meets the selected criteria.")


def generate_cache_with_progress():
    """Generate frontend cache with progress display"""
    try:
        with st.spinner("🔄 Generating analytics cache..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create generator
            status_text.text("Initializing cache generator...")
            generator = FrontendDataCacheGenerator()
            progress_bar.progress(10)
            
            # Generate cache
            status_text.text("Running full analysis and generating cache...")
            progress_bar.progress(20)
            
            cache_data = generator.generate_full_cache()
            progress_bar.progress(90)
            
            # Clear all caches to force reload of new data
            status_text.text("Clearing old cache...")
            clear_all_caches()
            progress_bar.progress(100)
            
            status_text.text("✅ Cache generation completed!")
            st.success(f"🎉 Successfully generated cache with {len(cache_data['unit_profiles'])} unit profiles!")
            
    except Exception as e:
        st.error(f"❌ Cache generation failed: {str(e)}")
        st.info("Please check the database connection and analytics system configuration.")


# Helper functions


def _generate_fallback_findings(overview_data: Dict, basic_stats: Dict) -> List[str]:
    """Generate summary bullet points from available data when key_findings is empty"""
    findings = []

    total = basic_stats.get('total_coffees', 0)
    countries = basic_stats.get('countries_analyzed', 0)
    if total:
        findings.append(
            f"**{total} coffees** analyzed across **{countries} countries**."
        )

    geo = overview_data.get('geographic_data', [])
    if geo:
        sorted_geo = sorted(geo, key=lambda x: x.get('total_coffees', 0), reverse=True)
        top_country = sorted_geo[0]
        findings.append(
            f"**{top_country['country']}** has the most coffees ({top_country['total_coffees']}) "
            f"with {top_country.get('flavor_families', '?')} flavor families represented."
        )

    top_units = overview_data.get('top_units', [])
    if top_units:
        top = top_units[0]
        name = top.get('unit') or top.get('entity_name', '?')
        findings.append(
            f"**{name}** is the most distinctive origin in the dataset."
        )

    parse_rate = basic_stats.get('flavor_parse_rate', 0)
    if parse_rate:
        findings.append(
            f"**{parse_rate:.0%}** of coffees have parseable flavor notes for analysis."
        )

    return findings


def extract_basic_stats(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract basic statistics from analysis results"""
    stats = {}
    
    try:
        # Try to get from summary first
        if 'summary' in all_results and 'dataset_overview' in all_results['summary']:
            return all_results['summary']['dataset_overview']
        
        # Fallback: extract from raw data
        if 'data' in all_results:
            data = all_results['data']
            
            # Extract from raw dataframe
            if 'raw_merged_df' in data:
                df = data['raw_merged_df']
                stats['total_coffees'] = len(df)
                
                if 'country_final' in df.columns:
                    stats['countries_analyzed'] = df['country_final'].nunique()
                if 'subregion_final' in df.columns:
                    stats['regions_analyzed'] = df['subregion_final'].nunique()
                if 'has_flavors' in df.columns:
                    stats['flavor_parse_rate'] = df['has_flavors'].mean()
            
            # Extract from aggregated data
            if 'country_aggregated' in data:
                stats['countries_analyzed'] = len(data['country_aggregated'])
            if 'region_aggregated' in data:
                stats['regions_analyzed'] = len(data['region_aggregated'])
                
    except Exception as e:
        st.warning(f"Could not extract all statistics: {e}")
    
    return stats


def get_most_distinctive_regions_for_flavor(target_flavor: str, taxonomy_level: str) -> pd.DataFrame:
    """Get regions most distinctive for a specific flavor using reverse TF-IDF lookup"""
    try:
        # Load export data which contains TF-IDF summary
        loader = get_cached_data_loader()
        export_data = loader._load_component('export_ready_data')
        tfidf_summary = export_data.get('tfidf_summary', [])
        
        if not tfidf_summary:
            return pd.DataFrame()
        
        # Convert to DataFrame and filter for the target flavor and taxonomy level
        tfidf_df = pd.DataFrame(tfidf_summary)
        
        # Filter for the specific flavor and taxonomy level
        flavor_results = tfidf_df[
            (tfidf_df['flavor'] == target_flavor) & 
            (tfidf_df['taxonomy_level'] == taxonomy_level)
        ]
        
        if flavor_results.empty:
            return pd.DataFrame()
        
        # Sort by TF-IDF score and return top results
        flavor_results = flavor_results.sort_values('tfidf_score', ascending=False)
        
        # Rename columns for display
        flavor_results = flavor_results.rename(columns={'unit_name': 'region_name'})
        
        return flavor_results[['region_name', 'tfidf_score']].head(15)
        
    except Exception as e:
        st.error(f"Error loading distinctive regions: {e}")
        return pd.DataFrame()


def get_cooccurring_flavors_for_flavor(target_flavor: str, taxonomy_level: str) -> List[Dict]:
    """Get flavors that commonly co-occur with the target flavor using cached co-occurrence data"""
    try:
        cooccurrence_data = load_cooccurrence_data()
        if not cooccurrence_data:
            return []

        # Map taxonomy level to cache key
        level_key = f'{taxonomy_level}_level'
        level_data = cooccurrence_data.get(level_key, {})
        top_pairs = level_data.get('top_pairs_by_count', [])

        if not top_pairs:
            return []

        # Find pairs involving the target flavor
        results = []
        for pair in top_pairs:
            f1 = pair.get('flavor_1', '')
            f2 = pair.get('flavor_2', '')
            total = pair.get('total_coffees', 1)

            if f1 == target_flavor:
                results.append({
                    'flavor': f2,
                    'cooccurrence_count': pair.get('cooccurrence_count', 0),
                    'cooccurrence_rate': pair.get('cooccurrence_count', 0) / total if total > 0 else 0,
                })
            elif f2 == target_flavor:
                results.append({
                    'flavor': f1,
                    'cooccurrence_count': pair.get('cooccurrence_count', 0),
                    'cooccurrence_rate': pair.get('cooccurrence_count', 0) / total if total > 0 else 0,
                })

        # Sort by co-occurrence rate
        results.sort(key=lambda x: x['cooccurrence_rate'], reverse=True)
        return results[:10]

    except Exception as e:
        st.error(f"Error loading co-occurring flavors: {e}")
        return []


def get_statistical_results_for_flavor(target_flavor: str, taxonomy_level: str) -> pd.DataFrame:
    """Get statistical significance results for a specific flavor across origins"""
    try:
        # Load export data which contains statistical summary
        loader = get_cached_data_loader()
        export_data = loader._load_component('export_ready_data')
        statistical_summary = export_data.get('statistical_summary', [])
        
        if not statistical_summary:
            return pd.DataFrame()
        
        # Convert to DataFrame and filter
        stats_df = pd.DataFrame(statistical_summary)
        
        # Filter for the specific flavor and taxonomy level
        flavor_stats = stats_df[
            (stats_df['flavor'] == target_flavor) & 
            (stats_df['taxonomy_level'] == taxonomy_level)
        ]
        
        return flavor_stats
        
    except Exception as e:
        st.error(f"Error loading statistical results: {e}")
        return pd.DataFrame()


# Note: get_available_units and load_unit_profile are now imported from cached_data_loader


def display_unit_profile(profile: Dict[str, Any]):
    """Display comprehensive unit profile"""
    st.subheader(f"Analysis for {profile['unit_name']}")

    # Flavor Signature — quick summary of most distinctive flavors
    tfidf_findings = profile.get('tfidf_findings', {})
    family_findings = tfidf_findings.get('family', {})
    top_family_flavors = family_findings.get('top_flavors', [])
    if top_family_flavors:
        sig_flavors = [f['flavor'] for f in top_family_flavors[:5]]
        st.markdown(
            f"**Flavor Signature:** {', '.join(sig_flavors)}  \n"
            f"*These are the flavors that most distinguish {profile['unit_name']} from other origins, "
            f"ranked by uniqueness score (TF-IDF).*"
        )

    # Overview metrics
    overview = profile.get('overview', {})
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Coffees", overview.get('total_coffees', 'N/A'))
    with col2:
        unique_families = overview.get('unique_flavor_families', [])
        st.metric("Unique Flavor Families", len(unique_families) if unique_families else 'N/A')
    with col3:
        flavor_parse_rate = overview.get('flavor_parse_rate', 0)
        st.metric("Flavor Parse Rate", f"{flavor_parse_rate:.1%}" if flavor_parse_rate else 'N/A')

    # Additional overview info
    if overview.get('unique_sellers') or overview.get('unique_subregions'):
        col1, col2 = st.columns(2)
        with col1:
            sellers = overview.get('unique_sellers', [])
            if sellers:
                st.metric("Unique Sellers", len(sellers))
        with col2:
            regions = overview.get('unique_subregions', [])
            if regions:
                st.metric("Subregions", len(regions))
    
    # Statistical findings
    statistical_findings = profile.get('statistical_findings', {})
    if any(statistical_findings.values()):
        st.subheader("📊 Statistical Significance Results")
        
        for level in ['family', 'genus', 'species']:
            findings = statistical_findings.get(level, [])
            if findings:
                with st.expander(f"{level.title()} Level Findings ({len(findings)} significant flavors)"):
                    df = pd.DataFrame(findings)
                    if not df.empty:
                        # Display key columns
                        display_columns = ['flavor', 'odds_ratio', 'p_value_corrected']
                        available_columns = [col for col in display_columns if col in df.columns]
                        if available_columns:
                            st.dataframe(df[available_columns])
                        else:
                            st.dataframe(df)
    
    # TF-IDF findings
    tfidf_findings = profile.get('tfidf_findings', {})
    if any(tfidf_findings.values()):
        st.subheader("🎯 Distinctiveness Analysis (TF-IDF)")
        st.caption(
            "TF-IDF (\"uniqueness score\") measures how distinctive a flavor is to this origin. "
            "Higher scores mean the flavor is both common here AND rare elsewhere — "
            "the combination that makes a flavor truly characteristic of a place."
        )

        for level in ['family', 'genus', 'species']:
            level_data = tfidf_findings.get(level, {})
            top_flavors = level_data.get('top_flavors', [])
            if top_flavors:
                with st.expander(f"{level.title()} Level Distinctiveness ({len(top_flavors)} top flavors)"):
                    df = pd.DataFrame(top_flavors)
                    if not df.empty:
                        display = df[['flavor', 'tfidf_score']].head(10).copy()
                        display = display.rename(columns={
                            'flavor': 'Flavor',
                            'tfidf_score': 'Uniqueness Score',
                        })
                        st.dataframe(display, hide_index=True)
    
    # Consensus findings
    consensus_findings = profile.get('consensus_findings', {})
    strong_consensus = consensus_findings.get('strong', [])
    moderate_consensus = consensus_findings.get('moderate', [])

    if strong_consensus or moderate_consensus:
        st.subheader("🤝 Cross-Method Consensus")
        st.caption(
            "Three independent methods are used to identify distinctive flavors: "
            "**(1) Statistical testing** (are flavors significantly more common here?), "
            "**(2) TF-IDF uniqueness** (how distinctive is this flavor to this origin?), and "
            "**(3) Hierarchical cascade** (does the pattern hold across family → genus → species levels?). "
            "When multiple methods agree, we have higher confidence the finding is real."
        )

        if strong_consensus:
            st.write("**Strong Consensus (all 3 methods agree):**")
            for finding in strong_consensus[:5]:
                flavor = finding.get('flavor', 'Unknown')
                methods = finding.get('methods', [])
                st.write(f"- {flavor} (verified by: {', '.join(methods)})")

        if moderate_consensus:
            st.write("**Moderate Consensus (2 of 3 methods agree):**")
            for finding in moderate_consensus[:5]:
                flavor = finding.get('flavor', 'Unknown')
                methods = finding.get('methods', [])
                st.write(f"- {flavor} (verified by: {', '.join(methods)})")
    
    # Recommendations
    recommendations = profile.get('recommendations', [])
    if recommendations:
        st.subheader("💡 Recommendations")
        for rec in recommendations:
            st.write(rec)


def display_flavor_analysis(family: str, genus: str, species: str):
    """Display analysis results for selected flavor"""
    st.subheader(f"Analysis for Selected Flavor")
    
    flavor_desc = []
    if family != "All":
        flavor_desc.append(f"Family: {family}")
    if genus != "All":
        flavor_desc.append(f"Genus: {genus}")
    if species != "All":
        flavor_desc.append(f"Species: {species}")
    
    st.write(f"**Selected Flavor:** {' → '.join(flavor_desc)}")
    
    # Determine the most specific flavor level selected
    if species != "All":
        target_flavor = species
        taxonomy_level = "species"
    elif genus != "All":
        target_flavor = genus
        taxonomy_level = "genus"
    elif family != "All":
        target_flavor = family
        taxonomy_level = "family"
    else:
        st.info("Please select a specific flavor to see analysis results.")
        return
    
    # Get real analysis results for this flavor
    distinctive_regions = get_most_distinctive_regions_for_flavor(target_flavor, taxonomy_level)
    cooccurring_flavors = get_cooccurring_flavors_for_flavor(target_flavor, taxonomy_level)
    statistical_results = get_statistical_results_for_flavor(target_flavor, taxonomy_level)
    
    # Display distinctive regions
    st.subheader("🌍 Most Distinctive Regions for this Flavor")
    if distinctive_regions is not None and not distinctive_regions.empty:
        # Show top 10 regions
        top_regions = distinctive_regions.head(10)
        
        # Create bar chart
        if 'tfidf_score' in top_regions.columns:
            fig = px.bar(
                top_regions,
                x='region_name',
                y='tfidf_score',
                title=f"Regions Most Distinctive for {target_flavor}",
                labels={'tfidf_score': 'TF-IDF Distinctiveness Score', 'region_name': 'Region'}
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.dataframe(
            top_regions[['region_name', 'tfidf_score']].round(3),
            column_config={
                "region_name": "Region",
                "tfidf_score": st.column_config.ProgressColumn(
                    "Distinctiveness Score", 
                    min_value=0, 
                    max_value=top_regions['tfidf_score'].max() if 'tfidf_score' in top_regions.columns else 1
                )
            },
            hide_index=True
        )
    else:
        st.info(f"No distinctive regions found for {target_flavor} at {taxonomy_level} level.")
    
    # Display co-occurring flavors
    st.subheader("🤝 Commonly Co-occurring Flavors")
    if cooccurring_flavors:
        st.write(f"Flavors that frequently appear with **{target_flavor}**:")
        for cooccur in cooccurring_flavors[:10]:  # Show top 10
            st.write(f"• {cooccur['flavor']} (appears together {cooccur['cooccurrence_rate']:.1%} of the time)")
    else:
        st.info(f"Co-occurrence analysis not available for {target_flavor}.")
    
    # Display statistical significance
    if statistical_results is not None and not statistical_results.empty:
        st.subheader("📊 Statistical Significance Across Origins")
        with st.expander(f"Statistical analysis for {target_flavor}"):
            significant_results = statistical_results[statistical_results['is_significant'] == True]
            if not significant_results.empty:
                st.write(f"**{len(significant_results)} origins show statistically significant associations:**")
                st.dataframe(
                    significant_results[['unit_name', 'odds_ratio', 'p_value_corrected']].round(4),
                    column_config={
                        "unit_name": "Origin",
                        "odds_ratio": "Odds Ratio",
                        "p_value_corrected": "P-Value (corrected)"
                    },
                    hide_index=True
                )
            else:
                st.info(f"No statistically significant associations found for {target_flavor}.")
    
    # Summary insights
    st.subheader("💡 Key Insights")
    insights = []
    
    if distinctive_regions is not None and not distinctive_regions.empty:
        top_region = distinctive_regions.iloc[0]
        insights.append(f"**{top_region['region_name']}** is the most distinctive region for {target_flavor} (TF-IDF: {top_region['tfidf_score']:.3f})")
    
    if cooccurring_flavors:
        top_cooccur = cooccurring_flavors[0]
        insights.append(f"**{target_flavor}** most commonly appears with **{top_cooccur['flavor']}** ({top_cooccur['cooccurrence_rate']:.1%} of the time)")
    
    if statistical_results is not None and not statistical_results.empty:
        significant_count = len(statistical_results[statistical_results['is_significant'] == True])
        total_count = len(statistical_results)
        insights.append(f"Shows statistically significant associations in **{significant_count}/{total_count}** analyzed origins")
    
    if insights:
        for insight in insights:
            st.write(f"• {insight}")
    else:
        st.info("Additional analysis may be needed for comprehensive insights about this flavor.")


def display_comparison(entities: List[str], entity_type: str):
    """Display comparison between selected entities"""
    st.subheader("Comparison Results")
    
    # Load profiles for each entity
    entity_profiles = []
    for entity in entities:
        profile = load_unit_profile(entity, entity_type)
        if profile:
            entity_profiles.append(profile)
    
    if not entity_profiles:
        st.warning("No profile data available for selected entities")
        return
    
    # Create columns for each entity
    cols = st.columns(len(entity_profiles))
    
    for i, profile in enumerate(entity_profiles):
        with cols[i]:
            st.write(f"**{profile['unit_name']}**")
            
            overview = profile.get('overview', {})
            st.metric("Total Coffees", overview.get('total_coffees', 'N/A'))
            
            # Count distinctive flavors
            statistical_findings = profile.get('statistical_findings', {})
            total_distinctive = sum(len(findings) for findings in statistical_findings.values())
            st.metric("Distinctive Flavors", total_distinctive if total_distinctive > 0 else 'N/A')
            
            # Show top distinctive flavor
            tfidf_findings = profile.get('tfidf_findings', {})
            family_findings = tfidf_findings.get('family', {})
            top_flavors = family_findings.get('top_flavors', [])
            if top_flavors:
                top_flavor = top_flavors[0]
                st.write(f"**Top Flavor:** {top_flavor['flavor']}")
                st.write(f"TF-IDF: {top_flavor['tfidf_score']:.3f}")
    
    # Comparison summary table
    st.subheader("Comparison Summary")
    
    comparison_data = []
    for profile in entity_profiles:
        overview = profile.get('overview', {})
        statistical_findings = profile.get('statistical_findings', {})
        tfidf_findings = profile.get('tfidf_findings', {})
        
        # Get top family flavor
        top_flavor = "None"
        tfidf_score = 0
        family_findings = tfidf_findings.get('family', {})
        top_flavors = family_findings.get('top_flavors', [])
        if top_flavors:
            top_flavor = top_flavors[0]['flavor']
            tfidf_score = top_flavors[0]['tfidf_score']
        
        comparison_data.append({
            'Entity': profile['unit_name'],
            'Total Coffees': overview.get('total_coffees', 0),
            'Distinctive Flavors': sum(len(findings) for findings in statistical_findings.values()),
            'Top Distinctive Flavor': top_flavor,
            'TF-IDF Score': f"{tfidf_score:.3f}" if tfidf_score > 0 else "N/A"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, hide_index=True)


def generate_rankings(category: str, level: str, min_coffees: int) -> pd.DataFrame:
    """Generate rankings based on selected criteria"""
    rankings_data = load_rankings_data()

    category_map = {
        "Most Distinctive Overall": 'most_distinctive',
        "Most Specialized": 'most_specialized',
        "Most Diverse": 'most_diverse',
        "Best Value (Lowest Price)": 'best_value',
        "Highest Priced": 'highest_priced',
        "Fastest Moving (Shortest Lifespan)": 'fastest_moving',
        "Longest Lasting": 'longest_lasting',
    }

    key = category_map.get(category)
    rankings = rankings_data.get(key, []) if key else []

    if not rankings:
        return pd.DataFrame()

    rankings_df = pd.DataFrame(rankings)

    # Normalize coffee count column name
    coffee_col = 'total_coffees' if 'total_coffees' in rankings_df.columns else 'coffee_count'
    if coffee_col in rankings_df.columns:
        rankings_df = rankings_df[rankings_df[coffee_col] >= min_coffees]

    # Sort: price/lifespan categories ascending, others descending
    ascending_categories = {
        "Best Value (Lowest Price)", "Fastest Moving (Shortest Lifespan)"
    }
    ascending = category in ascending_categories

    if 'score' in rankings_df.columns:
        rankings_df = rankings_df.sort_values('score', ascending=ascending)

    return rankings_df


def display_rankings(rankings_df: pd.DataFrame, category: str):
    """Display rankings with medals and formatting"""
    st.subheader(f"Rankings: {category}")

    if rankings_df.empty:
        st.info("No data available for rankings")
        return

    rankings_display = rankings_df.head(25).copy().reset_index(drop=True)

    # Rank column with medals
    rank_list = []
    for i in range(len(rankings_display)):
        if i == 0:
            rank_list.append("1")
        elif i == 1:
            rank_list.append("2")
        elif i == 2:
            rank_list.append("3")
        else:
            rank_list.append(str(i + 1))
    rankings_display.insert(0, 'Rank', rank_list)

    # Rename common columns for display
    rename_map = {
        'entity_name': 'Name',
        'unit': 'Name',
        'unit_type': 'Type',
        'type': 'Type',
        'total_coffees': 'Coffees',
        'coffee_count': 'Coffees',
    }
    for old, new in rename_map.items():
        if old in rankings_display.columns:
            rankings_display = rankings_display.rename(columns={old: new})

    # Format top_flavors for display if present
    if 'top_flavors' in rankings_display.columns:
        rankings_display['Top Distinctive Flavors'] = rankings_display['top_flavors'].apply(
            lambda x: ', '.join(f for f in x if f) if isinstance(x, list) and x else ''
        )
        rankings_display = rankings_display.drop(columns=['top_flavors'])

    # Category-specific formatting
    column_config = {}
    price_cats = {"Best Value (Lowest Price)", "Highest Priced"}
    lifespan_cats = {"Fastest Moving (Shortest Lifespan)", "Longest Lasting"}

    if category in price_cats:
        if 'score' in rankings_display.columns:
            rankings_display = rankings_display.rename(columns={
                'score': 'Median Price ($/lb)',
                'mean_price': 'Mean Price ($/lb)',
            })
    elif category in lifespan_cats:
        if 'score' in rankings_display.columns:
            rankings_display = rankings_display.rename(columns={
                'score': 'Median Lifespan (days)',
                'mean_lifespan': 'Mean Lifespan (days)',
            })
    else:
        if 'score' in rankings_display.columns:
            column_config["score"] = st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=float(rankings_display['score'].max() or 1)
            )

    st.dataframe(
        rankings_display,
        column_config=column_config,
        hide_index=True,
    )


if __name__ == "__main__":
    main()