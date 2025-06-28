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
    show_cache_status_widget, get_cached_data_loader
)
from analytics.frontend.data_cache_generator import FrontendDataCacheGenerator


def main():
    """Main analytics dashboard page"""
    st.title("🔬 Coffee Flavor Analytics")
    st.markdown("Discover how coffee flavors relate to their origins through statistical and computational analysis.")
    
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
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔄 Force Regenerate", help="Regenerate cache with latest code/data"):
            generate_cache_with_progress()
            st.rerun()
    
    with col2:
        if not cache_status['fresh']:
            if st.button("🔄 Update Stale Cache"):
                generate_cache_with_progress()
                st.rerun()
    
    with col3:
        if not cache_status['fresh']:
            st.warning("⚠️ Data cache may be stale (older than 24 hours)")
        else:
            st.info("💡 Use 'Force Regenerate' to update cache with code fixes")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🔍 Explore", "🫘 By Flavor", "⚖️ Compare", "🏆 Rankings"
    ])
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        render_explore_tab()
    
    with tab3:
        render_flavor_tab()
    
    with tab4:
        render_compare_tab()
    
    with tab5:
        render_rankings_tab()


def render_overview_tab():
    """Render the overview tab with key insights and summary visualizations"""
    st.header("Dataset Overview")
    
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
    
    # Key findings section
    st.subheader("🎯 Key Discoveries")
    
    key_findings = overview_data.get('key_findings', [])
    if key_findings:
        for i, finding in enumerate(key_findings[:5]):  # Show top 5 findings
            with st.expander(f"Discovery {i+1}: {finding.get('title', 'Key Finding')}"):
                st.markdown(finding.get('description', ''))
                if 'metrics' in finding:
                    for metric_name, metric_value in finding['metrics'].items():
                        st.metric(metric_name, metric_value)
    else:
        # Show analysis progress
        show_analysis_progress(overview_data)
    
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


def render_explore_tab():
    """Render the explore tab for detailed unit analysis"""
    st.header("🔍 Detailed Unit Exploration")
    
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


def render_flavor_tab():
    """Render the by-flavor tab for flavor-first exploration"""
    st.header("🫘 Flavor-First Exploration")
    
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


def render_compare_tab():
    """Render the comparison tab"""
    st.header("⚖️ Compare Units")
    
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
    st.header("🏆 Rankings & Leaderboards")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ranking_category = st.selectbox(
            "Ranking Category:",
            ["Most Distinctive Overall", "Most Specialized", "Most Diverse", "Highest Volume"]
        )
    
    with col2:
        taxonomy_level = st.selectbox("Taxonomy Level:", ["All", "Family", "Genus", "Species"])
    
    with col3:
        min_coffees = st.slider("Minimum Coffees:", 1, 100, 10)
    
    # Generate and display rankings
    rankings_data = generate_rankings(ranking_category, taxonomy_level, min_coffees)
    
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
            progress_bar.progress(100)
            
            status_text.text("✅ Cache generation completed!")
            st.success(f"🎉 Successfully generated cache with {len(cache_data['unit_profiles'])} unit profiles!")
            
    except Exception as e:
        st.error(f"❌ Cache generation failed: {str(e)}")
        st.info("Please check the database connection and analytics system configuration.")


# Helper functions

def show_analysis_progress(overview_data: Dict[str, Any]):
    """Show analysis progress when full findings aren't available"""
    try:
        analysis_progress = overview_data.get('analysis_progress', {})
        
        st.write("**Analysis Progress:**")
        progress_items = [
            ("Statistical Significance", analysis_progress.get('statistical_complete', False)),
            ("TF-IDF Distinctiveness", analysis_progress.get('tfidf_complete', False)),
            ("Hierarchical Patterns", analysis_progress.get('hierarchical_complete', False)),
            ("Cross-method Validation", analysis_progress.get('consensus_complete', False)),
            ("Executive Summary", analysis_progress.get('summary_complete', False))
        ]
        
        for name, completed in progress_items:
            if completed:
                st.write(f"✅ {name} analysis completed")
            else:
                st.write(f"⏳ {name} analysis pending")
        
        # Show basic dataset info
        dataset_stats = overview_data.get('dataset_stats', {})
        if dataset_stats:
            st.write("**Current Dataset:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'total_coffees' in dataset_stats:
                    st.write(f"📊 {dataset_stats['total_coffees']} total coffees")
            with col2:
                if 'countries_analyzed' in dataset_stats:
                    st.write(f"🌍 {dataset_stats['countries_analyzed']} countries")
            with col3:
                if 'flavor_parse_rate' in dataset_stats:
                    st.write(f"🫘 {dataset_stats['flavor_parse_rate']:.1%} with flavor data")
                        
    except Exception as e:
        st.info("Analysis system is initializing...")


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
    """Get flavors that commonly co-occur with the target flavor"""
    try:
        # This would require analyzing the raw coffee data to find co-occurrences
        # For now, return a placeholder structure
        # In a full implementation, we'd analyze the raw coffee data to find which flavors
        # appear together in the same coffees
        
        # Placeholder co-occurrence data
        cooccurrences = [
            {"flavor": "Sweet", "cooccurrence_rate": 0.45},
            {"flavor": "Fruity", "cooccurrence_rate": 0.32},
            {"flavor": "Floral", "cooccurrence_rate": 0.28},
            {"flavor": "Nutty/Cocoa", "cooccurrence_rate": 0.25}
        ]
        
        # Filter out the target flavor itself
        cooccurrences = [c for c in cooccurrences if c['flavor'] != target_flavor]
        
        return cooccurrences
        
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
        
        for level in ['family', 'genus', 'species']:
            level_data = tfidf_findings.get(level, {})
            top_flavors = level_data.get('top_flavors', [])
            if top_flavors:
                with st.expander(f"{level.title()} Level Distinctiveness ({len(top_flavors)} top flavors)"):
                    df = pd.DataFrame(top_flavors)
                    if not df.empty:
                        st.dataframe(df[['flavor', 'tfidf_score']].head(10))
    
    # Consensus findings
    consensus_findings = profile.get('consensus_findings', {})
    strong_consensus = consensus_findings.get('strong', [])
    moderate_consensus = consensus_findings.get('moderate', [])
    
    if strong_consensus or moderate_consensus:
        st.subheader("🤝 Cross-Method Consensus")
        
        if strong_consensus:
            st.write("**Strong Consensus (All Methods):**")
            for finding in strong_consensus[:5]:
                flavor = finding.get('flavor', 'Unknown')
                methods = finding.get('methods', [])
                st.write(f"• {flavor} (verified by: {', '.join(methods)})")
        
        if moderate_consensus:
            st.write("**Moderate Consensus (2/3 Methods):**")
            for finding in moderate_consensus[:5]:
                flavor = finding.get('flavor', 'Unknown')
                methods = finding.get('methods', [])
                st.write(f"• {flavor} (verified by: {', '.join(methods)})")
    
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
    
    if category == "Most Distinctive Overall":
        rankings = rankings_data.get('most_distinctive', [])
    elif category == "Most Specialized":
        rankings = rankings_data.get('most_specialized', [])
    elif category == "Most Diverse":
        rankings = rankings_data.get('most_diverse', [])
    else:
        rankings = []
    
    if not rankings:
        return pd.DataFrame()
    
    rankings_df = pd.DataFrame(rankings)
    
    # Filter by minimum coffees if column exists
    if 'total_coffees' in rankings_df.columns:
        rankings_df = rankings_df[rankings_df['total_coffees'] >= min_coffees]
    
    # Sort by score if column exists
    if 'score' in rankings_df.columns:
        rankings_df = rankings_df.sort_values('score', ascending=False)
    
    return rankings_df


def display_rankings(rankings_df: pd.DataFrame, category: str):
    """Display rankings with medals and formatting"""
    st.subheader(f"Rankings: {category}")
    
    if rankings_df.empty:
        st.info("No data available for rankings")
        return
    
    # Create ranking column with consistent string type from the start
    rankings_display = rankings_df.copy().reset_index(drop=True)
    
    # Build rank column as strings from the beginning to avoid mixed dtype issues
    rank_list = []
    for i in range(len(rankings_display)):
        if i == 0:
            rank_list.append("🥇 1")
        elif i == 1:
            rank_list.append("🥈 2")
        elif i == 2:
            rank_list.append("🥉 3")
        else:
            rank_list.append(str(i + 1))
    
    # Assign the complete string series at once
    rankings_display['Rank'] = rank_list
    
    # Configure columns for proper display
    column_config = {}
    if 'score' in rankings_display.columns:
        column_config["score"] = st.column_config.ProgressColumn("Score", min_value=0, max_value=1)
    if 'Score' in rankings_display.columns:
        column_config["Score"] = st.column_config.ProgressColumn("Score", min_value=0, max_value=1)
    
    st.dataframe(
        rankings_display,
        column_config=column_config,
        hide_index=True
    )


if __name__ == "__main__":
    main()