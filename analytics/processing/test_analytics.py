"""
Test script for analytics processing system

This script tests the statistical, TF-IDF, and hierarchical analysis modules
to ensure they're working correctly with the data extraction system.
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from analytics.db_access.coffee_data_extractor import get_analytics_data
from analytics.processing.statistical_analysis import FlavorStatisticalAnalyzer, run_full_statistical_analysis
from analytics.processing.tfidf_analysis import FlavorTFIDFAnalyzer, run_full_tfidf_analysis
from analytics.processing.hierarchical_analysis import HierarchicalFlavorAnalyzer, run_full_hierarchical_analysis


def test_analytics_processing():
    """Test all analytics processing modules"""
    
    st.title("Analytics Processing System Test")
    
    # Extract data first
    st.header("1. Data Extraction Test")
    with st.spinner("Loading data..."):
        data = get_analytics_data()
    
    if not data:
        st.error("Failed to load data")
        return
    
    st.success("✅ Data extraction successful")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Countries", len(data['country_aggregated']))
    with col2:
        st.metric("Regions", len(data['region_aggregated']))
    with col3:
        st.metric("Sellers", len(data['seller_aggregated']))
    
    # Test Statistical Analysis
    st.header("2. Statistical Analysis Test")
    try:
        with st.spinner("Running statistical analysis..."):
            contingency_df = data['contingency_format']
            stat_analyzer = FlavorStatisticalAnalyzer(contingency_df)
            
            # Test single analysis
            st.subheader("2.1 Single Analysis Test")
            country_family_results = stat_analyzer.analyze_all_associations(
                unit_type='country',
                taxonomy_level='family',
                min_occurrences=3
            )
            
            st.write(f"Total associations tested: {len(country_family_results)}")
            
            # Get significant results
            significant = stat_analyzer.get_significant_associations(country_family_results)
            st.write(f"Significant associations found: {len(significant)}")
            
            if not significant.empty:
                st.subheader("Top 5 Significant Associations")
                display_cols = ['unit_name', 'flavor', 'odds_ratio', 'p_value_corrected', 'cramers_v']
                st.dataframe(significant[display_cols].head())
            
        st.success("✅ Statistical analysis working")
        
    except Exception as e:
        st.error(f"❌ Statistical analysis failed: {str(e)}")
        st.exception(e)
    
    # Test TF-IDF Analysis
    st.header("3. TF-IDF Analysis Test")
    try:
        with st.spinner("Running TF-IDF analysis..."):
            tfidf_data = data['tfidf_format']
            tfidf_analyzer = FlavorTFIDFAnalyzer(tfidf_data)
            
            # Test single calculation
            st.subheader("3.1 Single TF-IDF Calculation")
            family_scores = tfidf_analyzer.calculate_tfidf_scores('family', 'countries')
            
            st.write(f"Total TF-IDF scores calculated: {len(family_scores)}")
            
            if not family_scores.empty:
                # Show top distinctive flavors
                st.subheader("Top 5 Distinctive Country-Flavor Pairs")
                top_scores = family_scores.nlargest(5, 'tfidf_score')
                display_cols = ['unit_name', 'flavor', 'tfidf_score', 'tf_score', 'idf_score']
                st.dataframe(top_scores[display_cols])
                
                # Test flavor fingerprints
                st.subheader("3.2 Flavor Fingerprints Test")
                fingerprints = tfidf_analyzer.create_flavor_fingerprints('family', top_n=3)
                
                if 'countries' in fingerprints:
                    sample_country = list(fingerprints['countries'].keys())[0]
                    st.write(f"Sample fingerprint for {sample_country}:")
                    st.json(fingerprints['countries'][sample_country])
        
        st.success("✅ TF-IDF analysis working")
        
    except Exception as e:
        st.error(f"❌ TF-IDF analysis failed: {str(e)}")
        st.exception(e)
    
    # Test Hierarchical Analysis
    st.header("4. Hierarchical Analysis Test")
    try:
        with st.spinner("Running hierarchical analysis..."):
            hierarchical_data = data['hierarchical_format']
            hier_analyzer = HierarchicalFlavorAnalyzer(hierarchical_data)
            
            # Test cascade analysis
            st.subheader("4.1 Cascade Pattern Analysis")
            sample_unit = list(hierarchical_data.keys())[0]
            cascade_patterns = hier_analyzer.analyze_cascade_patterns(sample_unit)
            
            st.write(f"Testing cascade analysis for: {sample_unit}")
            st.write(f"Number of patterns found: {len(cascade_patterns.get('patterns', []))}")
            
            if cascade_patterns.get('patterns'):
                st.write("Sample pattern:")
                sample_pattern = cascade_patterns['patterns'][0]
                pattern_info = {
                    'family': sample_pattern['family'],
                    'frequency': sample_pattern['family_frequency'],
                    'genera_count': len(sample_pattern['genera']),
                    'specificity': sample_pattern['specificity_score']
                }
                st.json(pattern_info)
            
            # Test distinctiveness types
            st.subheader("4.2 Distinctiveness Types")
            dist_types = hier_analyzer.identify_distinctiveness_types(sample_unit)
            
            for dtype, patterns in dist_types.items():
                if patterns:
                    st.write(f"**{dtype}**: {len(patterns)} patterns")
        
        st.success("✅ Hierarchical analysis working")
        
    except Exception as e:
        st.error(f"❌ Hierarchical analysis failed: {str(e)}")
        st.exception(e)
    
    # Test Full Integration
    st.header("5. Full Analysis Integration Test")
    try:
        with st.spinner("Running full integrated analysis (this may take a moment)..."):
            # Run smaller subset for testing
            st.info("Running subset analysis for performance...")
            
            # Test statistical on smaller subset
            limited_contingency = contingency_df[
                contingency_df['unit_type'] == 'country'
            ].head(10)  # Test with just 10 countries
            
            if not limited_contingency.empty:
                stat_results = {}
                for level in ['family']:  # Test just family level
                    analyzer = FlavorStatisticalAnalyzer(limited_contingency)
                    results = analyzer.analyze_all_associations('country', level, min_occurrences=2)
                    significant = analyzer.get_significant_associations(results)
                    stat_results[f'country_{level}_significant'] = significant
                
                st.write(f"Statistical results: {len(stat_results)} datasets")
                
                # Test TF-IDF subset
                tfidf_results = {}
                tfidf_analyzer = FlavorTFIDFAnalyzer(tfidf_data)
                for level in ['family']:
                    scores = tfidf_analyzer.calculate_tfidf_scores(level, 'countries')
                    # Limit to countries in our test set
                    test_countries = limited_contingency['unit'].unique()
                    limited_scores = scores[scores['unit_name'].isin(test_countries)]
                    tfidf_results[f'{level}_countries_scores'] = limited_scores
                
                st.write(f"TF-IDF results: {len(tfidf_results)} datasets")
                
                # Test hierarchical subset
                test_hier_data = {
                    country: data for country, data in hierarchical_data.items()
                    if country in test_countries[:5]  # Just 5 countries
                }
                
                hier_results = run_full_hierarchical_analysis(
                    test_hier_data, stat_results, tfidf_results
                )
                
                st.write(f"Hierarchical results: {len(hier_results['profiles'])} profiles")
                
                # Show sample integrated findings
                st.subheader("5.1 Sample Integration Results")
                if hier_results['top_hierarchical_units']:
                    top_unit = hier_results['top_hierarchical_units'][0]
                    st.json({
                        'top_distinctive_unit': top_unit['unit_name'],
                        'distinctiveness_score': top_unit['distinctiveness_score'],
                        'unit_type': top_unit['unit_type']
                    })
        
        st.success("✅ Full integration test successful")
        
    except Exception as e:
        st.error(f"❌ Integration test failed: {str(e)}")
        st.exception(e)
    
    # Performance Summary
    st.header("6. Performance Summary")
    
    performance_metrics = {
        'Data Extraction': '✅ Working',
        'Statistical Analysis': '✅ Working' if 'country_family_results' in locals() else '❌ Failed',
        'TF-IDF Analysis': '✅ Working' if 'family_scores' in locals() else '❌ Failed', 
        'Hierarchical Analysis': '✅ Working' if 'cascade_patterns' in locals() else '❌ Failed',
        'Integration': '✅ Working' if 'hier_results' in locals() else '❌ Failed'
    }
    
    for component, status in performance_metrics.items():
        if '✅' in status:
            st.success(f"{component}: {status}")
        else:
            st.error(f"{component}: {status}")
    
    # Recommendations
    st.header("7. Next Steps")
    
    if all('✅' in status for status in performance_metrics.values()):
        st.success("🎉 All analytics components are working correctly!")
        st.info("Ready to proceed with frontend development.")
    else:
        failed_components = [comp for comp, status in performance_metrics.items() if '❌' in status]
        st.warning(f"The following components need attention: {', '.join(failed_components)}")


if __name__ == "__main__":
    test_analytics_processing()