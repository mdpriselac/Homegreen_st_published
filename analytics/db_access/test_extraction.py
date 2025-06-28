"""
Test script for coffee data extraction
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analytics.db_access.coffee_data_extractor import CoffeeDataExtractor, get_analytics_data

def test_extraction():
    """Test the data extraction functionality"""
    
    st.title("Coffee Data Extraction Test")
    
    try:
        # Initialize extractor
        extractor = CoffeeDataExtractor()
        
        # Test raw data extraction
        st.header("1. Testing Raw Data Extraction")
        with st.spinner("Extracting raw data..."):
            attributes_df, coffee_seller_df = extractor.extract_raw_data()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Coffee Attributes Records", len(attributes_df))
        with col2:
            st.metric("Coffee-Seller Mappings", len(coffee_seller_df))
        
        # Show sample data
        if not attributes_df.empty:
            st.subheader("Sample Coffee Attributes")
            st.dataframe(attributes_df.head())
        
        if not coffee_seller_df.empty:
            st.subheader("Sample Coffee-Seller Mapping")
            st.dataframe(coffee_seller_df.head())
        
        # Test full data preparation
        st.header("2. Testing Full Data Preparation")
        with st.spinner("Preparing all data formats..."):
            all_data = get_analytics_data()
        
        st.success("Data extraction and preparation completed!")
        
        # Display summary statistics
        st.header("3. Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Countries", len(all_data['country_aggregated']))
        with col2:
            st.metric("Total Regions", len(all_data['region_aggregated']))
        with col3:
            st.metric("Total Sellers", len(all_data['seller_aggregated']))
        
        # Show sample aggregated data
        st.header("4. Sample Aggregated Data")
        
        # Country sample
        if all_data['country_aggregated']:
            sample_country = list(all_data['country_aggregated'].keys())[0]
            st.subheader(f"Sample Country Data: {sample_country}")
            country_metadata = all_data['country_aggregated'][sample_country]['metadata']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Coffees", country_metadata['total_coffees'])
                st.metric("Total Flavor Instances", country_metadata['total_flavor_instances'])
            with col2:
                st.metric("Unique Subregions", len(country_metadata['unique_subregions']))
                st.metric("Unique Sellers", len(country_metadata['unique_sellers']))
            
            st.write("Unique Flavor Families:", country_metadata['unique_flavor_families'][:5])
        
        # Contingency format sample
        st.header("5. Contingency Format Sample")
        contingency_df = all_data['contingency_format']
        if not contingency_df.empty:
            st.dataframe(contingency_df.head())
            
            # Show available flavor columns
            flavor_cols = [col for col in contingency_df.columns if col.startswith('has_') or col.startswith('count_') or col.startswith('freq_')]
            st.write(f"Total flavor columns: {len(flavor_cols)}")
            st.write("Sample flavor columns:", flavor_cols[:10])
        
        # TF-IDF format sample
        st.header("6. TF-IDF Format Structure")
        tfidf_data = all_data['tfidf_format']
        st.write("Available levels:", list(tfidf_data.keys()))
        st.write("Available unit types:", list(tfidf_data['family_level'].keys()))
        
        # Show sample terms for a country
        if tfidf_data['family_level']['countries']:
            sample_country = list(tfidf_data['family_level']['countries'].keys())[0]
            sample_terms = tfidf_data['family_level']['countries'][sample_country][:10]
            st.write(f"Sample family terms for {sample_country}:", sample_terms)
        
        # Hierarchical format sample
        st.header("7. Hierarchical Format Sample")
        hierarchical_data = all_data['hierarchical_format']
        if hierarchical_data:
            sample_unit = list(hierarchical_data.keys())[0]
            st.subheader(f"Hierarchical Data for: {sample_unit}")
            
            unit_data = hierarchical_data[sample_unit]
            st.write(f"Unit Type: {unit_data['unit_type']}")
            st.write(f"Total Coffees: {unit_data['total_coffees']}")
            st.write(f"Total Flavor Instances: {unit_data['total_flavor_instances']}")
            
            # Show sample hierarchy tree
            st.write("Sample Hierarchy Tree:")
            tree_sample = dict(list(unit_data['hierarchy_tree'].items())[:3])
            st.json(tree_sample)
        
        st.header("8. Data Quality Check")
        
        # Check for missing values
        raw_df = all_data['raw_merged_df']
        missing_countries = raw_df['country_final'].isna().sum()
        missing_flavors = raw_df['categorized_flavors'].isna().sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Missing Countries", missing_countries)
        with col2:
            st.metric("Missing Flavors", missing_flavors)
        
        # Flavor parsing success rate
        total_with_flavors = (raw_df['has_flavors'] == True).sum()
        flavor_parse_rate = total_with_flavors / len(raw_df) * 100 if len(raw_df) > 0 else 0
        st.metric("Flavor Parse Success Rate", f"{flavor_parse_rate:.1f}%")
        
    except Exception as e:
        st.error(f"Error during extraction: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    test_extraction()