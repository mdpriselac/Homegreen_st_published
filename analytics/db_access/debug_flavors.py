"""
Debug script to investigate flavor data issues
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analytics.db_access.coffee_data_extractor import CoffeeDataExtractor
import pandas as pd
import json

def debug_flavors():
    """Debug the flavor data extraction"""
    
    st.title("Flavor Data Debug")
    
    try:
        # Initialize extractor
        extractor = CoffeeDataExtractor()
        
        # Get raw data
        st.header("1. Raw Flavor Data Investigation")
        attributes_df, _ = extractor.extract_raw_data()
        
        # Check the categorized_flavors column
        st.subheader("Flavor Data Statistics")
        total_records = len(attributes_df)
        non_null_flavors = attributes_df['categorized_flavors'].notna().sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", total_records)
            st.metric("Non-null Flavors", non_null_flavors)
        with col2:
            st.metric("Null Flavors", total_records - non_null_flavors)
            st.metric("Non-null %", f"{non_null_flavors/total_records*100:.1f}%")
        
        # Sample some non-null flavor data
        st.subheader("Sample Non-null Flavor Data")
        sample_flavors = attributes_df[attributes_df['categorized_flavors'].notna()].head(10)
        
        for idx, row in sample_flavors.iterrows():
            st.write(f"**Coffee ID: {row['coffee_id']}**")
            st.write(f"Country: {row['country_final']}")
            st.write(f"Raw flavor data type: {type(row['categorized_flavors'])}")
            st.write(f"Raw flavor data: {row['categorized_flavors']}")
            
            # Try to parse
            parsed = extractor._parse_flavors(row['categorized_flavors'])
            st.write(f"Parsed result: {parsed}")
            st.write(f"Parsed count: {len(parsed)}")
            st.divider()
        
        # Check different data types in the column
        st.subheader("Data Types in categorized_flavors")
        type_counts = {}
        empty_list_count = 0
        
        for val in attributes_df['categorized_flavors']:
            val_type = type(val).__name__
            if val_type not in type_counts:
                type_counts[val_type] = 0
            type_counts[val_type] += 1
            
            # Check for empty lists
            if isinstance(val, list) and len(val) == 0:
                empty_list_count += 1
        
        st.write("Type distribution:")
        for dtype, count in type_counts.items():
            st.write(f"- {dtype}: {count}")
        st.write(f"- Empty lists: {empty_list_count}")
        
        # Try different parsing approaches
        st.header("2. Parsing Test Results")
        
        successful_parses = 0
        failed_parses = 0
        empty_results = 0
        
        for idx, row in attributes_df.iterrows():
            parsed = extractor._parse_flavors(row['categorized_flavors'])
            if parsed and len(parsed) > 0:
                successful_parses += 1
            elif parsed == []:
                empty_results += 1
            else:
                failed_parses += 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Successful Parses", successful_parses)
        with col2:
            st.metric("Empty Results", empty_results)
        with col3:
            st.metric("Failed Parses", failed_parses)
        
        # Look for patterns in successful vs unsuccessful parses
        st.header("3. Pattern Analysis")
        
        # Get a sample of each type
        st.subheader("Successful Parse Examples")
        success_df = attributes_df[attributes_df['categorized_flavors'].apply(
            lambda x: len(extractor._parse_flavors(x)) > 0
        )].head(5)
        
        for idx, row in success_df.iterrows():
            st.write(f"Coffee ID: {row['coffee_id']}, Country: {row['country_final']}")
            st.json(row['categorized_flavors'])
        
        st.subheader("Empty/Failed Parse Examples")
        fail_df = attributes_df[attributes_df['categorized_flavors'].apply(
            lambda x: len(extractor._parse_flavors(x)) == 0
        )].head(5)
        
        for idx, row in fail_df.iterrows():
            st.write(f"Coffee ID: {row['coffee_id']}, Country: {row['country_final']}")
            st.write(f"Raw value: {repr(row['categorized_flavors'])}")
            st.write(f"Type: {type(row['categorized_flavors'])}")
        
    except Exception as e:
        st.error(f"Error during debugging: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    debug_flavors()