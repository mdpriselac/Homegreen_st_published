"""
Database data loading functions replacing CSV/S3 approach
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from frontend.database.supabase_client import get_supabase_client

@st.cache_data(ttl=300)
def load_full_dataset() -> pd.DataFrame:
    """
    Load full dataset from Supabase (replaces CSV loading)
    """
    client = get_supabase_client()
    df = client.get_all_coffees()
    
    # Add uid column for compatibility with existing code
    df['uid'] = df['id'].astype(str)
    
    # Process subregion_final to match expected format
    df['subregion_final'] = df['subregion_final'].apply(
        lambda x: str(x) if pd.notna(x) else "['UNKNOWN']"
    )
    
    # Process categorized_flavors - keep the original data structure
    df['categorized_flavors'] = df['categorized_flavors'].apply(
        lambda x: str(x) if pd.notna(x) and x != '' and x != '{}' else "{}"
    )
    
    # Categorized flavors are now properly preserved
    
    # Add derived columns for compatibility
    df['expired'] = ~df['is_active']
    df['first_date_observed'] = pd.to_datetime(df['first_observed']).dt.date
    
    # Map database fields to expected column names
    column_mapping = {
        'name': 'Name',
        'process_type_final': 'process_type',
        'flavor_notes': 'Flavor Notes'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # Handle varietal data - use the varietal column from coffee_attributes
    if 'varietal' in df.columns:
        df['Varietal'] = df['varietal']
        df['Varietal Cleaned'] = df['varietal']  # Use same data for both
    else:
        df['Varietal'] = ''
        df['Varietal Cleaned'] = ''
    
    # Add other placeholder columns that might be expected by existing code
    if 'Score' not in df.columns:
        df['Score'] = ''
    
    return df

@st.cache_data(ttl=600)
def load_all_coffees_df() -> pd.DataFrame:
    """
    Load and format data for the full dataset page
    """
    df = load_full_dataset()
    
    # Select columns needed for display
    display_cols = [
        'seller', 'uid', 'Name', 'country_final', 'subregion_final',
        'micro_final', 'Flavor Notes', 'process_type', 'fermentation',
        'Varietal Cleaned', 'Varietal', 'first_date_observed', 'expired',
        'categorized_flavors'  # Add this so it doesn't get filtered out
    ]
    
    # Filter to available columns
    available_cols = [col for col in display_cols if col in df.columns]
    out_df = df[available_cols].copy()
    
    # categorized_flavors is now properly included in the available columns
    
    # Merge Varietal columns
    if 'Varietal Cleaned' in out_df.columns and 'Varietal' in out_df.columns:
        out_df['Varietal(s)'] = out_df['Varietal Cleaned'].fillna(out_df['Varietal'])
    
    # Rename columns to match expected display format
    col_renames = {
        'Name': 'Name',
        'country_final': 'Country',
        'seller': 'Seller', 
        'subregion_final': 'Region(s)',
        'micro_final': 'Micro Location',
        'process_type': 'Process',
        'fermentation': 'Fermented?',
        'first_date_observed': 'Date First Seen',
        'expired': 'Expired?',
        'Flavor Notes': 'Flavor Notes'
    }
    
    out_df.rename(columns=col_renames, inplace=True)
    out_df.reset_index(drop=True, inplace=True)
    
    return out_df

def get_coffee_details(coffee_id: int) -> Dict[str, Any]:
    """
    Get detailed information for individual coffee page
    """
    client = get_supabase_client()
    return client.get_coffee_by_id(coffee_id)