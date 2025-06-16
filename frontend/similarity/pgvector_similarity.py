"""
pgvector-based similarity search replacing ChromaDB
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from frontend.database.supabase_client import get_supabase_client

class PgvectorSimilaritySearch:
    """Similarity search using PostgreSQL pgvector"""
    
    def __init__(self):
        self.client = get_supabase_client()
    
    def find_full_similarity_matches(self, coffee_id: str, original_df: pd.DataFrame, num_matches: int = 10) -> pd.DataFrame:
        """
        Find full profile similarity matches (replaces ChromaDB function)
        
        Args:
            coffee_id: Coffee UID as string
            original_df: Original dataframe (for compatibility)
            num_matches: Number of matches to return
            
        Returns:
            DataFrame with similar coffees
        """
        try:
            # Convert string UID to integer ID
            coffee_int_id = int(coffee_id)
            
            # Get similar coffees using pgvector
            similar_df = self.client.find_similar_coffees(
                coffee_id=coffee_int_id,
                similarity_type='full',
                limit=num_matches
            )
            
            if similar_df.empty:
                return pd.DataFrame()
            
            # Format for compatibility with existing UI code
            similar_df['uid'] = similar_df['id'].astype(str)
            
            # Get additional data from original_df to match expected format
            result_data = []
            for _, row in similar_df.iterrows():
                uid = row['uid']
                # Try to get additional info from original_df
                orig_row = original_df[original_df['uid'] == uid]
                if not orig_row.empty:
                    orig_data = orig_row.iloc[0]
                    result_data.append({
                        'uid': uid,
                        'Name': row.get('name', orig_data.get('Name', '')),
                        'Seller': row.get('seller', orig_data.get('Seller', '')),
                        'Country': row.get('country_final', orig_data.get('Country', '')),
                        'Region': row.get('subregion_final', orig_data.get('Region(s)', '')),
                        'Process': row.get('process_type_final', orig_data.get('Process', '')),
                        'Fermented': row.get('fermentation', orig_data.get('Fermented?', '')),
                        'Flavor Notes': row.get('flavor_notes', orig_data.get('Flavor Notes', ''))
                    })
                else:
                    # Fallback to row data only
                    result_data.append({
                        'uid': uid,
                        'Name': row.get('name', ''),
                        'Seller': row.get('seller', ''),
                        'Country': row.get('country_final', ''),
                        'Region': row.get('subregion_final', ''),
                        'Process': row.get('process_type_final', ''),
                        'Fermented': row.get('fermentation', ''),
                        'Flavor Notes': row.get('flavor_notes', '')
                    })
            
            return pd.DataFrame(result_data)
            
        except Exception as e:
            st.error(f"Error finding similarity matches: {e}")
            return pd.DataFrame()
    
    def find_flavor_similarity_matches(self, coffee_id: str, original_df: pd.DataFrame, num_matches: int = 10) -> pd.DataFrame:
        """
        Find flavor-only similarity matches (replaces ChromaDB function)
        """
        try:
            coffee_int_id = int(coffee_id)
            
            similar_df = self.client.find_similar_coffees(
                coffee_id=coffee_int_id,
                similarity_type='flavor',
                limit=num_matches
            )
            
            if similar_df.empty:
                return pd.DataFrame()
            
            # Format for compatibility
            similar_df['uid'] = similar_df['id'].astype(str)
            
            # Get additional data from original_df to match expected format
            result_data = []
            for _, row in similar_df.iterrows():
                uid = row['uid']
                # Try to get additional info from original_df
                orig_row = original_df[original_df['uid'] == uid]
                if not orig_row.empty:
                    orig_data = orig_row.iloc[0]
                    result_data.append({
                        'uid': uid,
                        'Name': row.get('name', orig_data.get('Name', '')),
                        'Seller': row.get('seller', orig_data.get('Seller', '')),
                        'Country': row.get('country_final', orig_data.get('Country', '')),
                        'Region': row.get('subregion_final', orig_data.get('Region(s)', '')),
                        'Process': row.get('process_type_final', orig_data.get('Process', '')),
                        'Flavor Notes': row.get('flavor_notes', orig_data.get('Flavor Notes', ''))
                    })
                else:
                    # Fallback to row data only
                    result_data.append({
                        'uid': uid,
                        'Name': row.get('name', ''),
                        'Seller': row.get('seller', ''),
                        'Country': row.get('country_final', ''),
                        'Region': row.get('subregion_final', ''),
                        'Process': row.get('process_type_final', ''),
                        'Flavor Notes': row.get('flavor_notes', '')
                    })
            
            return pd.DataFrame(result_data)
            
        except Exception as e:
            st.error(f"Error finding flavor similarity matches: {e}")
            return pd.DataFrame()

# Global similarity search instance
@st.cache_resource
def get_similarity_search():
    return PgvectorSimilaritySearch()

# Compatibility functions for existing code
def find_full_similarity_matches(coffee_id: str, original_df: pd.DataFrame, full_coll_conn=None, num_matches: int = 10) -> pd.DataFrame:
    """Compatibility wrapper for existing code"""
    similarity_search = get_similarity_search()
    return similarity_search.find_full_similarity_matches(coffee_id, original_df, num_matches)

def find_flavor_similarity_matches(coffee_id: str, original_df: pd.DataFrame, flav_coll_conn=None, num_matches: int = 10) -> pd.DataFrame:
    """Compatibility wrapper for existing code"""
    similarity_search = get_similarity_search()
    return similarity_search.find_flavor_similarity_matches(coffee_id, original_df, num_matches)