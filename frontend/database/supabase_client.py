"""
Supabase client for frontend database operations
"""
import os
import streamlit as st
from supabase import create_client, Client
from typing import List, Dict, Any, Optional
import pandas as pd

class FrontendSupabaseClient:
    """Supabase client optimized for frontend operations"""
    
    def __init__(self):
        # Get credentials from Streamlit secrets
        self.supabase_url = st.secrets["supabase"]["url"]
        self.supabase_anon_key = st.secrets["supabase"]["anon_key"]
        self.client = create_client(self.supabase_url, self.supabase_anon_key)
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_all_coffees(_self, include_expired: bool = False) -> pd.DataFrame:
        """Get coffees with their attributes using optimized single query
        
        Args:
            include_expired: If True, includes both active and inactive coffees.
                           If False (default), only returns active coffees.
        """
        
        try:
            # Build query
            query = _self.client.table('coffees').select(
                'id, url, name, first_observed, last_observed, is_active, sellers(name, homepage), coffee_attributes(country_final, subregion_final, micro_final, process_type_final, fermentation, categorized_flavors, altitude_low, altitude_high, observation_date, flavor_notes, varietal)'
            )
            
            # Only filter by is_active if we don't want expired coffees
            if not include_expired:
                query = query.eq('is_active', True)
                
            # Order and execute
            result = query.order('first_observed', desc=True).limit(5000).execute()
            
            flattened_data = []
            
            for coffee in result.data:
                # Extract seller info (handle both dict and list formats)
                seller_info = coffee.get('sellers', {})
                if isinstance(seller_info, list) and len(seller_info) > 0:
                    seller_info = seller_info[0]
                elif not isinstance(seller_info, dict):
                    seller_info = {}
                
                # Extract coffee attributes (handle both dict and list formats)
                coffee_attrs = coffee.get('coffee_attributes', {})
                if isinstance(coffee_attrs, list) and len(coffee_attrs) > 0:
                    coffee_attrs = coffee_attrs[0]
                elif not isinstance(coffee_attrs, dict):
                    coffee_attrs = {}
                
                flattened_row = {
                    'id': coffee.get('id', ''),
                    'url': coffee.get('url', ''),
                    'name': coffee.get('name', ''),
                    'first_observed': coffee.get('first_observed', ''),
                    'last_observed': coffee.get('last_observed', ''),
                    'is_active': coffee.get('is_active'),
                    'seller': seller_info.get('name', ''),
                    'seller_website': seller_info.get('homepage', ''),
                    'country_final': coffee_attrs.get('country_final', ''),
                    'subregion_final': coffee_attrs.get('subregion_final', ''),
                    'micro_final': coffee_attrs.get('micro_final', ''),
                    'process_type_final': coffee_attrs.get('process_type_final', ''),
                    'fermentation': coffee_attrs.get('fermentation', ''),
                    'categorized_flavors': coffee_attrs.get('categorized_flavors', ''),
                    'altitude_low': coffee_attrs.get('altitude_low', ''),
                    'altitude_high': coffee_attrs.get('altitude_high', ''),
                    'observation_date': coffee_attrs.get('observation_date', ''),
                    'flavor_notes': coffee_attrs.get('flavor_notes', ''),
                    'varietal': coffee_attrs.get('varietal', '')
                }
                flattened_data.append(flattened_row)
            
            df = pd.DataFrame(flattened_data)
            return df
            
        except Exception as e:
            st.error(f"Error loading data from Supabase: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'id', 'url', 'name', 'first_observed', 'last_observed', 'is_active',
                'seller', 'seller_website', 'country_final', 'subregion_final', 
                'micro_final', 'process_type_final', 'fermentation', 'categorized_flavors',
                'altitude_low', 'altitude_high', 'observation_date', 'flavor_notes'
            ])
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_coffee_by_id(_self, coffee_id: int) -> Dict[str, Any]:
        """Get detailed information for a specific coffee"""
        result = _self.client.table('coffees').select("""
            *,
            sellers(*),
            coffee_attributes(*),
            coffee_embeddings(*)
        """).eq('id', coffee_id).single().execute()
        
        return result.data
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_single_coffee_formatted(_self, coffee_id: int) -> pd.DataFrame:
        """Get a single coffee in the same format as get_all_coffees"""
        try:
            # Get coffee data
            coffee_result = _self.client.table('coffees').select('*').eq('id', coffee_id).execute()
            
            if not coffee_result.data:
                return pd.DataFrame()
            
            coffee = coffee_result.data[0]
            seller_id = coffee.get('seller_id')
            
            # Get seller info
            seller_info = {}
            if seller_id:
                try:
                    seller_result = _self.client.table('sellers').select('name, homepage').eq('id', seller_id).execute()
                    if seller_result.data:
                        seller_info = seller_result.data[0]
                except:
                    pass
            
            # Get coffee attributes
            coffee_attrs = {}
            try:
                attrs_result = _self.client.table('coffee_attributes').select('*').eq('coffee_id', coffee_id).execute()
                if attrs_result.data:
                    coffee_attrs = attrs_result.data[0]
            except Exception:
                pass
            
            flattened_row = {
                'id': coffee.get('id', ''),
                'url': coffee.get('url', ''),
                'name': coffee.get('name', ''),
                'first_observed': coffee.get('first_observed', ''),
                'last_observed': coffee.get('last_observed', ''),
                'is_active': coffee.get('is_active'),
                'seller': seller_info.get('name', ''),
                'seller_website': seller_info.get('homepage', ''),
                'country_final': coffee_attrs.get('country_final', ''),
                'subregion_final': coffee_attrs.get('subregion_final', ''),
                'micro_final': coffee_attrs.get('micro_final', ''),
                'process_type_final': coffee_attrs.get('process_type_final', ''),
                'fermentation': coffee_attrs.get('fermentation', ''),
                'categorized_flavors': coffee_attrs.get('categorized_flavors', ''),
                'altitude_low': coffee_attrs.get('altitude_low', ''),
                'altitude_high': coffee_attrs.get('altitude_high', ''),
                'observation_date': coffee_attrs.get('observation_date', ''),
                'flavor_notes': coffee_attrs.get('flavor_notes', ''),
                'varietal': coffee_attrs.get('varietal', '')
            }
            
            return pd.DataFrame([flattened_row])
            
        except Exception as e:
            st.error(f"Error loading coffee {coffee_id} from Supabase: {e}")
            return pd.DataFrame()
    
    def find_similar_coffees(self, coffee_id: int, similarity_type: str = 'full', limit: int = 10) -> pd.DataFrame:
        """Find similar coffees using pgvector similarity via RPC call"""

        if similarity_type not in ['full', 'flavor']:
            raise ValueError("similarity_type must be 'full' or 'flavor'")

        # This assumes a PostgreSQL function `get_similar_coffees` exists.
        rpc_params = {
            'p_coffee_id': coffee_id,
            'p_similarity_type': similarity_type,
            'p_match_count': limit
        }

        try:
            response = self.client.rpc('get_similar_coffees', rpc_params).execute()
            if response.data:
                return pd.DataFrame(response.data)
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error finding similar coffees: {e}")
            return pd.DataFrame()

# Global client instance
@st.cache_resource
def get_supabase_client():
    return FrontendSupabaseClient()