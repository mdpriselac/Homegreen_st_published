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
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_all_coffees(_self) -> pd.DataFrame:
        """Get all active coffees with their attributes"""
        
        try:
            # Get coffees data first, ordered by most recently first seen
            coffees_result = _self.client.table('coffees').select('*').eq('is_active', True).order('first_observed', desc=True).limit(50).execute()
            
            flattened_data = []
            
            for coffee in coffees_result.data:
                coffee_id = coffee['id']
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
                    'is_active': coffee.get('is_active', True),
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
    
    @st.cache_data(ttl=600)  # Cache for 10 minutes
    def get_coffee_by_id(_self, coffee_id: int) -> Dict[str, Any]:
        """Get detailed information for a specific coffee"""
        result = _self.client.table('coffees').select("""
            *,
            sellers(*),
            coffee_attributes(*),
            coffee_embeddings(*)
        """).eq('id', coffee_id).single().execute()
        
        return result.data
    
    def find_similar_coffees(self, coffee_id: int, similarity_type: str = 'full', limit: int = 10) -> pd.DataFrame:
        """Find similar coffees using pgvector similarity"""
        
        if similarity_type == 'full':
            vector_column = 'combined_vector_pg'
        elif similarity_type == 'flavor':
            vector_column = 'flavor_vector_pg'
        else:
            raise ValueError("similarity_type must be 'full' or 'flavor'")
        
        # Get target coffee's vector
        try:
            target_embedding = self.client.table('coffee_embeddings').select(
                vector_column
            ).eq('coffee_id', coffee_id).single().execute()
        except Exception:
            return pd.DataFrame()
        
        if not target_embedding.data or not target_embedding.data.get(vector_column):
            return pd.DataFrame()
        
        target_vector = target_embedding.data[vector_column]
        
        # Get all embeddings for comparison - filter out null vectors properly
        try:
            all_embeddings = self.client.table('coffee_embeddings').select(f"""
                coffee_id, {vector_column}
            """).not_.is_(vector_column, 'null').neq('coffee_id', coffee_id).limit(20).execute()
            
        except Exception:
            return pd.DataFrame()
        
        # Get coffee IDs and fetch full details for each
        similar_coffee_ids = [row['coffee_id'] for row in all_embeddings.data[:limit]]
        
        # Get complete coffee details for these IDs using the same method as get_all_coffees
        similar_coffees = []
        for coffee_id_sim in similar_coffee_ids:
            try:
                # Get coffee basic info
                coffee_result = self.client.table('coffees').select('*').eq('id', coffee_id_sim).execute()
                if not coffee_result.data:
                    continue
                    
                coffee = coffee_result.data[0]
                seller_id = coffee.get('seller_id')
                
                # Get seller info
                seller_info = {}
                if seller_id:
                    try:
                        seller_result = self.client.table('sellers').select('name, homepage').eq('id', seller_id).execute()
                        if seller_result.data:
                            seller_info = seller_result.data[0]
                    except:
                        pass
                
                # Get coffee attributes
                coffee_attrs = {}
                try:
                    attrs_result = self.client.table('coffee_attributes').select('*').eq('coffee_id', coffee_id_sim).execute()
                    if attrs_result.data:
                        coffee_attrs = attrs_result.data[0]
                except:
                    pass
                
                similar_coffees.append({
                    'id': coffee['id'],
                    'name': coffee['name'],
                    'url': coffee['url'],
                    'seller': seller_info.get('name', ''),
                    'seller_website': seller_info.get('homepage', ''),
                    'country_final': coffee_attrs.get('country_final', ''),
                    'subregion_final': coffee_attrs.get('subregion_final', ''),
                    'process_type_final': coffee_attrs.get('process_type_final', ''),
                    'fermentation': coffee_attrs.get('fermentation', ''),
                    'flavor_notes': coffee_attrs.get('flavor_notes', ''),
                    'categorized_flavors': coffee_attrs.get('categorized_flavors', '')
                })
            except Exception:
                continue
        
        return pd.DataFrame(similar_coffees)

# Global client instance
@st.cache_resource
def get_supabase_client():
    return FrontendSupabaseClient()