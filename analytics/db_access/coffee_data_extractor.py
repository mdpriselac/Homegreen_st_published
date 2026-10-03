"""
Coffee data extraction module for analytics

This module handles all database access for the coffee flavor analytics system.
It extracts data from Supabase and prepares it for analysis.
"""

import streamlit as st
from supabase import create_client, Client
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import json
from datetime import datetime
import numpy as np

from analytics.processing.data_hygiene import (
    add_region_key, clean_flavors, clean_text, normalise_subregion, normalize_process,
)
from analytics.processing.varietals import clean_varietal_list, split_varietal_string

# pandas 2.x: opt into the future no-silent-downcasting behaviour (it is the default,
# and the option is deprecated, in pandas 3)
if int(pd.__version__.split('.')[0]) < 3:
    pd.set_option('future.no_silent_downcasting', True)


class CoffeeDataExtractor:
    """Extract and prepare coffee data for analytics"""
    
    def __init__(self):
        """Initialize Supabase client"""
        self.supabase_url = st.secrets["supabase"]["url"]
        self.supabase_anon_key = st.secrets["supabase"]["anon_key"]
        self.client = create_client(self.supabase_url, self.supabase_anon_key)
    
    def extract_raw_data(_self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract raw data from database
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (coffee_attributes_df, coffee_seller_mapping_df)
        """
        # Step 1: Pull core dataset from coffee_attributes where is_cleaned = true
        try:
            attributes_result = _self.client.table('coffee_attributes').select(
                'coffee_id, country_final, subregion_final, categorized_flavors, '
                'process_type_final, varietal, average_per_lb, cheapest_per_lb, highest_per_lb'
            ).eq('is_cleaned', True).execute()
            
            attributes_df = pd.DataFrame(attributes_result.data)
            
        except Exception as e:
            st.error(f"Error extracting coffee attributes: {e}")
            attributes_df = pd.DataFrame()
        
        # Step 2: Pull coffee-seller mapping with temporal data
        try:
            coffee_seller_result = _self.client.table('coffees').select(
                'id, name, seller_id, first_observed, last_observed, is_active, sellers(id, name)'
            ).execute()
            
            # Flatten the nested seller data
            coffee_seller_data = []
            for coffee in coffee_seller_result.data:
                seller_info = coffee.get('sellers', {})
                if isinstance(seller_info, list) and len(seller_info) > 0:
                    seller_info = seller_info[0]
                elif not isinstance(seller_info, dict):
                    seller_info = {}
                
                coffee_seller_data.append({
                    'coffee_id': coffee['id'],
                    'coffee_name': coffee['name'],
                    'seller_id': coffee.get('seller_id'),
                    'seller_name': seller_info.get('name', 'Unknown'),
                    'first_observed': coffee.get('first_observed'),
                    'last_observed': coffee.get('last_observed'),
                    'is_active': coffee.get('is_active')
                })
            
            coffee_seller_df = pd.DataFrame(coffee_seller_data)
            
        except Exception as e:
            st.error(f"Error extracting coffee-seller mapping: {e}")
            coffee_seller_df = pd.DataFrame()
        
        return attributes_df, coffee_seller_df
    
    def _parse_flavors(self, flavors_json: Any) -> List[Dict[str, str]]:
        """
        Parse categorized_flavors JSON field
        
        Args:
            flavors_json: JSON string or list of flavor dictionaries
            
        Returns:
            List[Dict[str, str]]: List of flavor dictionaries with family, genus, species
        """
        if pd.isna(flavors_json) or flavors_json is None:
            return []
        
        # If it's already a list, return it
        if isinstance(flavors_json, list):
            return flavors_json
        
        # If it's a string, parse it
        if isinstance(flavors_json, str):
            try:
                # First try standard JSON parsing
                parsed = json.loads(flavors_json)
                if isinstance(parsed, list):
                    return parsed
            except:
                try:
                    # If that fails, try replacing single quotes with double quotes
                    # This handles the common case of Python dict string representation
                    fixed_json = flavors_json.replace("'", '"')
                    parsed = json.loads(fixed_json)
                    if isinstance(parsed, list):
                        return parsed
                except:
                    # If both fail, try using ast.literal_eval as last resort
                    try:
                        import ast
                        parsed = ast.literal_eval(flavors_json)
                        if isinstance(parsed, list):
                            return parsed
                    except:
                        pass
        
        return []

    def _parse_varietal(self, varietal_val) -> List[str]:
        """Parse varietal field into a list of canonical varietal names.

        Placeholders ("UNKNOWN", "[]", "") yield []; spellings are normalised
        (see analytics.processing.varietals) and duplicates removed.
        """
        if varietal_val is None or (isinstance(varietal_val, float) and pd.isna(varietal_val)):
            return []

        if isinstance(varietal_val, (list, tuple)):
            return clean_varietal_list(list(varietal_val))

        if isinstance(varietal_val, str):
            val = varietal_val.strip()
            if not val:
                return []
            # Try parsing as list representation
            for parser in [json.loads, lambda s: json.loads(s.replace("'", '"'))]:
                try:
                    parsed = parser(val)
                    if isinstance(parsed, list):
                        return clean_varietal_list([str(v) for v in parsed if v])
                except Exception:
                    continue
            try:
                import ast
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return clean_varietal_list([str(v) for v in parsed if v])
            except Exception:
                pass
            # Free text: comma-separated (commas inside parentheses are kept)
            return clean_varietal_list(split_varietal_string(val))

        return []

    def _normalize_process_type(self, process_val) -> Optional[str]:
        """Exact-match process normalisation; unknown values become NaN (logged)."""
        return normalize_process(process_val)

    def merge_and_prepare_data(self, attributes_df: pd.DataFrame,
                             coffee_seller_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge datasets and prepare for analysis
        
        Args:
            attributes_df: DataFrame with coffee attributes
            coffee_seller_df: DataFrame with coffee-seller mapping
            
        Returns:
            pd.DataFrame: Merged and prepared dataset
        """
        # Merge on coffee_id
        merged_df = attributes_df.merge(
            coffee_seller_df, 
            on='coffee_id', 
            how='left'
        )
        
        # Placeholders ("UNKNOWN", "", "N/A", ...) are missing values, not categories
        for col in ['country_final', 'subregion_final', 'seller_name']:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].map(clean_text).astype(object)

        # Known subregion spelling variants are merged BEFORE the region key is built,
        # so raw columns and keys agree (placeholders become NaN here too)
        merged_df['subregion_final'] = pd.Series(
            [normalise_subregion(c, s) for c, s in zip(merged_df['country_final'], merged_df['subregion_final'])],
            index=merged_df.index, dtype=object)

        # Parse flavors (falsy family/genus/species dropped at their level)
        merged_df['flavors_parsed'] = merged_df['categorized_flavors'].apply(
            lambda v: clean_flavors(self._parse_flavors(v)))

        # Add metadata
        merged_df['has_flavors'] = merged_df['flavors_parsed'].apply(lambda x: len(x) > 0)
        merged_df['flavor_count'] = merged_df['flavors_parsed'].apply(len)

        # Region identity = country + subregion (NaN if either is missing)
        merged_df['region_key'] = add_region_key(merged_df, 'country_final', 'subregion_final')

        # Parse varietal field
        merged_df['varietals_parsed'] = merged_df['varietal'].apply(self._parse_varietal)
        merged_df['has_varietal'] = merged_df['varietals_parsed'].apply(lambda x: len(x) > 0)

        # Normalize process type (exact mapping; unrecognised -> NaN)
        merged_df['process_type_clean'] = merged_df['process_type_final'].apply(
            self._normalize_process_type).astype(object)
        merged_df['has_process'] = merged_df['process_type_clean'].notna()

        # Parse price fields to numeric
        for price_col in ['average_per_lb', 'cheapest_per_lb', 'highest_per_lb']:
            if price_col in merged_df.columns:
                merged_df[price_col] = pd.to_numeric(merged_df[price_col], errors='coerce')
        # Headline price = cheapest per-lb price offered (usually the largest bag).
        # average_per_lb mixes small-bag and bulk per-lb prices, so it is not used.
        merged_df['has_price'] = merged_df['cheapest_per_lb'].notna()

        # Parse date fields and compute lifespan
        merged_df['first_observed'] = pd.to_datetime(merged_df['first_observed'], errors='coerce')
        merged_df['last_observed'] = pd.to_datetime(merged_df['last_observed'], errors='coerce')
        merged_df['lifespan_days'] = (merged_df['last_observed'] - merged_df['first_observed']).dt.days

        return merged_df
    
    def prepare_cross_feature_format(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare flat per-coffee DataFrame for cross-feature analysis.
        Each row is one coffee with all attributes available.
        """
        rows = []
        for _, row in merged_df.iterrows():
            # Extract flavor families as a list
            flavor_families = list(set(
                f['family'] for f in row.get('flavors_parsed', [])
                if f.get('family')
            ))
            flavor_genera = list(set(
                f['genus'] for f in row.get('flavors_parsed', [])
                if f.get('genus')
            ))
            flavor_species = list(set(
                f['species'] for f in row.get('flavors_parsed', [])
                if f.get('species')
            ))

            rows.append({
                'coffee_id': row.get('coffee_id'),
                'coffee_name': row.get('coffee_name'),
                'country': row.get('country_final'),
                'region': row.get('region_key'),   # country+subregion, never bare subregion
                'subregion': row.get('subregion_final'),
                'seller': row.get('seller_name'),
                'process_type': row.get('process_type_clean'),
                'varietals': row.get('varietals_parsed', []),
                'price_per_lb': row.get('cheapest_per_lb'),
                'max_price': row.get('highest_per_lb'),
                'flavor_families': flavor_families,
                'flavor_genera': flavor_genera,
                'flavor_species': flavor_species,
                'first_observed': row.get('first_observed'),
                'last_observed': row.get('last_observed'),
                'is_active': row.get('is_active'),
                'lifespan_days': row.get('lifespan_days'),
                'has_flavors': row.get('has_flavors', False),
                'has_price': row.get('has_price', False),
                'has_varietal': row.get('has_varietal', False),
                'has_process': row.get('has_process', False),
                'has_region': pd.notna(row.get('region_key')),
            })

        return pd.DataFrame(rows)

    def extract_and_prepare_all_data(self) -> Dict[str, Any]:
        """
        Main method to extract and prepare all data formats
        
        Returns:
            Dict containing all prepared data formats
        """
        # Extract raw data
        attributes_df, coffee_seller_df = self.extract_raw_data()
        
        # Merge and prepare
        merged_df = self.merge_and_prepare_data(attributes_df, coffee_seller_df)
        
        # Prepare cross-feature flat format
        cross_feature_df = self.prepare_cross_feature_format(merged_df)

        return {
            'raw_merged_df': merged_df,
            'cross_feature_df': cross_feature_df,
            'extraction_timestamp': datetime.now().isoformat()
        }


# Used only by the offline cache generator; the live site reads the cache file, not the database.
def get_analytics_data():
    """Extract and prepare all analytics data"""
    extractor = CoffeeDataExtractor()
    return extractor.extract_and_prepare_all_data()