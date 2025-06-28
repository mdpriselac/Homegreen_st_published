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


class CoffeeDataExtractor:
    """Extract and prepare coffee data for analytics"""
    
    def __init__(self):
        """Initialize Supabase client"""
        self.supabase_url = st.secrets["supabase"]["url"]
        self.supabase_anon_key = st.secrets["supabase"]["anon_key"]
        self.client = create_client(self.supabase_url, self.supabase_anon_key)
    
    @st.cache_data(ttl=3600)
    def extract_raw_data(_self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract raw data from database
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (coffee_attributes_df, coffee_seller_mapping_df)
        """
        # Step 1: Pull core dataset from coffee_attributes where is_cleaned = true
        try:
            attributes_result = _self.client.table('coffee_attributes').select(
                'coffee_id, country_final, subregion_final, categorized_flavors'
            ).eq('is_cleaned', True).execute()
            
            attributes_df = pd.DataFrame(attributes_result.data)
            
        except Exception as e:
            st.error(f"Error extracting coffee attributes: {e}")
            attributes_df = pd.DataFrame()
        
        # Step 2: Pull coffee-seller mapping
        try:
            coffee_seller_result = _self.client.table('coffees').select(
                'id, name, seller_id, sellers(id, name)'
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
                    'seller_name': seller_info.get('name', 'Unknown')
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
        
        # Parse flavors
        merged_df['flavors_parsed'] = merged_df['categorized_flavors'].apply(self._parse_flavors)
        
        # Add metadata
        merged_df['has_flavors'] = merged_df['flavors_parsed'].apply(lambda x: len(x) > 0)
        merged_df['flavor_count'] = merged_df['flavors_parsed'].apply(len)
        
        # Create region key
        merged_df['region_key'] = merged_df.apply(
            lambda row: f"{row['country_final']}_{row['subregion_final']}" 
            if pd.notna(row['subregion_final']) and row['subregion_final'] 
            else None,
            axis=1
        )
        
        return merged_df
    
    def aggregate_by_country(self, merged_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate data by country
        
        Args:
            merged_df: Merged dataset
            
        Returns:
            Dict[str, Dict[str, Any]]: Country-level aggregated data
        """
        country_data = {}
        
        for country in merged_df['country_final'].unique():
            if pd.isna(country) or not country:
                continue
                
            country_df = merged_df[merged_df['country_final'] == country]
            
            # Collect all coffees
            coffees = []
            all_flavors = []
            
            for _, row in country_df.iterrows():
                coffee_info = {
                    'coffee_id': row['coffee_id'],
                    'coffee_name': row['coffee_name'],
                    'seller_name': row['seller_name'],
                    'subregion': row['subregion_final'],
                    'flavors': row['flavors_parsed']
                }
                coffees.append(coffee_info)
                all_flavors.extend(row['flavors_parsed'])
            
            # Calculate metadata
            unique_subregions = country_df['subregion_final'].dropna().unique().tolist()
            unique_sellers = country_df['seller_name'].dropna().unique().tolist()
            
            # Get unique flavor counts at each level
            unique_families = set(f['family'] for f in all_flavors if 'family' in f)
            unique_genera = set(f['genus'] for f in all_flavors if 'genus' in f)
            unique_species = set(f['species'] for f in all_flavors if 'species' in f)
            
            country_data[country] = {
                'coffees': coffees,
                'all_flavors': all_flavors,
                'metadata': {
                    'total_coffees': len(country_df),
                    'unique_subregions': unique_subregions,
                    'unique_sellers': unique_sellers,
                    'total_flavor_instances': len(all_flavors),
                    'unique_flavor_families': list(unique_families),
                    'unique_flavor_genera': list(unique_genera),
                    'unique_flavor_species': list(unique_species)
                }
            }
        
        return country_data
    
    def aggregate_by_region(self, merged_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate data by region (country_subregion)
        
        Args:
            merged_df: Merged dataset
            
        Returns:
            Dict[str, Dict[str, Any]]: Region-level aggregated data
        """
        region_data = {}
        
        # Filter for valid regions
        region_df = merged_df[merged_df['region_key'].notna()]
        
        for region_key in region_df['region_key'].unique():
            region_rows = region_df[region_df['region_key'] == region_key]
            
            # Extract country and subregion
            country = region_rows.iloc[0]['country_final']
            subregion = region_rows.iloc[0]['subregion_final']
            
            # Collect all coffees
            coffees = []
            all_flavors = []
            
            for _, row in region_rows.iterrows():
                coffee_info = {
                    'coffee_id': row['coffee_id'],
                    'coffee_name': row['coffee_name'],
                    'seller_name': row['seller_name'],
                    'flavors': row['flavors_parsed']
                }
                coffees.append(coffee_info)
                all_flavors.extend(row['flavors_parsed'])
            
            # Calculate metadata
            unique_sellers = region_rows['seller_name'].dropna().unique().tolist()
            
            # Get unique flavor counts at each level
            unique_families = set(f['family'] for f in all_flavors if 'family' in f)
            unique_genera = set(f['genus'] for f in all_flavors if 'genus' in f)
            unique_species = set(f['species'] for f in all_flavors if 'species' in f)
            
            region_data[region_key] = {
                'country': country,
                'subregion': subregion,
                'coffees': coffees,
                'all_flavors': all_flavors,
                'metadata': {
                    'total_coffees': len(region_rows),
                    'unique_sellers': unique_sellers,
                    'total_flavor_instances': len(all_flavors),
                    'unique_flavor_families': list(unique_families),
                    'unique_flavor_genera': list(unique_genera),
                    'unique_flavor_species': list(unique_species)
                }
            }
        
        return region_data
    
    def aggregate_by_seller(self, merged_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate data by seller
        
        Args:
            merged_df: Merged dataset
            
        Returns:
            Dict[str, Dict[str, Any]]: Seller-level aggregated data
        """
        seller_data = {}
        
        for seller in merged_df['seller_name'].unique():
            if pd.isna(seller) or not seller or seller == 'Unknown':
                continue
                
            seller_df = merged_df[merged_df['seller_name'] == seller]
            
            # Get seller_id
            seller_id = seller_df['seller_id'].iloc[0] if not seller_df.empty else None
            
            # Collect all coffees
            coffees = []
            all_flavors = []
            
            for _, row in seller_df.iterrows():
                coffee_info = {
                    'coffee_id': row['coffee_id'],
                    'coffee_name': row['coffee_name'],
                    'country': row['country_final'],
                    'subregion': row['subregion_final'],
                    'flavors': row['flavors_parsed']
                }
                coffees.append(coffee_info)
                all_flavors.extend(row['flavors_parsed'])
            
            # Calculate metadata
            unique_countries = seller_df['country_final'].dropna().unique().tolist()
            unique_subregions = seller_df['subregion_final'].dropna().unique().tolist()
            
            # Get unique flavor counts at each level
            unique_families = set(f['family'] for f in all_flavors if 'family' in f)
            unique_genera = set(f['genus'] for f in all_flavors if 'genus' in f)
            unique_species = set(f['species'] for f in all_flavors if 'species' in f)
            
            seller_data[seller] = {
                'seller_id': seller_id,
                'coffees': coffees,
                'all_flavors': all_flavors,
                'metadata': {
                    'total_coffees': len(seller_df),
                    'unique_countries': unique_countries,
                    'unique_subregions': unique_subregions,
                    'total_flavor_instances': len(all_flavors),
                    'unique_flavor_families': list(unique_families),
                    'unique_flavor_genera': list(unique_genera),
                    'unique_flavor_species': list(unique_species)
                }
            }
        
        return seller_data
    
    def prepare_contingency_format(self, country_data: Dict[str, Dict[str, Any]],
                                 region_data: Dict[str, Dict[str, Any]],
                                 seller_data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare data in contingency table format for frequency analysis
        
        Returns:
            pd.DataFrame: Contingency format data
        """
        rows = []
        
        # Process countries
        for country, data in country_data.items():
            base_row = {
                'unit': country,
                'unit_type': 'country',
                'total_coffees': data['metadata']['total_coffees'],
                'total_flavor_instances': data['metadata']['total_flavor_instances']
            }
            
            # Count flavors at each level
            flavor_counts = self._count_flavors_by_level(data['all_flavors'])
            
            # Add flavor columns
            for level in ['family', 'genus', 'species']:
                for flavor, count in flavor_counts[level].items():
                    base_row[f'has_{level}_{flavor}'] = True
                    base_row[f'count_{level}_{flavor}'] = count
                    base_row[f'freq_{level}_{flavor}'] = count / data['metadata']['total_flavor_instances'] if data['metadata']['total_flavor_instances'] > 0 else 0
            
            rows.append(base_row)
        
        # Process regions
        for region, data in region_data.items():
            base_row = {
                'unit': region,
                'unit_type': 'region',
                'total_coffees': data['metadata']['total_coffees'],
                'total_flavor_instances': data['metadata']['total_flavor_instances']
            }
            
            # Count flavors at each level
            flavor_counts = self._count_flavors_by_level(data['all_flavors'])
            
            # Add flavor columns
            for level in ['family', 'genus', 'species']:
                for flavor, count in flavor_counts[level].items():
                    base_row[f'has_{level}_{flavor}'] = True
                    base_row[f'count_{level}_{flavor}'] = count
                    base_row[f'freq_{level}_{flavor}'] = count / data['metadata']['total_flavor_instances'] if data['metadata']['total_flavor_instances'] > 0 else 0
            
            rows.append(base_row)
        
        # Process sellers
        for seller, data in seller_data.items():
            base_row = {
                'unit': seller,
                'unit_type': 'seller',
                'total_coffees': data['metadata']['total_coffees'],
                'total_flavor_instances': data['metadata']['total_flavor_instances']
            }
            
            # Count flavors at each level
            flavor_counts = self._count_flavors_by_level(data['all_flavors'])
            
            # Add flavor columns
            for level in ['family', 'genus', 'species']:
                for flavor, count in flavor_counts[level].items():
                    base_row[f'has_{level}_{flavor}'] = True
                    base_row[f'count_{level}_{flavor}'] = count
                    base_row[f'freq_{level}_{flavor}'] = count / data['metadata']['total_flavor_instances'] if data['metadata']['total_flavor_instances'] > 0 else 0
            
            rows.append(base_row)
        
        df = pd.DataFrame(rows)
        
        # Fill NaN values with appropriate defaults
        # Boolean columns
        bool_cols = [col for col in df.columns if col.startswith('has_')]
        for col in bool_cols:
            df[col] = df[col].fillna(False).infer_objects(copy=False)
        
        # Count columns
        count_cols = [col for col in df.columns if col.startswith('count_')]
        for col in count_cols:
            df[col] = df[col].fillna(0).astype(int)
        
        # Frequency columns
        freq_cols = [col for col in df.columns if col.startswith('freq_')]
        for col in freq_cols:
            df[col] = df[col].fillna(0.0).astype(float)
        
        return df
    
    def prepare_tfidf_format(self, country_data: Dict[str, Dict[str, Any]],
                           region_data: Dict[str, Dict[str, Any]],
                           seller_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """
        Prepare data in document-term format for TF-IDF analysis
        
        Returns:
            Dict: Document-term format data
        """
        tfidf_data = {
            'family_level': {
                'countries': {},
                'regions': {},
                'sellers': {}
            },
            'genus_level': {
                'countries': {},
                'regions': {},
                'sellers': {}
            },
            'species_level': {
                'countries': {},
                'regions': {},
                'sellers': {}
            }
        }
        
        # Process countries
        for country, data in country_data.items():
            family_terms = [f['family'] for f in data['all_flavors'] if 'family' in f]
            genus_terms = [f['genus'] for f in data['all_flavors'] if 'genus' in f]
            species_terms = [f['species'] for f in data['all_flavors'] if 'species' in f]
            
            tfidf_data['family_level']['countries'][country] = family_terms
            tfidf_data['genus_level']['countries'][country] = genus_terms
            tfidf_data['species_level']['countries'][country] = species_terms
        
        # Process regions
        for region, data in region_data.items():
            family_terms = [f['family'] for f in data['all_flavors'] if 'family' in f]
            genus_terms = [f['genus'] for f in data['all_flavors'] if 'genus' in f]
            species_terms = [f['species'] for f in data['all_flavors'] if 'species' in f]
            
            tfidf_data['family_level']['regions'][region] = family_terms
            tfidf_data['genus_level']['regions'][region] = genus_terms
            tfidf_data['species_level']['regions'][region] = species_terms
        
        # Process sellers
        for seller, data in seller_data.items():
            family_terms = [f['family'] for f in data['all_flavors'] if 'family' in f]
            genus_terms = [f['genus'] for f in data['all_flavors'] if 'genus' in f]
            species_terms = [f['species'] for f in data['all_flavors'] if 'species' in f]
            
            tfidf_data['family_level']['sellers'][seller] = family_terms
            tfidf_data['genus_level']['sellers'][seller] = genus_terms
            tfidf_data['species_level']['sellers'][seller] = species_terms
        
        return tfidf_data
    
    def prepare_hierarchical_format(self, country_data: Dict[str, Dict[str, Any]],
                                  region_data: Dict[str, Dict[str, Any]],
                                  seller_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Prepare data in hierarchical format for multi-level analysis
        
        Returns:
            Dict: Hierarchical format data
        """
        hierarchical_data = {}
        
        # Process all unit types
        all_data = [
            ('country', country_data),
            ('region', region_data),
            ('seller', seller_data)
        ]
        
        for unit_type, unit_data in all_data:
            for unit_name, data in unit_data.items():
                # Count flavors at each level
                flavor_counts = self._count_flavors_by_level(data['all_flavors'])
                
                # Build hierarchy tree
                hierarchy_tree = self._build_hierarchy_tree(data['all_flavors'])
                
                # Calculate frequencies
                total_instances = data['metadata']['total_flavor_instances']
                
                hierarchical_data[unit_name] = {
                    'unit_type': unit_type,
                    'family_level': {
                        flavor: {
                            'count': count,
                            'frequency': count / total_instances if total_instances > 0 else 0
                        }
                        for flavor, count in flavor_counts['family'].items()
                    },
                    'genus_level': {
                        flavor: {
                            'count': count,
                            'frequency': count / total_instances if total_instances > 0 else 0,
                            'parent_family': self._find_parent_family(flavor, data['all_flavors'])
                        }
                        for flavor, count in flavor_counts['genus'].items()
                    },
                    'species_level': {
                        flavor: {
                            'count': count,
                            'frequency': count / total_instances if total_instances > 0 else 0,
                            'parent_genus': self._find_parent_genus(flavor, data['all_flavors']),
                            'parent_family': self._find_parent_family_for_species(flavor, data['all_flavors'])
                        }
                        for flavor, count in flavor_counts['species'].items()
                    },
                    'hierarchy_tree': hierarchy_tree,
                    'total_coffees': data['metadata']['total_coffees'],
                    'total_flavor_instances': total_instances
                }
        
        return hierarchical_data
    
    def _count_flavors_by_level(self, flavors: List[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
        """Count flavor occurrences at each taxonomy level"""
        counts = {
            'family': {},
            'genus': {},
            'species': {}
        }
        
        for flavor in flavors:
            if 'family' in flavor and flavor['family']:
                counts['family'][flavor['family']] = counts['family'].get(flavor['family'], 0) + 1
            if 'genus' in flavor and flavor['genus']:
                counts['genus'][flavor['genus']] = counts['genus'].get(flavor['genus'], 0) + 1
            if 'species' in flavor and flavor['species']:
                counts['species'][flavor['species']] = counts['species'].get(flavor['species'], 0) + 1
        
        return counts
    
    def _build_hierarchy_tree(self, flavors: List[Dict[str, str]]) -> Dict[str, Dict[str, List[str]]]:
        """Build hierarchical tree structure of flavors"""
        tree = {}
        
        for flavor in flavors:
            family = flavor.get('family', '')
            genus = flavor.get('genus', '')
            species = flavor.get('species', '')
            
            if family:
                if family not in tree:
                    tree[family] = {}
                if genus:
                    if genus not in tree[family]:
                        tree[family][genus] = []
                    if species and species not in tree[family][genus]:
                        tree[family][genus].append(species)
        
        return tree
    
    def _find_parent_family(self, genus: str, flavors: List[Dict[str, str]]) -> str:
        """Find parent family for a genus"""
        for flavor in flavors:
            if flavor.get('genus') == genus:
                return flavor.get('family', '')
        return ''
    
    def _find_parent_genus(self, species: str, flavors: List[Dict[str, str]]) -> str:
        """Find parent genus for a species"""
        for flavor in flavors:
            if flavor.get('species') == species:
                return flavor.get('genus', '')
        return ''
    
    def _find_parent_family_for_species(self, species: str, flavors: List[Dict[str, str]]) -> str:
        """Find parent family for a species"""
        for flavor in flavors:
            if flavor.get('species') == species:
                return flavor.get('family', '')
        return ''
    
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
        
        # Aggregate by different levels
        country_data = self.aggregate_by_country(merged_df)
        region_data = self.aggregate_by_region(merged_df)
        seller_data = self.aggregate_by_seller(merged_df)
        
        # Prepare different formats
        contingency_df = self.prepare_contingency_format(country_data, region_data, seller_data)
        tfidf_data = self.prepare_tfidf_format(country_data, region_data, seller_data)
        hierarchical_data = self.prepare_hierarchical_format(country_data, region_data, seller_data)
        
        return {
            'raw_merged_df': merged_df,
            'country_aggregated': country_data,
            'region_aggregated': region_data,
            'seller_aggregated': seller_data,
            'contingency_format': contingency_df,
            'tfidf_format': tfidf_data,
            'hierarchical_format': hierarchical_data,
            'extraction_timestamp': datetime.now().isoformat()
        }


# Convenience function for caching
@st.cache_data(ttl=3600)
def get_analytics_data():
    """Get all analytics data with caching"""
    extractor = CoffeeDataExtractor()
    return extractor.extract_and_prepare_all_data()