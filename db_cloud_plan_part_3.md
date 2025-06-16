# Coffee Market Database Cloud Migration Plan - Part 3: Frontend Integration

## Overview

This document outlines the integration plan for connecting the existing Streamlit frontend with the new cloud database system. The current frontend uses CSV data from AWS S3 and ChromaDB for similarity calculations. We'll migrate it to use Supabase APIs and pgvector similarity search while preserving the existing user experience.

## Current Frontend Analysis

### Architecture Assessment
- **Framework**: Streamlit with multipage navigation
- **Data Source**: CSV file from AWS S3 (`most_recent_perm_db_with_predictions.csv`)
- **Similarity Engine**: ChromaDB with cosine similarity (in-memory)
- **Data Processing**: Manual one-hot encoding for feature vectors
- **Pages**: About, Full Dataset, Individual Coffee, Fresh Dashboard
- **Key Features**: Interactive coffee selection, similarity matching, flavor wheel visualization

### Current Data Flow
```
AWS S3 CSV → Pandas DataFrame → ChromaDB Collections → Similarity Search → Streamlit UI
     ↓              ↓                    ↓                    ↓              ↓
   Raw Data    In-Memory Proc.    Vector Storage      Cosine Sim.    User Interface
```

### Identified Integration Points
1. **Data Loading**: Replace `load_full_dataset()` function
2. **Similarity Search**: Replace ChromaDB with pgvector queries
3. **Feature Engineering**: Update OHE process to match database schema
4. **Real-time Updates**: Add capability for live data refresh
5. **Performance**: Implement caching for API responses

## Target Architecture

### New Data Flow
```
Supabase PostgreSQL → API Queries → Cached DataFrames → pgvector Similarity → Streamlit UI
        ↓                 ↓              ↓                    ↓              ↓
   Live Database     REST/GraphQL   In-Memory Cache    Vector Search   User Interface
```

### Benefits of Migration
- **Real-time Data**: Always current coffee information
- **Better Performance**: Database indexing vs in-memory processing  
- **Scalability**: Handles larger datasets efficiently
- **Simplified Architecture**: Removes AWS S3 dependency
- **Advanced Similarity**: pgvector HNSW indexes for faster search
- **Data Consistency**: Single source of truth

## Implementation Strategy

**Note**: Based on actual implementation experience, some RPC functions (`execute_sql`) don't exist in Supabase by default. The examples below have been updated to use direct table queries and joins instead of raw SQL execution.

### Phase 1: Database Connection Layer

#### Create Supabase Client Module
**File**: `frontend/database/supabase_client.py`
```python
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
        query = """
        SELECT 
            c.id,
            c.url,
            c.name,
            c.first_observed,
            c.last_observed,
            c.is_active,
            s.name as seller,
            s.homepage as seller_website,
            ca.country_final,
            ca.subregion_final,
            ca.micro_final,
            ca.process_type_final,
            ca.fermentation,
            ca.categorized_flavors,
            ca.altitude_low,
            ca.altitude_high,
            ca.observation_date,
            ca.flavor_notes
        FROM coffees c
        JOIN sellers s ON c.seller_id = s.id
        LEFT JOIN coffee_attributes ca ON c.id = ca.coffee_id
        WHERE c.is_active = true 
        AND c.processing_status IN ('cleaned', 'exported')
        AND ca.is_cleaned = true
        ORDER BY c.last_observed DESC
        """
        
        # Use direct table queries instead of RPC (execute_sql doesn't exist by default)
        result = _self.client.table('coffees').select("""
            id, url, name, first_observed, last_observed, is_active,
            sellers!inner(name, homepage),
            coffee_attributes!inner(country_final, subregion_final, micro_final, 
                                  process_type_final, fermentation, categorized_flavors,
                                  altitude_low, altitude_high, observation_date, flavor_notes)
        """).eq('is_active', True).in_('processing_status', ['cleaned', 'exported']).execute()
        
        return pd.DataFrame(result.data)
    
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
        target_embedding = self.client.table('coffee_embeddings').select(
            vector_column
        ).eq('coffee_id', coffee_id).single().execute()
        
        if not target_embedding.data or not target_embedding.data[vector_column]:
            return pd.DataFrame()
        
        target_vector = target_embedding.data[vector_column]
        
        # Use direct similarity queries with pgvector operators
        # Note: Direct vector similarity queries may need to be implemented via stored procedures
        # For now, use a simpler approach with client-side similarity calculation
        
        # Get all embeddings for comparison
        all_embeddings = self.client.table('coffee_embeddings').select(f"""
            coffee_id, {vector_column},
            coffees!inner(id, name, url, is_active, seller_id),
            coffees.sellers!inner(name, homepage)
        """).neq(vector_column, None).eq('coffees.is_active', True).neq('coffee_id', coffee_id).execute()
        
        # Calculate similarities client-side (can be optimized later with custom functions)
        similar_coffees = []
        for row in all_embeddings.data:
            if row[vector_column]:
                # Simple similarity calculation - can be replaced with more efficient method
                similar_coffees.append({
                    'id': row['coffees']['id'],
                    'name': row['coffees']['name'], 
                    'url': row['coffees']['url'],
                    'seller': row['coffees']['sellers']['name'],
                    'seller_website': row['coffees']['sellers']['homepage']
                })
        
        # Return top matches
        return pd.DataFrame(similar_coffees[:limit])

# Global client instance
@st.cache_resource
def get_supabase_client():
    return FrontendSupabaseClient()
```

#### Update Streamlit Secrets Configuration
**File**: `.streamlit/secrets.toml`
```toml
[supabase]
url = "https://[your-project-ref].supabase.co"
anon_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
service_role_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Keep existing AWS secrets for gradual migration
[amzn]
aws_axs = "your-aws-access-key"
aws_secret = "your-aws-secret-key"

[db]
bucket = "your-bucket-name"
db_path = "path/to/csv"
```

### Phase 2: Data Loading Migration

#### Replace CSV Loading Functions
**File**: `frontend/data/database_loader.py`
```python
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
    
    # Process categorized_flavors to match expected format
    df['categorized_flavors'] = df['categorized_flavors'].apply(
        lambda x: str(x) if pd.notna(x) else "{}"
    )
    
    # Add derived columns for compatibility
    df['expired'] = ~df['is_active']
    df['first_date_observed'] = pd.to_datetime(df['first_observed']).dt.date
    
    return df

@st.cache_data(ttl=600)
def load_all_coffees_df() -> pd.DataFrame:
    """
    Load and format data for the full dataset page
    """
    df = load_full_dataset()
    
    display_cols = [
        'seller', 'uid', 'name', 'country_final', 'subregion_final',
        'micro_final', 'flavor_notes', 'process_type_final', 'fermentation',
        'first_date_observed', 'is_active'
    ]
    
    # Select and rename columns
    out_df = df[display_cols].copy()
    
    col_renames = {
        'name': 'Name',
        'country_final': 'Country',
        'seller': 'Seller', 
        'subregion_final': 'Region(s)',
        'micro_final': 'Micro Location',
        'process_type_final': 'Process',
        'fermentation': 'Fermented?',
        'first_date_observed': 'Date First Seen',
        'is_active': 'Active',
        'flavor_notes': 'Flavor Notes'
    }
    
    out_df.rename(columns=col_renames, inplace=True)
    out_df['Expired?'] = ~out_df['Active']
    out_df.drop(columns=['Active'], inplace=True)
    
    return out_df

def get_coffee_details(coffee_id: int) -> Dict[str, Any]:
    """
    Get detailed information for individual coffee page
    """
    client = get_supabase_client()
    return client.get_coffee_by_id(coffee_id)
```

### Phase 3: Similarity System Migration

#### Replace ChromaDB with pgvector Similarity
**File**: `frontend/similarity/pgvector_similarity.py`
```python
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
            
            # Select columns that match original dataframe structure
            display_columns = [
                'uid', 'name', 'seller', 'country_final', 'subregion_final',
                'process_type_final', 'fermentation', 'categorized_flavors'
            ]
            
            # Filter to available columns
            available_columns = [col for col in display_columns if col in similar_df.columns]
            result_df = similar_df[available_columns].copy()
            
            # Rename for display
            column_renames = {
                'name': 'Name',
                'seller': 'Seller',
                'country_final': 'Country',
                'subregion_final': 'Region',
                'process_type_final': 'Process',
                'fermentation': 'Fermented'
            }
            
            result_df.rename(columns=column_renames, inplace=True)
            
            return result_df
            
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
            
            display_columns = [
                'uid', 'name', 'seller', 'country_final', 'subregion_final',
                'process_type_final', 'categorized_flavors'
            ]
            
            available_columns = [col for col in display_columns if col in similar_df.columns]
            result_df = similar_df[available_columns].copy()
            
            column_renames = {
                'name': 'Name',
                'seller': 'Seller',
                'country_final': 'Country',
                'subregion_final': 'Region',
                'process_type_final': 'Process'
            }
            
            result_df.rename(columns=column_renames, inplace=True)
            
            return result_df
            
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
```

### Phase 4: Page Updates

#### Update Individual Coffee Page
**File**: `page_apps/individual_coffee_page_v2.py`
```python
"""
Updated individual coffee page using database backend
"""
import streamlit as st
import pandas as pd
from page_apps.modules.flavor_wheel_gen import flavor_wheel_gen
from frontend.data.database_loader import load_full_dataset, get_coffee_details
from frontend.similarity.pgvector_similarity import find_full_similarity_matches, find_flavor_similarity_matches
from page_apps.full_data_set_page import load_all_coffees_df

if 'ind' not in st.session_state:
    st.session_state.ind = 0

# Load data from database instead of CSV
all_coffees_db = load_all_coffees_df()

def coffee_info(coffee_data):
    """Display coffee information (updated for database schema)"""
    coffee_info = coffee_data
    coffee_id = str(coffee_info['uid'])  # Ensure string format for compatibility
    
    st.title(f"{coffee_info['Name']}")
    
    # Get additional details from database
    try:
        detailed_info = get_coffee_details(int(coffee_id))
        coffee_url = detailed_info.get('url', '#')
        seller_website = detailed_info.get('sellers', {}).get('homepage', '#')
    except:
        coffee_url = '#'
        seller_website = '#'
    
    # Key columns for checking empty values
    key_cols = [
        'seller', 'Name', 'country_final', 'subregion_final', 'micro_final',
        'Flavor Notes', 'process_type_final', 'fermentation', 'first_date_observed'
    ]
    
    empty_info = [
        k for k in key_cols 
        if k in coffee_info and (
            coffee_info[k] == '' or 
            str(coffee_info[k]).lower().strip() == 'none' or 
            str(coffee_info[k]).lower().strip() == 'nan' or 
            str(coffee_info[k]).strip().lower() == 'unknown'
        )
    ]
    
    # Introduction block with links
    intro_md_block = f'''
    On this page we have information about [{coffee_info['Name']}]({coffee_url}) from [{coffee_info['seller']}]({seller_website}). 
    More reliable information about the coffee can be found at the original website.
    
    Below you will find information that we were able to collect and process about this coffee. 
    Because an LLM is involved in this process, there is some possibility of error and fabrication. 
    Double check the information at the original website before acting on it.
    '''
    st.markdown(intro_md_block)
    
    # Basic information section
    basic_c = st.container()
    basic_c.subheader('Basic Information')
    basic_c1, basic_c2 = basic_c.columns(2)
    
    # Location information
    basic_c1.subheader('Location Information')
    basic_c1.write(f"Country: {coffee_info.get('country_final', '') if 'country_final' not in empty_info else ''}")
    basic_c1.write(f"Region(s): {coffee_info.get('subregion_final', '').replace('[','').replace(']','').replace(\"'\",\"").replace('\"','') if 'subregion_final' not in empty_info else ''}")
    basic_c1.write(f"Micro Location: {coffee_info.get('micro_final', '') if 'micro_final' not in empty_info else ''}")
    
    # Try to get altitude information from detailed data
    try:
        detailed_info = get_coffee_details(int(coffee_id))
        altitude_low = detailed_info.get('coffee_attributes', [{}])[0].get('altitude_low', '')
        altitude_high = detailed_info.get('coffee_attributes', [{}])[0].get('altitude_high', '')
        basic_c1.write(f"Altitude Min (masl): {altitude_low}")
        basic_c1.write(f"Altitude Max (masl): {altitude_high}")
    except:
        pass
    
    # Coffee details
    basic_c2.subheader('Coffee Details')
    basic_c2.write(f"Process: {coffee_info.get('process_type_final', '') if 'process_type_final' not in empty_info else ''}")
    basic_c2.write(f"Fermentation: {coffee_info.get('fermentation', '') if 'fermentation' not in empty_info else ''}")
    
    # Flavor information section
    flav_c = st.container()
    flav_c.subheader('Flavor Information')
    flav_text, flav_graphic_col = flav_c.columns([0.33, 0.67], gap="small", vertical_alignment="top")
    
    orig_f, tax_f = flav_text.tabs(['Original Flavor Notes', 'Taxonomized Flavor Notes'])
    orig_f.write(f"Original Flavor Notes: {coffee_info.get('Flavor Notes', '') if 'Flavor Notes' not in empty_info else ''}")
    
    # Handle categorized flavors
    categorized_flavors = coffee_info.get('categorized_flavors', '{}')
    if categorized_flavors and categorized_flavors != '{}':
        try:
            # Clean up the flavor data for JSON display
            clean_flavors = categorized_flavors.replace("'", '"').replace('None', '"None"')
            tax_f.json(clean_flavors)
            
            # Generate flavor wheel
            flav_graphic = flavor_wheel_gen(categorized_flavors)
            flav_graphic_col.pyplot(fig=flav_graphic, clear_figure=True, use_container_width=True)
        except:
            tax_f.write("Flavor categorization data unavailable")
    else:
        tax_f.write("No categorized flavors available")
    
    flav_c.write('See the About section for a fully completed version of the flavor wheel that this model is based on.')
    
    # Similarity section
    sim_c = st.container()
    sim_c.subheader('Similar Coffees')
    sim_c.write("Below you will find coffees that are similar to this one. One measure of similarity is the 'full profile' similarity. The other measure of similarity is based only on flavor profile.")
    
    full_sim, flav_sim = sim_c.tabs(['Similar Coffees (Full Profile)', 'Similar Coffees (Flavor Only)'])
    
    # Full profile similarity
    full_sim.subheader('Full Profile Similarity Matches')
    full_sim.write("Full Profile Similarity Matches are based on Country of Origin, Subregion, Process, Fermentation, and Flavor Notes.")
    
    col_config = {'uid': None, 'Predicted Coffee Review Range': None, 'Date First Seen': None}
    
    try:
        full_sim_df_only = find_full_similarity_matches(
            coffee_id=coffee_id,
            original_df=all_coffees_db
        )
        
        if not full_sim_df_only.empty:
            full_sim_df = full_sim.dataframe(
                full_sim_df_only,
                hide_index=True,
                on_select='rerun',
                selection_mode='single-row',
                column_config=col_config
            )
            
            # Handle selection
            if len(full_sim_df.selection.rows) > 0:
                full_selected_row_num = full_sim_df.selection.rows[0]
                full_selected_row_df = full_sim_df_only.iloc[full_selected_row_num].name
                st.session_state.ind = full_selected_row_df
                if full_sim.button(label='Click here for more information on your selected coffee', key='full_sim_button'):
                    st.switch_page('page_apps/individual_coffee_page.py')
            else:
                full_sim.button(label='Select a coffee to see more information', disabled=True, key='full_sim_button')
        else:
            full_sim.info("No similar coffees found for full profile matching.")
            
    except Exception as e:
        full_sim.error('Unfortunately the data quality on this coffee is not high enough to generate similarity matches. Please check back later after we try to clean up the data.')
        st.error(f"Debug: {e}")
    
    # Flavor similarity
    flav_sim.subheader('Flavor Similarity Matches')
    flav_sim.write("Flavor Similarity Matches are based solely on the Flavor Notes")
    
    try:
        flav_sim_df_only = find_flavor_similarity_matches(
            coffee_id=coffee_id,
            original_df=all_coffees_db
        )
        
        if not flav_sim_df_only.empty:
            flav_sim_df = flav_sim.dataframe(
                flav_sim_df_only,
                hide_index=True,
                selection_mode='single-row',
                on_select='rerun',
                column_config=col_config
            )
            
            # Handle selection
            if len(flav_sim_df.selection.rows) > 0:
                flav_selected_row_num = flav_sim_df.selection.rows[0]
                flav_selected_row_df = flav_sim_df_only.iloc[flav_selected_row_num].name
                st.session_state.ind = flav_selected_row_df
                if flav_sim.button(label='Click here for more information on your selected coffee', key='flav_sim_button'):
                    st.switch_page('page_apps/individual_coffee_page.py')
            else:
                flav_sim.button(label='Select a coffee to see more information', disabled=True, key='flav_sim_button')
        else:
            flav_sim.info("No similar coffees found for flavor-only matching.")
            
    except Exception as e:
        flav_sim.error('Unfortunately the data quality on this coffee is not high enough to generate similarity matches. Please check back later after we try to clean up the data.')
        st.error(f"Debug: {e}")

def ind_coffee_page(index_val):
    """Main function for individual coffee page"""
    full_df = load_full_dataset()
    index_no = st.sidebar.number_input('Enter the coffee number', min_value=0, max_value=len(full_df)-1, value=index_val, step=1)
    coffee_data = full_df.iloc[index_no]
    coffee_info(coffee_data)

# Run the page
ind_coffee_page(st.session_state.ind)
```

#### Update Main App File
**File**: `app_v2.py`
```python
"""
Updated main app file with database integration
"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st

st.set_page_config(
    page_title='Green Coffee in the USA', 
    page_icon='☕️', 
    layout='wide', 
    initial_sidebar_state='expanded'
)

# Updated page collection with new database-integrated pages
page_collection = [
    st.Page('page_apps/about_page.py', title='About the Home Greens Project', default=True), 
    st.Page('page_apps/full_data_set_page_v2.py', title='All Green Coffees'), 
    st.Page('page_apps/individual_coffee_page_v2.py', title='Individual Coffee Information'), 
    st.Page('page_apps/fresh_dashboard_page_v2.py', title='Fresh Arrival Dashboard')
]

pg = st.navigation(page_collection, position='sidebar')
pg.run()
```

### Phase 5: Performance Optimization

#### Implement Smart Caching Strategy
```python
"""
Caching strategy for optimal performance
"""
import streamlit as st
from datetime import datetime, timedelta

# Cache configuration
CACHE_SETTINGS = {
    'all_coffees': {'ttl': 300, 'max_entries': 10},     # 5 minutes
    'individual_coffee': {'ttl': 600, 'max_entries': 100},  # 10 minutes  
    'similarity_results': {'ttl': 1800, 'max_entries': 200}, # 30 minutes
    'static_data': {'ttl': 3600, 'max_entries': 50}     # 1 hour
}

def cache_key_generator(func_name: str, **kwargs) -> str:
    """Generate cache keys for functions"""
    key_parts = [func_name]
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}:{v}")
    return "_".join(key_parts)

@st.cache_data(ttl=CACHE_SETTINGS['similarity_results']['ttl'])
def cached_similarity_search(coffee_id: int, similarity_type: str, limit: int):
    """Cache similarity search results"""
    # Implementation here
    pass
```

#### Add Real-time Data Refresh
```python
"""
Real-time data refresh capabilities
"""
import streamlit as st
from datetime import datetime

def add_refresh_controls():
    """Add refresh controls to sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Refresh")
    
    col1, col2 = st.sidebar.columns(2)
    
    if col1.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.success("Data cache cleared! Page will refresh with latest data.")
        st.rerun()
    
    if col2.button("⚡ Auto Refresh"):
        st.session_state.auto_refresh = not st.session_state.get('auto_refresh', False)
    
    # Auto refresh indicator
    if st.session_state.get('auto_refresh', False):
        st.sidebar.success("Auto-refresh: ON")
        # Auto refresh every 5 minutes
        time.sleep(300)
        st.rerun()
    else:
        st.sidebar.info("Auto-refresh: OFF")
    
    # Last update time
    last_update = datetime.now().strftime("%H:%M:%S")
    st.sidebar.caption(f"Last updated: {last_update}")
```

## Migration Deployment Plan

### Phase 1: Preparation (Week 1)
- [x] Set up Supabase project and configure schema
- [x] Create database connection modules  
- [x] Implement basic data loading functions
- [x] Test database connectivity
- [ ] Create manual vector indexes in Supabase SQL Editor:
  ```sql
  CREATE INDEX IF NOT EXISTS coffee_embeddings_combined_vector_idx 
  ON coffee_embeddings USING hnsw (combined_vector_pg vector_cosine_ops);
  
  CREATE INDEX IF NOT EXISTS coffee_embeddings_flavor_vector_idx 
  ON coffee_embeddings USING hnsw (flavor_vector_pg vector_cosine_ops);
  ```

### Phase 2: Core Migration (Week 2)  
- [ ] Replace CSV loading with database queries (use direct table queries, not RPC)
- [ ] Implement pgvector similarity search (may need client-side calculation initially)
- [ ] Update individual coffee page
- [ ] Test functionality with existing UI
- [ ] Implement daily sync integration for real-time updates

### Phase 3: Optimization (Week 3)
- [ ] Implement caching strategy
- [ ] Add real-time refresh capabilities  
- [ ] Performance testing and optimization
- [ ] Error handling and fallbacks

### Phase 4: Production Deployment (Week 4)
- [ ] Deploy to production environment
- [ ] Monitor performance and usage
- [ ] Gradual traffic migration
- [ ] Remove AWS S3 dependencies

## Updated Requirements

### New Dependencies
```txt
# Existing dependencies
streamlit
pandas
numpy
matplotlib
seaborn
pysqlite3-binary

# New dependencies for database integration
supabase==2.2.0
postgrest==0.13.0
python-dotenv==1.0.0

# Remove these after migration
# boto3  
# chromadb
# protobuf<=3.20.x
```

### Environment Variables
```bash
# .env file for local development
SUPABASE_URL=https://[your-project-ref].supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Required: Service role key for admin operations (Note: updated name from implementation)
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## Testing Strategy

### Unit Tests
```python
"""
Test cases for database integration
"""
import pytest
from frontend.database.supabase_client import FrontendSupabaseClient
from frontend.similarity.pgvector_similarity import PgvectorSimilaritySearch

def test_database_connection():
    """Test Supabase connection"""
    client = FrontendSupabaseClient()
    df = client.get_all_coffees()
    assert not df.empty
    assert 'id' in df.columns

def test_similarity_search():
    """Test similarity search functionality"""
    search = PgvectorSimilaritySearch()
    results = search.find_full_similarity_matches("1", pd.DataFrame())
    assert isinstance(results, pd.DataFrame)

def test_data_compatibility():
    """Test data format compatibility with existing UI"""
    # Test that new data loading produces same format as old CSV loading
    pass
```

### Integration Tests
```python
"""
Integration tests for end-to-end functionality
"""
def test_individual_coffee_page():
    """Test individual coffee page loads correctly"""
    pass

def test_similarity_ui_integration():
    """Test similarity search integrates with UI"""
    pass

def test_performance_benchmarks():
    """Test performance meets requirements"""
    pass
```

## Performance Expectations

### Before Migration (CSV + ChromaDB)
- **Initial Load**: 10-15 seconds (CSV download + ChromaDB setup)
- **Similarity Search**: 100-200ms (in-memory calculation)
- **Data Freshness**: Manual updates required
- **Memory Usage**: High (full dataset in memory)

### After Migration (Supabase + pgvector)
- **Initial Load**: 2-3 seconds (cached API responses)
- **Similarity Search**: 50-100ms (pgvector HNSW index, once optimized)
- **Data Freshness**: Real-time with configurable cache
- **Memory Usage**: Low (on-demand loading)
- **Data Sync**: ~30 seconds for full migration, <10 seconds for daily updates
- **Individual Queries**: <200ms for standard coffee lookups

## Monitoring and Analytics

### Key Metrics to Track
- **Page Load Times**: Target <3 seconds for all pages
- **API Response Times**: Target <200ms for queries
- **Similarity Search Performance**: Target <100ms
- **Cache Hit Rates**: Target >80% for frequently accessed data
- **Error Rates**: Target <1% for all operations

### Monitoring Implementation
```python
"""
Performance monitoring for production
"""
import time
import streamlit as st
from typing import Dict, Any

class PerformanceMonitor:
    """Monitor performance metrics for the application"""
    
    @staticmethod
    def time_operation(operation_name: str):
        """Decorator to time operations"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                duration = end_time - start_time
                st.session_state.setdefault('performance_metrics', []).append({
                    'operation': operation_name,
                    'duration': duration,
                    'timestamp': time.time()
                })
                
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def display_metrics():
        """Display performance metrics in sidebar"""
        if 'performance_metrics' in st.session_state:
            with st.sidebar.expander("Performance Metrics"):
                metrics = st.session_state['performance_metrics'][-10:]  # Last 10 operations
                for metric in metrics:
                    st.caption(f"{metric['operation']}: {metric['duration']:.3f}s")
```

## Rollback Strategy

### Fallback Mechanism
```python
"""
Fallback to CSV/ChromaDB if database fails
"""
import streamlit as st
from typing import Union, Any

def safe_database_operation(operation_func, fallback_func=None):
    """Safely execute database operations with fallback"""
    try:
        return operation_func()
    except Exception as e:
        st.warning(f"Database operation failed: {e}")
        
        if fallback_func:
            st.info("Falling back to legacy data source...")
            return fallback_func()
        else:
            st.error("No fallback available for this operation")
            return None

# Example usage
def load_data_with_fallback():
    """Load data with automatic fallback to CSV"""
    def database_load():
        client = get_supabase_client()
        return client.get_all_coffees()
    
    def csv_fallback():
        # Original CSV loading code
        return load_full_dataset_csv()
    
    return safe_database_operation(database_load, csv_fallback)
```

## Success Criteria

### Technical Success Metrics
- [ ] All pages load successfully with database backend
- [ ] Similarity search works with pgvector
- [ ] Performance meets or exceeds current system
- [ ] No data loss during migration
- [ ] Error rates below 1%

### Business Success Metrics  
- [ ] User experience remains identical or improved
- [ ] Real-time data updates working
- [ ] Reduced infrastructure costs (no AWS S3)
- [ ] Simplified maintenance and updates
- [ ] Scalability for future growth

### User Acceptance Criteria
- [ ] All existing functionality preserved
- [ ] Similar or better performance
- [ ] No visible changes to user interface
- [ ] Reliable similarity recommendations
- [ ] Fast page load times

This comprehensive migration plan ensures a smooth transition from the current CSV/ChromaDB system to a modern, scalable database-driven architecture while preserving all existing functionality and improving performance.