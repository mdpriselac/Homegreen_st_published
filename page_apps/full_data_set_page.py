"""
Updated full dataset page using database backend
"""
import streamlit as st
import pandas as pd
from frontend.data.database_loader import load_all_coffees_df

@st.cache_data(ttl=300)
def load_all_coffees_df_cached():
    """Load data with caching"""
    return load_all_coffees_df()

def full_data_page():
    df_filtered = load_all_coffees_df_cached()
    st.header('Data Overview')
    st.write("Below you will find the full data set. You can filter the data by selecting values in the sidebar. The data will update as you select filters. You can sort the data by clicking on the column headers. You can resize the columns by dragging them and resize the data frame by dragging the bottom right corner. Clicking on the far lefthand side of a row will select that row. Clicking the button below will take you to a page with more information on the selected coffee.")

    # Add filters
    columns_to_filter = ['Country', 'Seller','Process','Expired?']
    filters = {}
    
    for column in columns_to_filter:
        if column in df_filtered.columns:
            unique_values = df_filtered[column].dropna().unique()
            if len(unique_values) > 0:
                filters[column] = st.sidebar.multiselect(f'{column}', unique_values)
    
    # Apply filters, the data needs to update as a filter is selected    
    for column, values in filters.items():
        if values and column in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[column].isin(values)]
    
    # Sort by 'Date First Seen' column by default if it exists
    if 'Date First Seen' in df_filtered.columns:
        df_filtered = df_filtered.sort_values(by='Date First Seen', ascending=False)
    
    col_config = {'uid': None}
    
    # Define column order, using only columns that exist
    desired_order = [
        'Seller', 'Name', 'Country', 'Region(s)', 'Micro Location', 
        'Flavor Notes', 'Varietal(s)', 'Process', 'Fermented?', 
        'Date First Seen', 'Expired?'
    ]
    
    available_columns = [col for col in desired_order if col in df_filtered.columns]
    
    selectable = st.dataframe(
        df_filtered,
        hide_index=True,
        selection_mode='single-row',
        on_select='rerun',
        column_config=col_config,
        column_order=available_columns
    )
    
    if len(selectable.selection.rows) > 0:
        selected_row_index = selectable.selection.rows[0]
        selected_coffee_uid = df_filtered.iloc[selected_row_index]['uid']
        st.session_state.coffee_uid = selected_coffee_uid
        if st.button(label='Click here for more information on your selected coffee'):
            st.switch_page('page_apps/individual_coffee_page.py')
    else:
        st.button(label='Select a coffee to see more information', disabled=True)

# Add refresh controls to sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Data Refresh")

col1, col2 = st.sidebar.columns(2)

if col1.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.success("Data cache cleared! Page will refresh with latest data.")
    st.rerun()

# Last update indicator
from datetime import datetime
last_update = datetime.now().strftime("%H:%M:%S")
st.sidebar.caption(f"Last updated: {last_update}")

full_data_page()