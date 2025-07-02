"""
Updated full dataset page using database backend
"""
import streamlit as st
import pandas as pd
from frontend.data.database_loader import load_all_coffees_df

def full_data_page():
    st.header('Data Overview')
    
    # Add sidebar controls
    st.sidebar.markdown("### Filters")
    
    # Add checkbox for including expired coffees
    include_expired = st.sidebar.checkbox("Include expired coffees that are no longer for sale", value=False)
    st.sidebar.markdown("---")
    
    # Load data based on checkbox
    df_filtered = load_all_coffees_df(include_expired=include_expired)
    
    # Show coffee counts
    if include_expired and 'Expired?' in df_filtered.columns:
        total_coffees = len(df_filtered)
        expired_count = df_filtered['Expired?'].sum()
        active_count = total_coffees - expired_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Coffees", total_coffees)
        with col2:
            st.metric("Active Coffees", active_count)
        with col3:
            st.metric("Expired Coffees", expired_count)
    
    if include_expired:
        st.write("Below you will find a dataset of coffees in our database. By default we only display coffees we believe are actively for sale. To see all coffees we've observed, including coffees we believe are no longer for sale, click the 'Include expired coffees that are no longer for sale' checkbox to the left.  \n\n- The list is sorted by how recently we've spotted them for sale. \n\n- You can filter the data by selecting values in the sidebar. The data will update as you select filters. \n\n- You can sort the data by clicking on the column headers. \n\n- You can resize the columns by dragging them and resize the data frame by dragging the bottom right corner. \n\n- Clicking on a box at the far lefthand side of a row will select that row. Once you've selected a coffee by checking its box, clicking the button below will take you to a page with more information on the selected coffee.")
    else:
        st.write("You are currently viewing all the coffees we have in our database, whether they are for sale or not. To view only coffees we believe to be for sale uncheck the 'include expired coffees tat are no longer for sale' box to the left.  \n\n- The list is sorted by how recently we've spotted them for sale. \n\n- You can filter the data by selecting values in the sidebar. The data will update as you select filters. \n\n- You can sort the data by clicking on the column headers. \n\n- You can resize the columns by dragging them and resize the data frame by dragging the bottom right corner. \n\n- Clicking on a box at the far lefthand side of a row will select that row. Once you've selected a coffee by checking its box, clicking the button below will take you to a page with more information on the selected coffee.")
    
    # Add a info message about pricing columns and data
    st.info("Price information is now available in three columns: 'Avg \$/LB', 'Min \$/LB', and 'Max \$/LB'. If a coffee does not have a price, it will show as 'N/A'.\n\n The price information is collected only when the coffee information is first observed so it may be out of date. \n\n When you select a coffee you can see the price information for each quantity that it is sold in by a seller. Here we report the minimum (i.e. price for the largest quantity), maximum (price for the smallest quantity), and average price per pound. Prices may or may not include shipping depending on the seller. Take this information only as a rough guide to the actual price.")

    # Add filters
    columns_to_filter = ['Country', 'Seller','Process']
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
    
    # Handle pricing columns - format as currency and handle nulls
    pricing_cols = ['Avg $/LB', 'Min $/LB', 'Max $/LB']
    for col in pricing_cols:
        if col in df_filtered.columns:
            # Convert to numeric first, then format
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
            # Format as currency string, replace NaN with N/A
            #df_filtered[col] = df_filtered[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) and x > 0 else 'N/A')
    
    col_config = {'uid': None}
    
    # Define column order, using only columns that exist
    desired_order = [
        'Seller', 'Name', 'Country', 'Region(s)', 'Micro Location', 
        'Avg $/LB', 'Min $/LB', 'Max $/LB',
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

# Run the main function
full_data_page()