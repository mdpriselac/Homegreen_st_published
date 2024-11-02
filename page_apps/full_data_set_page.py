import streamlit as st
import pandas as pd
#from page_apps.modules.df_location import s3_client, bucket, db_path
from data_create import load_full_dataset
df_display_cols = ['seller','uid','Name','country_final','subregion_final','micro_final','Flavor Notes','process_type','fermentation','Varietal Cleaned','first_date_observed','expired','quality_prediction']


@st.cache_data
def load_all_coffees_df():
    in_df = load_full_dataset()
    out_df = in_df[df_display_cols].copy()
    #merge 'Varietal Cleaned and 'Varietal' columns into 'Varietal(s)' column
    out_df['Varietal(s)'] = out_df['Varietal Cleaned'].fillna(out_df['Varietal'])
    col_renames = {'country_final':'Country',
                   'seller':'Seller',
                   'subregion_final':'Region(s)',
                   'micro_final': 'Micro Location',
                   'process_type': 'Process',
                   'fermentation':'Fermented?',
                   'first_date_observed':'Date First Seen',
                   'expired':'Expired?',
                   'quality_prediction':'Predicted Coffee Review Range'}
    out_df.rename(columns=col_renames, inplace=True)
    out_df.reset_index(inplace=True)
    out_df.drop(columns=['index'],inplace=True)
    return out_df

def full_data_page():
    df_filtered = load_all_coffees_df()
    st.header('Data Overview')
    st.write("Below you will find the full data set. You can filter the data by selecting values in the sidebar. The data will update as you select filters. You can sort the data by clicking on the column headers. You can resize the columns by dragging them and resize the data frame by dragging the bottom right corner. Clicking on the far lefthand side of a row will select that row. Clicking the button below will take you to a page with more information on the selected coffee.")

    # Add filters
    columns_to_filter = ['Country', 'Seller','Process','Expired?']
    filters = {}
    for column in columns_to_filter:
        filters[column] = st.sidebar.multiselect(f'{column}', df_filtered[column].unique())
    
    # Apply filters, the data needs to update as a filter is selected    
    for column, values in filters.items():
        if values:
            df_filtered = df_filtered[df_filtered[column].isin(values)]
    #sort by 'Date First Seen' column by default
    df_filtered = df_filtered.sort_values(by='Date First Seen',ascending=False)
    col_config = {'uid':None}
    selectable = st.dataframe(df_filtered,
                              hide_index=True,
                              selection_mode='single-row',
                              on_select='rerun',
                              column_config=col_config,
                              column_order=['Seller','Name','Country','Region(s)','Micro Location','Flavor Notes','Varietal(s)','Process','Fermented?','Date First Seen','Expired?','Predicted Coffee Review Range'])
    
    if len(selectable.selection.rows) > 0:
        selected_row_df = df_filtered.iloc[selectable.selection.rows[0]].name
        st.session_state.ind = selected_row_df
        if st.button(label='Click here for more information on your selected coffee'):
            st.switch_page('page_apps/individual_coffee_page.py')
    else:
        st.button(label='Select a coffee to see more information',disabled=True)

full_data_page()