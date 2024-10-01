import streamlit as st
import pandas as pd
from page_apps.modules.flavor_wheel_gen import flavor_wheel_gen
from data_create import load_full_dataset,flav_only_conn,full_conn
from page_apps.modules.similarity_db_tools import find_full_similarity_matches, find_flavor_similarity_matches
from page_apps.full_data_set_page import load_all_coffees_df

if 'ind' not in st.session_state:
    st.session_state.ind = 0

all_coffees_db = load_all_coffees_df()

def coffee_info(coffee_data):
    coffee_info = coffee_data
    coffee_id = coffee_info['uid']
    st.title(f"{coffee_info['Name']}")
    
    key_cols = ['seller','Name','country_final','subregion_final','micro_final','Flavor Notes','process_type','Varietal Cleaned','Score','first_date_observed','expired']
    empty_info = [k for k in key_cols if coffee_info[k]=='' or str(coffee_info[k]).lower().strip()=='none' or str(coffee_info[k]).lower().strip()=='nan' or str(coffee_info[k]).strip().lower()=='unknown']
    
    # Let's do a text block introducing the information on this page
    # In the text block let's have a link to the coffee's page on the seller's website
    intro_md_block = f'''On this page we have information about [{coffee_info['Name']}]({coffee_info['url']}) from [{coffee_info['seller']}]({coffee_info['seller_website']}). More reliable information about the coffee can be found at the original website. \n\nBelow you will find information that we were able to collect and process about this coffee. Because an LLM is involved in this process, there is some possibility of error and fabrication. Double check the information at the original website before acting on it.'''
    st.markdown(intro_md_block)
    
    # Let's do a container with two columns
    ## Column 1 is location information: Country, Subregion, Microregion, Alitude
    ## Column 2 is process, fermentation, varietal, 
    basic_c = st.container()
    basic_c.subheader('Basic Information')
    basic_c1, basic_c2 = basic_c.columns(2)
    basic_c1.subheader('Location Information')
    basic_c2.subheader('Coffee Details')
    basic_c1.write(f"Country: {coffee_info['country_final'] if 'country_final' not in empty_info else ''}")
    basic_c1.write(f'''Region(s): {coffee_info['subregion_final'].replace('[','').replace(']','').replace("'",'').replace('"','') if 'country_final' not in empty_info else ''}''')
    basic_c1.write(f"Micro Location Information: {coffee_info['micro_final']if 'country_final' not in empty_info else ''}")
    basic_c1.write(f"Altitude Min (masl): {coffee_info['altitude_low'] if 'altitude_low' not in empty_info else ''}")
    basic_c1.write(f"Altitude Max (masl): {coffee_info['altitude_high'] if 'altitude_high' not in empty_info else ''}")
    basic_c2.write(f"Process: {coffee_info['process_type'] if 'process_type' not in empty_info else ''}")
    basic_c2.write(f"Fermentation: {coffee_info['fermentation'] if 'fermentation' not in empty_info else ''}")
    basic_c2.write(f'''Varietal(s): {coffee_info['Varietal Cleaned'].replace('[','').replace(']','').replace('"','').replace("'",'') if 'Varietal Cleaned' not in empty_info else ''}''')
    

    # Finally, let's do a final container with three tabs
    ## Tab 1 is the raw flavor notes
    ## Tab 2 is the taxonomized flavors
    ## Tab 3 is the flavor graphic
    
    flav_c = st.container()
    flav_c.subheader('Flavor Information')
    flav_text, flav_graphic_col = flav_c.columns([.33,.67],gap="small",vertical_alignment="top")

    orig_f,tax_f = flav_text.tabs(['Original Flavor Notes', 'Taxonomized Flavor Notes'])
    orig_f.write(f"Original Flavor Notes: {coffee_info['Flavor Notes'] if 'Flavor Notes' not in empty_info else ''}")
    tax_f.json(f'''{coffee_info['categorized_flavors'].replace("'",'"').replace('None','"None"')}''')
    #graphic
    
    flav_graphic = flavor_wheel_gen(coffee_info['categorized_flavors'])
    flav_graphic_col.pyplot(fig=flav_graphic, clear_figure=True, use_container_width=True)
    flav_c.write('See the About section for a fully completed version of the flavor wheel that this model is based on.')
    
    #Let's do a final container with the similarity information. Two tabs in the container, one for full sim and one for flavor sim
    sim_c = st.container()
    sim_c.subheader('Similar Coffees')
    sim_c.write("Below you will find coffees that are similar to this one. One measure of similarity is the 'full profile' similarity. The other measure of similarity is based only on flavor profile. It can be interesting to find a coffee where the purely flavor based most similar coffees are from a different country or have a different processing method (e.g. that a coffee from Indonesia best matches coffees from East Africa when only considering flavor). That suggests you're looking at a somewhat unique coffee from the origin!")
    full_sim,flav_sim = sim_c.columns(2)
    full_sim.subheader('Full Profile Similarity Matches')
    full_sim.write("Full Profile Similarity Matches are based on Country of Origin, Subregion, Process, Fermentation, and Flavor Notes.")
    flav_sim.subheader('Flavor Similarity Matches')
    flav_sim.write("Flavor Similarity Matches are based solely on the Flavor Notes")
    col_config = {'uid':None,'Predicted Coffee Review Range':None,'Date First Seen':None,}
    full_sim_df_only = find_full_similarity_matches(coffee_id=coffee_id,
                                                    original_df=all_coffees_db,
                                                    full_coll_conn=full_conn)
    full_sim_df = full_sim.dataframe(full_sim_df_only,
                                     hide_index=True,
                                     on_select='rerun',
                                     selection_mode='single-row',
                                     column_config=col_config)
    flav_sim_df_only = find_flavor_similarity_matches(coffee_id=coffee_id,
                                                      original_df=all_coffees_db,
                                                      flav_coll_conn=flav_only_conn)
    flav_sim_df = flav_sim.dataframe(flav_sim_df_only,
                                     hide_index=True,
                                     selection_mode='single-row',
                                     on_select='rerun',
                                     column_config=col_config)
    #make full_sim_df clickable
    if len(full_sim_df.selection.rows) > 0:
        full_selected_row_num = full_sim_df.selection.rows[0]
        full_selected_row_df = full_sim_df_only.iloc[full_selected_row_num].name
        st.session_state.ind = full_selected_row_df
        if full_sim.button(label='Click here for more information on your selected coffee',key='full_sim_button'):
            st.switch_page('page_apps/individual_coffee_page.py')
    else:
        full_sim.button(label='Select a coffee to see more information',disabled=True,key='full_sim_button')
    
    #make flav_sim_df clickable
    if len(flav_sim_df.selection.rows) > 0:
        flav_selected_row_num = flav_sim_df.selection.rows[0]
        flav_selected_row_df = flav_sim_df_only.iloc[flav_selected_row_num].name
        st.session_state.ind = flav_selected_row_df
        if flav_sim.button(label='Click here for more information on your selected coffee',key='flav_sim_button'):
            st.switch_page('page_apps/individual_coffee_page.py')
    else:
        flav_sim.button(label='Select a coffee to see more information',disabled=True,key='flav_sim_button')
    
    

def ind_coffee_page(index_val):
    full_df = load_full_dataset()
    index_no = st.sidebar.number_input('Enter the coffee number', min_value=0, max_value=len(full_df)-1, value=index_val, step=1)
    coffee_data = full_df.iloc[index_no]
    coffee_info(coffee_data)

ind_coffee_page(st.session_state.ind)