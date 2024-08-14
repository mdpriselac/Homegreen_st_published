import streamlit as st
import pandas as pd
from pages.modules.flavor_wheel_gen import flavor_wheel_gen
from pages.modules.df_location import s3_client, bucket, db_path

@st.cache_data
def load_full_dataset():
    obj = s3_client.get_object(Bucket=bucket, Key=db_path)
    in_df = pd.read_csv(obj['Body'],index_col=0)
    out_df = in_df.copy()
    out_df.reset_index(inplace=True)
    out_df.drop(columns=['index'],inplace=True)
    return out_df


if 'ind' not in st.session_state:
    st.session_state.ind = 0

def coffee_info(coffee_data):
    coffee_info = coffee_data
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

def ind_coffee_page(index_val):
    full_df = load_full_dataset()
    index_no = st.sidebar.number_input('Enter the coffee number', min_value=0, max_value=len(full_df)-1, value=index_val, step=1)
    coffee_data = full_df.iloc[index_no]
    coffee_info(coffee_data)

ind_coffee_page(st.session_state.ind)