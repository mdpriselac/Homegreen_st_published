"""
Updated individual coffee page using database backend
"""
import streamlit as st
import pandas as pd
from page_apps.modules.flavor_wheel_gen import flavor_wheel_gen
from frontend.data.database_loader import load_full_dataset, get_coffee_details, get_coffee_by_uid
from frontend.similarity.pgvector_similarity import find_full_similarity_matches, find_flavor_similarity_matches
from frontend.data.database_loader import load_all_coffees_df

if 'coffee_uid' not in st.session_state:
    # Initialize with the first coffee's uid from the dataset
    all_coffees = load_all_coffees_df()
    if not all_coffees.empty:
        st.session_state.coffee_uid = all_coffees.iloc[0]['uid']
    else:
        st.session_state.coffee_uid = '1'  # Fallback

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
        seller_info = detailed_info.get('sellers', {})
        seller_website = seller_info.get('homepage', '#') if seller_info else '#'
    except:
        coffee_url = '#'
        seller_website = '#'
    
    # Key columns for checking empty values
    key_cols = [
        'seller', 'Name', 'country_final', 'subregion_final', 'micro_final',
        'Flavor Notes', 'process_type', 'fermentation', 'first_date_observed'
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
    On this page we have information about [{coffee_info['Name']}]({coffee_url}) from [{coffee_info.get('Seller', coffee_info.get('seller', ''))}]({seller_website}). 
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
    basic_c1.write(f"Country: {coffee_info.get('Country', coffee_info.get('country_final', '')) if 'country_final' not in empty_info else ''}")
    
    region_value = coffee_info.get('Region(s)', coffee_info.get('subregion_final', ''))
    if region_value:
        region_clean = str(region_value).replace('[','').replace(']','').replace("'","").replace('"','')
        basic_c1.write(f"Region(s): {region_clean if 'subregion_final' not in empty_info else ''}")
    
    basic_c1.write(f"Micro Location: {coffee_info.get('Micro Location', coffee_info.get('micro_final', '')) if 'micro_final' not in empty_info else ''}")
    
    # Try to get altitude information from detailed data
    try:
        detailed_info = get_coffee_details(int(coffee_id))
        coffee_attrs = detailed_info.get('coffee_attributes', [])
        if coffee_attrs:
            attrs = coffee_attrs[0] if isinstance(coffee_attrs, list) else coffee_attrs
            altitude_low = attrs.get('altitude_low', '')
            altitude_high = attrs.get('altitude_high', '')
            basic_c1.write(f"Altitude Min (masl): {altitude_low}")
            basic_c1.write(f"Altitude Max (masl): {altitude_high}")
    except:
        pass
    
    # Coffee details
    basic_c2.subheader('Coffee Details')
    basic_c2.write(f"Process: {coffee_info.get('Process', coffee_info.get('process_type', '')) if 'process_type' not in empty_info else ''}")
    basic_c2.write(f"Fermentation: {coffee_info.get('Fermented?', coffee_info.get('fermentation', '')) if 'fermentation' not in empty_info else ''}")
    
    # Handle varietal information
    varietal_cleaned = coffee_info.get('Varietal Cleaned', '')
    varietal = coffee_info.get('Varietal', '')
    varietal_display = coffee_info.get('Varietal(s)', '')
    
    if varietal_display:
        varietal_clean = str(varietal_display).replace('[','').replace(']','').replace('"','').replace("'",'')
        basic_c2.write(f"Varietal(s): {varietal_clean}")
    elif varietal_cleaned or varietal:
        combined_varietal = (str(varietal_cleaned) + str(varietal)).replace('[','').replace(']','').replace('"','').replace("'",'')
        basic_c2.write(f"Varietal(s): {combined_varietal}")
    
    # Pricing information section
    pricing_c = st.container()
    pricing_c.subheader('Pricing Information')
    
    # Get pricing data from detailed coffee info
    try:
        detailed_info = get_coffee_details(int(coffee_id))
        coffee_attrs = detailed_info.get('coffee_attributes', [])
        
        if coffee_attrs:
            attrs = coffee_attrs[0] if isinstance(coffee_attrs, list) else coffee_attrs
            full_pricing_data = attrs.get('full_pricing_data', None)
            
            if full_pricing_data and len(full_pricing_data) > 0:
                try:
                    # Handle both string JSON and already parsed list
                    if isinstance(full_pricing_data, str):
                        import json
                        pricing_list = json.loads(full_pricing_data)
                    else:
                        pricing_list = full_pricing_data
                    
                    if pricing_list and isinstance(pricing_list, list) and len(pricing_list) > 0:
                        # Sort by pounds from lowest to highest
                        sorted_pricing = sorted(pricing_list, key=lambda x: float(x.get('pounds', x.get('size', 0))))
                        
                        # Create a nice table display
                        pricing_data = []
                        for item in sorted_pricing:
                            # Use 'pounds' field if available, otherwise 'size'
                            pounds = item.get('pounds', item.get('size', 'N/A'))
                            total_price = item.get('total_price', 'N/A')
                            price_per_lb = item.get('price_per_lb', 'N/A')
                            
                            pricing_data.append({
                                'Pounds': f"{pounds} LB" if isinstance(pounds, (int, float)) else str(pounds),
                                'Total Price': f"${total_price:.2f}" if isinstance(total_price, (int, float)) else 'N/A',
                                'Price/LB': f"${price_per_lb:.2f}" if isinstance(price_per_lb, (int, float)) else 'N/A'
                            })
                        
                        # Display as a styled table
                        import pandas as pd
                        pricing_df = pd.DataFrame(pricing_data)
                        
                        # Use HTML table for better styling
                        html_table = pricing_df.to_html(index=False, escape=False, classes='pricing-table')
                        pricing_c.markdown(html_table, unsafe_allow_html=True)
                        
                        # Add some custom CSS for better styling
                        pricing_c.markdown("""
                        <style>
                        .pricing-table {
                            border-collapse: collapse;
                            margin: 20px 0;
                            font-size: 14px;
                            width: 100%;
                        }
                        .pricing-table th, .pricing-table td {
                            border: 1px solid #ddd;
                            padding: 8px;
                            text-align: center;
                        }
                        .pricing-table th {
                            background-color: #f2f2f2;
                            font-weight: bold;
                        }
                        .pricing-table tr:nth-child(even) {
                            background-color: #f9f9f9;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                    else:
                        pricing_c.write("No pricing data available for this coffee.")
                except (ValueError, TypeError, KeyError) as e:
                    pricing_c.write("No pricing data available for this coffee.")
            else:
                pricing_c.write("No pricing data available for this coffee.")
        else:
            pricing_c.write("No pricing data available for this coffee.")
    except Exception as e:
        pricing_c.write("No pricing data available for this coffee.")
    
    # Flavor information section
    flav_c = st.container()
    flav_c.subheader('Flavor Information')
    flav_text, flav_graphic_col = flav_c.columns([0.33, 0.67], gap="small", vertical_alignment="top")
    
    orig_f, tax_f = flav_text.tabs(['Original Flavor Notes', 'Taxonomized Flavor Notes'])
    
    flavor_notes = coffee_info.get('Flavor Notes', '')
    orig_f.write(f"{flavor_notes if flavor_notes and 'Flavor Notes' not in empty_info else ''}")
    
    # Handle categorized flavors
    categorized_flavors = coffee_info.get('categorized_flavors', '{}')
    
    if categorized_flavors and categorized_flavors != '{}' and categorized_flavors != '[]':
        try:
            # The flavors come as a string representation of a list
            import ast
            if isinstance(categorized_flavors, str):
                # Try to parse the string as a Python literal
                parsed_flavors = ast.literal_eval(categorized_flavors)
                tax_f.json(parsed_flavors)
                
                # Generate flavor wheel - pass the original string format
                flav_graphic = flavor_wheel_gen(categorized_flavors)
                flav_graphic_col.pyplot(fig=flav_graphic, clear_figure=True, use_container_width=True)
            else:
                tax_f.json(categorized_flavors)
                flav_graphic = flavor_wheel_gen(str(categorized_flavors))
                flav_graphic_col.pyplot(fig=flav_graphic, clear_figure=True, use_container_width=True)
        except Exception:
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
    
    # Initialize selection state to prevent first-click disruption
    if 'full_sim_initialized' not in st.session_state:
        st.session_state.full_sim_initialized = True
    if 'flav_sim_initialized' not in st.session_state:
        st.session_state.flav_sim_initialized = True
    
    try:
        # Load full dataset for similarity matching
        full_dataset = all_coffees_db  # Use the already loaded formatted data
        
        full_sim_df_only = find_full_similarity_matches(
            coffee_id=coffee_id,
            original_df=full_dataset
        )
        
        if not full_sim_df_only.empty:
            # Unfortunately, dataframe selection doesn't work properly in forms
            # So we'll use on_select='rerun' but try to minimize disruption
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
                selected_coffee_uid = full_sim_df_only.iloc[full_selected_row_num]['uid']
                selected_name = full_sim_df_only.iloc[full_selected_row_num]['Name']
                if full_sim.button(label=f'Click here for more information on: {selected_name}', key='full_sim_button'):
                    st.session_state.coffee_uid = selected_coffee_uid
                    st.rerun()
            else:
                full_sim.button(label='Select a coffee to see more information', disabled=True, key='full_sim_button')
        else:
            full_sim.info("No similar coffees found for full profile matching.")
            
    except Exception as e:
        full_sim.error(f'Error loading similarity matches: {str(e)}')
    
    # Flavor similarity
    flav_sim.subheader('Flavor Similarity Matches')
    flav_sim.write("Flavor Similarity Matches are based solely on the Flavor Notes")
    
    try:
        # Load full dataset for similarity matching  
        full_dataset = all_coffees_db  # Use the already loaded formatted data
        
        flav_sim_df_only = find_flavor_similarity_matches(
            coffee_id=coffee_id,
            original_df=full_dataset
        )
        
        if not flav_sim_df_only.empty:
            # Unfortunately, dataframe selection doesn't work properly in forms
            # So we'll use on_select='rerun' but try to minimize disruption
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
                selected_coffee_uid = flav_sim_df_only.iloc[flav_selected_row_num]['uid']
                selected_name = flav_sim_df_only.iloc[flav_selected_row_num]['Name']
                if flav_sim.button(label=f'Click here for more information on: {selected_name}', key='flav_sim_button'):
                    st.session_state.coffee_uid = selected_coffee_uid
                    st.rerun()
            else:
                flav_sim.button(label='Select a coffee to see more information', disabled=True, key='flav_sim_button')
        else:
            flav_sim.info("No similar coffees found for flavor-only matching.")
            
    except Exception as e:
        flav_sim.error(f'Error loading similarity matches: {str(e)}')

def ind_coffee_page(coffee_uid):
    """Main function for individual coffee page"""
    # Get coffee data by uid
    coffee_data = get_coffee_by_uid(coffee_uid)
    
    if coffee_data.empty:
        st.error(f"Coffee with ID {coffee_uid} not found.")
        return
    
    # Add a selector for navigating between coffees
    all_coffees = load_all_coffees_df()
    coffee_names_with_uid = [f"{row['Name']} ({row['uid']})" for _, row in all_coffees.iterrows()]
    
    # Find current coffee in the list, or add it if not found
    current_coffee_display = f"{coffee_data['Name']} ({coffee_uid})"
    if current_coffee_display not in coffee_names_with_uid:
        # Coffee is not in the initial 50, add it to the list
        coffee_names_with_uid.insert(0, current_coffee_display)
        current_index = 0
    else:
        current_index = coffee_names_with_uid.index(current_coffee_display)
    
    selected_coffee = st.sidebar.selectbox(
        'Select a coffee',
        coffee_names_with_uid,
        index=current_index,
        key='coffee_selector'
    )
    
    # Extract uid from selection
    selected_uid = selected_coffee.split('(')[-1].rstrip(')')
    if selected_uid != coffee_uid:
        st.session_state.coffee_uid = selected_uid
        st.rerun()
    
    coffee_info(coffee_data)

# Run the page
ind_coffee_page(st.session_state.coffee_uid)