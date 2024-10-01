import streamlit as st

st.set_page_config(page_title='Green Coffee in the USA', page_icon='☕️', layout='wide', initial_sidebar_state='expanded')

page_collection = [st.Page('page_apps/about_page.py',title='About the Home Greens Project',default=True), 
         st.Page('page_apps/full_data_set_page.py',title='All Green Coffees'), 
         st.Page('page_apps/individual_coffee_page.py',title='Individual Coffee Information'), 
         st.Page('page_apps/fresh_dashboard_page.py',title='Fresh Arrival Dashboard')]


pg = st.navigation(page_collection,position='sidebar')
pg.run()