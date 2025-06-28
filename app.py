"""
Coffee Market Streamlit App with Supabase Database Integration
"""
import streamlit as st

st.set_page_config(
    page_title='Green Coffee in the USA', 
    page_icon='☕️', 
    layout='wide', 
    initial_sidebar_state='expanded'
)

# Database-integrated pages
page_collection = [
    st.Page('page_apps/landing.py', title='Dashboard', default=True, icon='📊'),
    st.Page('page_apps/full_data_set_page.py', title='All Green Coffees', icon='☕'), 
    st.Page('page_apps/individual_coffee_page.py', title='Individual Coffee Information', icon='🔍'),
    st.Page('page_apps/analytics_page.py', title='Flavor Analytics', icon='🔬'),
    st.Page('page_apps/about_page.py', title='About the Home Greens Project', icon='ℹ️')
]

pg = st.navigation(page_collection, position='sidebar')
pg.run()