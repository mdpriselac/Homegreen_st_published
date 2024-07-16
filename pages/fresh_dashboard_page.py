import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from pages.individual_coffee_page import load_full_dataset

def fresh_dashboard():
    full_df = load_full_dataset()
    country_proportion = full_df['country_final'].value_counts(normalize=True)
    month_ago = datetime.today() - timedelta(days=30)
    full_df['first_date_observed_dto'] = pd.to_datetime(full_df['first_date_observed'])
    month_filter = full_df['first_date_observed_dto'] > month_ago
    
    this_month_country_counts = full_df[month_filter]['country_final'].value_counts(normalize=False).dropna()
    this_month_country_proportion = full_df[month_filter]['country_final'].value_counts(normalize=True).dropna()
    notably_new_countries = this_month_country_proportion/country_proportion
    fresh_szn = notably_new_countries.dropna().sort_values(ascending=False) * this_month_country_counts.dropna()
    fresh_szn.name = "Freshness Rating"
    
    st.header('Recent Arrival Dashboard')
    st.subheader('Number of New Coffees Arrived by Country in the last 30 days')
    st.dataframe(this_month_country_counts.sort_values(ascending=False))
    st.subheader('Fresh Coffee Season?')
    st.markdown("The Fresh metric is an attempt to estimate whether a crop of coffees is currently arriving.\n We're combining three pieces of information: \n- The proportion of all coffees in the data set that come from a country. Call this the country's baseline proportion. \n- The proportion of coffees arriving in the last 30 days that come from a country. Call the this the country's recent proportion. \n- The raw count of coffees arriving in the last 30 days from a country.\n\nWe're combining these pieces of information by multiplying the country's raw count of coffees arriving in the last 30 days by the raio of the country's recent proportion to its baseline proportion. So, when a country's Fresh metric coffees are arriving from that country at an unusually high rate. If it's very low, then though a new coffee or two as arrived, it's acutlly a slower arrival proportion than the country normallly makes up.")
    st.dataframe(fresh_szn.sort_values(ascending=False))
    
fresh_dashboard()