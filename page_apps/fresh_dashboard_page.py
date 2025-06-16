"""
Updated fresh dashboard page using database backend
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from frontend.data.database_loader import load_full_dataset

@st.cache_data(ttl=300)
def load_dashboard_data():
    """Load data for dashboard with caching"""
    return load_full_dataset()

def fresh_dashboard():
    full_df = load_dashboard_data()
    
    # Handle country data
    country_col = 'country_final' if 'country_final' in full_df.columns else 'Country'
    if country_col not in full_df.columns:
        st.error("Country information not available in the dataset")
        return
    
    country_proportion = full_df[country_col].value_counts(normalize=True)
    month_ago = datetime.today() - timedelta(days=30)
    
    # Handle date conversion
    date_col = 'first_date_observed' if 'first_date_observed' in full_df.columns else 'Date First Seen'
    if date_col not in full_df.columns:
        st.error("Date information not available in the dataset")
        return
    
    # Convert date column to datetime
    try:
        full_df['first_date_observed_dto'] = pd.to_datetime(full_df[date_col])
    except Exception as e:
        st.error(f"Error converting dates: {e}")
        return
    
    month_filter = full_df['first_date_observed_dto'] > month_ago
    
    this_month_country_counts = full_df[month_filter][country_col].value_counts(normalize=False).dropna()
    this_month_country_proportion = full_df[month_filter][country_col].value_counts(normalize=True).dropna()
    
    # Calculate freshness rating
    try:
        notably_new_countries = this_month_country_proportion / country_proportion
        fresh_szn = notably_new_countries.dropna().sort_values(ascending=False) * this_month_country_counts.dropna()
        fresh_szn.name = "Freshness Rating"
    except Exception as e:
        st.error(f"Error calculating freshness ratings: {e}")
        fresh_szn = pd.Series(dtype=float)
    
    st.header('Recent Arrival Dashboard')
    st.subheader('Number of New Coffees Arrived by Country in the last 30 days')
    
    if not this_month_country_counts.empty:
        st.dataframe(this_month_country_counts.sort_values(ascending=False))
    else:
        st.info("No new coffees arrived in the last 30 days")
    
    st.subheader('Fresh Coffee Season?')
    st.markdown("""The Fresh metric is an attempt to estimate whether a crop of coffees is currently arriving.
    
We're combining three pieces of information:
- The proportion of all coffees in the data set that come from a country. Call this the country's baseline proportion.
- The proportion of coffees arriving in the last 30 days that come from a country. Call this the country's recent proportion.
- The raw count of coffees arriving in the last 30 days from a country.

We're combining these pieces of information by multiplying the country's raw count of coffees arriving in the last 30 days by the ratio of the country's recent proportion to its baseline proportion. So, when a country's Fresh metric is high, coffees are arriving from that country at an unusually high rate. If it's very low, then though a new coffee or two has arrived, it's actually a slower arrival proportion than the country normally makes up.""")
    
    if not fresh_szn.empty:
        st.dataframe(fresh_szn.sort_values(ascending=False))
    else:
        st.info("No freshness data available for the last 30 days")
    
    # Add refresh controls
    st.markdown("---")
    if st.button("🔄 Refresh Dashboard Data"):
        st.cache_data.clear()
        st.success("Dashboard data refreshed!")
        st.rerun()

fresh_dashboard()