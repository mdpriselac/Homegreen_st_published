import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from frontend.database.supabase_client import get_supabase_client

st.set_page_config(
    page_title="Home Greens Coffee Dashboard",
    page_icon="☕",
    layout="wide"
)

@st.cache_data(ttl=3600)
def get_dashboard_metrics():
    """Get dashboard metrics using Supabase query builder"""
    client = get_supabase_client()
    
    try:
        # Get active coffee count
        active_result = client.client.table('coffees').select('id', count='exact').eq('is_active', True).execute()
        active_count = active_result.count
        
        # Get all coffees with attributes for manual aggregation
        # This is less efficient but works without raw SQL
        all_coffees = client.client.table('coffees').select(
            'id, is_active, first_observed, sellers(name), coffee_attributes(country_final)'
        ).limit(2000).execute()
        
        # Process data for aggregations
        country_counts = {}
        seller_counts = {}
        total_country_counts = {}
        recent_country_counts = {}
        
        thirty_days_ago = datetime.now() - timedelta(days=30)
        total_all_count = 0
        recent_all_count = 0
        
        for coffee in all_coffees.data:
            # Extract country - handle both dict and list formats
            country = 'Unknown'
            coffee_attrs = coffee.get('coffee_attributes')
            if coffee_attrs:
                if isinstance(coffee_attrs, list) and len(coffee_attrs) > 0:
                    country = coffee_attrs[0].get('country_final', 'Unknown')
                elif isinstance(coffee_attrs, dict):
                    country = coffee_attrs.get('country_final', 'Unknown')
            
            # Extract seller - handle both dict and list formats
            seller = 'Unknown'
            seller_info = coffee.get('sellers')
            if seller_info:
                if isinstance(seller_info, list) and len(seller_info) > 0:
                    seller = seller_info[0].get('name', 'Unknown')
                elif isinstance(seller_info, dict):
                    seller = seller_info.get('name', 'Unknown')
            
            # Count total by country
            if country != 'Unknown':
                total_country_counts[country] = total_country_counts.get(country, 0) + 1
                total_all_count += 1
            
            # Count recent by country
            first_observed_str = coffee.get('first_observed')
            if first_observed_str:
                try:
                    first_observed = pd.to_datetime(first_observed_str)
                    # Make both datetimes timezone-naive for comparison
                    if first_observed.tz is not None:
                        first_observed = first_observed.tz_localize(None)
                    thirty_days_ago_naive = thirty_days_ago.replace(tzinfo=None)
                    
                    if country != 'Unknown' and first_observed >= thirty_days_ago_naive:
                        recent_country_counts[country] = recent_country_counts.get(country, 0) + 1
                        recent_all_count += 1
                except (ValueError, TypeError) as e:
                    # Skip invalid dates
                    continue
            
            # Count active by country and seller
            if coffee.get('is_active', False):
                if country != 'Unknown':
                    country_counts[country] = country_counts.get(country, 0) + 1
                if seller != 'Unknown':
                    seller_counts[seller] = seller_counts.get(seller, 0) + 1
        
        # Convert to sorted lists, filtering out None/Unknown countries
        country_counts_list = [{'country_final': country, 'count': count} 
                              for country, count in sorted(country_counts.items(), key=lambda x: x[1], reverse=True)
                              if country and country != 'Unknown'][:15]
        
        seller_counts_list = [{'name': seller, 'count': count} 
                             for seller, count in sorted(seller_counts.items(), key=lambda x: x[1], reverse=True)]
        
        total_country_counts_list = [{'country_final': country, 'total_count': count} 
                                    for country, count in total_country_counts.items()]
        
        recent_country_counts_list = [{'country_final': country, 'recent_count': count} 
                                     for country, count in recent_country_counts.items()]
        
        return {
            'active_count': active_count,
            'country_counts': country_counts_list,
            'seller_counts': seller_counts_list,
            'total_country_counts': total_country_counts_list,
            'recent_country_counts': recent_country_counts_list,
            'total_all_count': total_all_count,
            'recent_all_count': recent_all_count
        }
        
    except Exception as e:
        st.error(f"Error loading dashboard metrics: {e}")
        return {
            'active_count': 0,
            'country_counts': [],
            'seller_counts': [],
            'total_country_counts': [],
            'recent_country_counts': [],
            'total_all_count': 1,
            'recent_all_count': 1
        }

def calculate_fresh_season_metric(metrics):
    """Calculate the Fresh Season metric for each country using aggregated counts"""
    total_countries = {item['country_final']: item['total_count'] for item in metrics['total_country_counts']}
    recent_countries = {item['country_final']: item['recent_count'] for item in metrics['recent_country_counts']}
    
    total_all = metrics['total_all_count']
    recent_all = metrics['recent_all_count']
    
    if recent_all == 0 or total_all == 0:
        st.warning(f"No data available: total_all={total_all}, recent_all={recent_all}")
        return pd.DataFrame(columns=['Country', 'Fresh Season Score'])
    
    # Calculate Fresh Season metric for each country
    fresh_season = {}
    for country, total_count in total_countries.items():
        # Skip None/empty countries
        if not country or country == 'Unknown':
            continue
            
        recent_count = recent_countries.get(country, 0)
        
        # Calculate proportions
        total_proportion = total_count / total_all
        recent_proportion = recent_count / recent_all if recent_all > 0 else 0
        
        # Fresh Season Score = ratio of recent proportion to total proportion
        if total_proportion > 0:
            fresh_season[country] = recent_proportion / total_proportion
        else:
            fresh_season[country] = 0
    
    # Convert to DataFrame and sort by score descending
    fresh_season_df = pd.DataFrame(list(fresh_season.items()), columns=['Country', 'Fresh Season Score'])
    fresh_season_df = fresh_season_df.sort_values('Fresh Season Score', ascending=False)
    
    # Remove any rows with NaN values and filter out zero scores
    fresh_season_df = fresh_season_df.dropna()
    fresh_season_df = fresh_season_df[fresh_season_df['Fresh Season Score'] > 0]
    
    return fresh_season_df

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a styled metric card"""
    with st.container():
        st.metric(title, value, delta, delta_color=delta_color)

def main():
    # Header
    st.title("☕ Home Greens Coffee Dashboard")
    st.markdown("---")
    
    # Welcome section
    with st.container():
        st.markdown("""
        ### Welcome to Home Greens Coffee Market Explorer
        
        This dashboard provides real-time insights into the specialty green coffee market. 
        Navigate through different sections using the sidebar to:
        
        - **📊 Dashboard** (You are here) - View market overview and trends
        - **☕ All Green Coffees** - Browse and filter the complete coffee inventory
        - **🔍 Individual Coffee Information** - Explore detailed profiles and flavor analysis. Select a coffee first at the All Green Coffees page.
        - **ℹ️ About the Home Greens Project** - Learn more about our mission and methodology
        """)
    
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading dashboard data..."):
        metrics = get_dashboard_metrics()
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("Active Coffees", f"{metrics['active_count']:,}")
    
    with col2:
        unique_countries = len(metrics['country_counts'])
        create_metric_card("Countries", unique_countries)
    
    with col3:
        unique_sellers = len(metrics['seller_counts'])
        create_metric_card("Active Sellers", unique_sellers)
    
    with col4:
        # Recent additions (from Fresh Season data)
        recent_additions = metrics['recent_all_count']
        create_metric_card("New Last 30 Days", recent_additions, delta=f"+{recent_additions}")
    
    st.markdown("---")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Top Coffee Origins")
        if metrics['country_counts']:
            countries = [item['country_final'] for item in metrics['country_counts']]
            counts = [item['count'] for item in metrics['country_counts']]
            
            fig_country = px.bar(
                x=counts,
                y=countries,
                orientation='h',
                labels={'x': 'Number of Active Coffees', 'y': 'Country'},
                color=counts,
                color_continuous_scale='Viridis'
            )
            fig_country.update_layout(
                showlegend=False,
                height=600,  # Increased height for 15 countries
                margin=dict(l=0, r=0, t=0, b=0),
                yaxis={'categoryorder': 'total ascending'}  # Ensures highest values at top
            )
            st.plotly_chart(fig_country, use_container_width=True)
        else:
            st.info("No country data available")
    
    with col2:
        st.subheader("🏪 Coffee Sellers")
        if metrics['seller_counts']:
            seller_names = [item['name'] for item in metrics['seller_counts']]
            seller_counts = [item['count'] for item in metrics['seller_counts']]
            
            fig_seller = px.pie(
                values=seller_counts,
                names=seller_names,
                hole=0.4
            )
            fig_seller.update_traces(textposition='inside', textinfo='percent+label')
            fig_seller.update_layout(
                showlegend=False,
                height=600,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig_seller, use_container_width=True)
        else:
            st.info("No seller data available")
    
    st.markdown("---")
    
    # Fresh Season Metric section
    st.subheader("🌱 Fresh Season Trends")
    st.markdown("""
    The Fresh Season metric identifies countries with higher-than-usual new coffee arrivals in the past 30 days.
    A score > 1.0 indicates the country has more fresh arrivals than its typical market share.
    """)
    
    fresh_season_df = calculate_fresh_season_metric(metrics)
    
    # Split into two columns for Most Fresh and Least Fresh
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌿 Most Fresh Origins")
        if not fresh_season_df.empty:
            # Get top 10 most fresh
            most_fresh_df = fresh_season_df.head(10)
            
            fig_most_fresh = go.Figure(go.Bar(
                x=most_fresh_df['Fresh Season Score'],
                y=most_fresh_df['Country'],
                orientation='h',
                marker=dict(
                    color=most_fresh_df['Fresh Season Score'],
                    colorscale='Greens',
                    showscale=False
                ),
                text=most_fresh_df['Fresh Season Score'].round(2),
                textposition='outside'
            ))
            
            fig_most_fresh.update_layout(
                xaxis_title="Fresh Season Score",
                yaxis_title="",
                height=400,
                margin=dict(l=0, r=30, t=0, b=0),
                yaxis={'categoryorder': 'total ascending'}
            )
            
            # Add reference line at 1.0
            fig_most_fresh.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="Avg")
            
            st.plotly_chart(fig_most_fresh, use_container_width=True)
        else:
            st.info("No fresh arrivals data available.")
    
    with col2:
        st.subheader("🍂 Least Fresh Origins")
        if not fresh_season_df.empty:
            # Get bottom 10 least fresh (scores < 1.0, lowest first)
            least_fresh_candidates = fresh_season_df[fresh_season_df['Fresh Season Score'] < 1.0]
            if not least_fresh_candidates.empty:
                least_fresh_df = least_fresh_candidates.tail(10).sort_values('Fresh Season Score', ascending=True)
            else:
                least_fresh_df = pd.DataFrame(columns=['Country', 'Fresh Season Score'])
            
            if not least_fresh_df.empty:
                fig_least_fresh = go.Figure(go.Bar(
                    x=least_fresh_df['Fresh Season Score'],
                    y=least_fresh_df['Country'],
                    orientation='h',
                    marker=dict(
                        color=least_fresh_df['Fresh Season Score'],
                        colorscale='Reds',
                        showscale=False
                    ),
                    text=least_fresh_df['Fresh Season Score'].round(2),
                    textposition='outside'
                ))
                
                fig_least_fresh.update_layout(
                    xaxis_title="Fresh Season Score",
                    yaxis_title="",
                    height=400,
                    margin=dict(l=0, r=30, t=0, b=0),
                    yaxis={'categoryorder': 'total descending'}  # Lowest values at bottom, highest at top
                )
                
                # Add reference line at 1.0
                fig_least_fresh.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="Avg")
                
                st.plotly_chart(fig_least_fresh, use_container_width=True)
            else:
                st.info("No countries with Fresh Season scores < 1.0")
        else:
            st.info("No fresh arrivals data available.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>Dashboard updates hourly • Data sourced from specialty coffee importers</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()