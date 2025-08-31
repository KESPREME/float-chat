"""Simplified demo version of FloatChat that works without complex dependencies."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import json

# Configure page
st.set_page_config(
    page_title="FloatChat Demo - ARGO Data Explorer",
    page_icon="🌊",
    layout="wide"
)

# Generate sample ARGO data for demo
@st.cache_data
def generate_demo_data():
    """Generate synthetic ARGO float data for demonstration."""
    np.random.seed(42)
    
    # Create 5 synthetic floats
    floats = []
    regions = [
        {"name": "Arabian Sea", "lat": 17.5, "lon": 65, "lat_range": 5, "lon_range": 10},
        {"name": "Bay of Bengal", "lat": 15, "lon": 90, "lat_range": 8, "lon_range": 8},
        {"name": "Equatorial Indian Ocean", "lat": 0, "lon": 80, "lat_range": 10, "lon_range": 15},
        {"name": "Southern Indian Ocean", "lat": -25, "lon": 70, "lat_range": 10, "lon_range": 20},
        {"name": "Western Pacific", "lat": 10, "lon": 140, "lat_range": 15, "lon_range": 20}
    ]
    
    data = []
    
    for i, region in enumerate(regions):
        platform_number = f"590{i+1:04d}"
        
        # Generate 30 profiles over 300 days
        for profile_idx in range(30):
            # Float position with drift
            lat = region["lat"] + np.random.normal(0, region["lat_range"]/3)
            lon = region["lon"] + np.random.normal(0, region["lon_range"]/3)
            
            # Profile date
            base_date = datetime.now() - timedelta(days=300)
            profile_date = base_date + timedelta(days=profile_idx * 10)
            
            # Generate depth profile
            depths = [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000]
            
            for depth in depths:
                # Realistic temperature profile
                surface_temp = 26 + 3 * np.sin(2 * np.pi * profile_date.month / 12)
                temp = surface_temp * np.exp(-depth / 1000) + 2 + np.random.normal(0, 0.5)
                
                # Realistic salinity profile
                salinity = 35 + (depth / 2000) * 0.5 + np.random.normal(0, 0.1)
                
                data.append({
                    'platform_number': platform_number,
                    'region': region["name"],
                    'date': profile_date,
                    'latitude': lat,
                    'longitude': lon,
                    'pressure': depth,
                    'temperature': round(temp, 2),
                    'salinity': round(salinity, 3),
                    'cycle_number': profile_idx + 1
                })
    
    return pd.DataFrame(data)

# Simple query processor
class SimpleQueryProcessor:
    """Simple query processor for demo without LLM dependencies."""
    
    def __init__(self, df):
        self.df = df
    
    def process_query(self, query):
        """Process natural language query and return filtered data."""
        query_lower = query.lower()
        filtered_df = self.df.copy()
        
        # Geographic filters
        if "arabian sea" in query_lower:
            filtered_df = filtered_df[filtered_df['region'] == 'Arabian Sea']
        elif "bay of bengal" in query_lower:
            filtered_df = filtered_df[filtered_df['region'] == 'Bay of Bengal']
        elif "equator" in query_lower:
            filtered_df = filtered_df[filtered_df['region'] == 'Equatorial Indian Ocean']
        elif "indian ocean" in query_lower:
            filtered_df = filtered_df[filtered_df['region'].str.contains('Indian Ocean')]
        elif "pacific" in query_lower:
            filtered_df = filtered_df[filtered_df['region'].str.contains('Pacific')]
        
        # Parameter filters
        if "temperature" in query_lower:
            filtered_df = filtered_df[filtered_df['temperature'].notna()]
        elif "salinity" in query_lower:
            filtered_df = filtered_df[filtered_df['salinity'].notna()]
        
        # Depth filters
        if "surface" in query_lower:
            filtered_df = filtered_df[filtered_df['pressure'] <= 50]
        elif "deep" in query_lower:
            filtered_df = filtered_df[filtered_df['pressure'] >= 1000]
        
        # Time filters
        if "recent" in query_lower or "latest" in query_lower:
            cutoff_date = datetime.now() - timedelta(days=60)
            filtered_df = filtered_df[filtered_df['date'] >= cutoff_date]
        elif "march" in query_lower:
            filtered_df = filtered_df[filtered_df['date'].dt.month == 3]
        
        return filtered_df

# Initialize demo data
df = generate_demo_data()
query_processor = SimpleQueryProcessor(df)

# Main app
def main():
    st.title("🌊 FloatChat Demo - ARGO Data Explorer")
    st.markdown("*AI-powered conversational interface for oceanographic data discovery*")
    
    # Sidebar
    with st.sidebar:
        st.header("Demo Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Overview", "💬 Chat Demo", "📊 Data Explorer", "🗺️ Map View"]
        )
        
        st.subheader("Demo Data")
        st.info(f"**{len(df['platform_number'].unique())} floats**")
        st.info(f"**{len(df):,} measurements**")
        st.info(f"**{len(df['region'].unique())} regions**")
    
    if page == "🏠 Overview":
        show_overview()
    elif page == "💬 Chat Demo":
        show_chat_demo()
    elif page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "🗺️ Map View":
        show_map_view()

def show_overview():
    """Show overview with statistics."""
    st.header("Demo Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Floats", len(df['platform_number'].unique()))
    
    with col2:
        st.metric("Total Measurements", f"{len(df):,}")
    
    with col3:
        st.metric("Regions Covered", len(df['region'].unique()))
    
    with col4:
        date_range = (df['date'].max() - df['date'].min()).days
        st.metric("Days of Data", date_range)
    
    # Regional distribution
    st.subheader("Regional Distribution")
    region_counts = df.groupby('region').size().reset_index(name='count')
    fig = px.bar(region_counts, x='region', y='count', title='Measurements per Region')
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

def show_chat_demo():
    """Show chat interface demo."""
    st.header("💬 Chat Interface Demo")
    
    # Sample queries
    st.subheader("Try these sample queries:")
    sample_queries = [
        "Show me temperature data in the Arabian Sea",
        "What salinity profiles are available near the equator?",
        "Display recent measurements from all floats",
        "Show deep water data below 1000m",
        "Compare temperature in different regions"
    ]
    
    for query in sample_queries:
        if st.button(f"📝 {query}", key=query):
            st.session_state.demo_query = query
    
    # Chat input
    user_query = st.text_input("Or type your own query:", key="chat_input")
    
    if st.button("Submit Query") or st.session_state.get('demo_query'):
        query = st.session_state.get('demo_query', user_query)
        
        if query:
            st.subheader("Query Results")
            
            # Process query
            with st.spinner("Processing query..."):
                result_df = query_processor.process_query(query)
                
                if len(result_df) > 0:
                    st.success(f"Found {len(result_df)} measurements matching your query!")
                    
                    # Show results
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write("**Data Summary:**")
                        st.write(f"• Floats: {result_df['platform_number'].nunique()}")
                        st.write(f"• Regions: {', '.join(result_df['region'].unique())}")
                        st.write(f"• Date range: {result_df['date'].min().strftime('%Y-%m-%d')} to {result_df['date'].max().strftime('%Y-%m-%d')}")
                        
                        if 'temperature' in query.lower():
                            temp_range = f"{result_df['temperature'].min():.1f}°C to {result_df['temperature'].max():.1f}°C"
                            st.write(f"• Temperature range: {temp_range}")
                        
                        if 'salinity' in query.lower():
                            sal_range = f"{result_df['salinity'].min():.2f} to {result_df['salinity'].max():.2f} PSU"
                            st.write(f"• Salinity range: {sal_range}")
                    
                    with col2:
                        # Create visualization based on query
                        if "map" in query.lower() or "location" in query.lower():
                            create_map_viz(result_df)
                        elif "profile" in query.lower() or "depth" in query.lower():
                            create_profile_viz(result_df)
                        else:
                            create_scatter_viz(result_df)
                    
                    # Show data table
                    st.subheader("Data Table")
                    st.dataframe(result_df.head(50), use_container_width=True)
                
                else:
                    st.warning("No data found matching your query. Try a different search term.")
            
            # Clear the demo query
            if 'demo_query' in st.session_state:
                del st.session_state.demo_query

def show_data_explorer():
    """Show data exploration interface."""
    st.header("📊 Data Explorer")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_regions = st.multiselect(
            "Select Regions:",
            df['region'].unique(),
            default=df['region'].unique()
        )
    
    with col2:
        selected_floats = st.multiselect(
            "Select Floats:",
            df['platform_number'].unique(),
            default=df['platform_number'].unique()[:3]
        )
    
    with col3:
        depth_range = st.slider(
            "Depth Range (m):",
            0, 2000, (0, 500)
        )
    
    # Filter data
    filtered_df = df[
        (df['region'].isin(selected_regions)) &
        (df['platform_number'].isin(selected_floats)) &
        (df['pressure'] >= depth_range[0]) &
        (df['pressure'] <= depth_range[1])
    ]
    
    st.write(f"**Showing {len(filtered_df)} measurements**")
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Profiles", "🗺️ Map", "📊 Statistics"])
    
    with tab1:
        create_profile_viz(filtered_df)
    
    with tab2:
        create_map_viz(filtered_df)
    
    with tab3:
        create_stats_viz(filtered_df)

def show_map_view():
    """Show map-focused view."""
    st.header("🗺️ Geographic View")
    
    # Create map visualization
    create_map_viz(df)
    
    # Regional statistics
    st.subheader("Regional Statistics")
    region_stats = df.groupby('region').agg({
        'platform_number': 'nunique',
        'temperature': ['mean', 'min', 'max'],
        'salinity': ['mean', 'min', 'max'],
        'pressure': 'max'
    }).round(2)
    
    st.dataframe(region_stats, use_container_width=True)

def create_map_viz(data_df):
    """Create map visualization."""
    if len(data_df) == 0:
        st.warning("No data to display on map.")
        return
    
    # Get unique locations per float
    location_df = data_df.groupby('platform_number').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'region': 'first',
        'temperature': 'mean',
        'salinity': 'mean'
    }).reset_index()
    
    # Create scatter plot on map
    fig = px.scatter_geo(
        location_df,
        lat='latitude',
        lon='longitude',
        color='region',
        size='temperature',
        hover_name='platform_number',
        hover_data=['temperature', 'salinity'],
        title='ARGO Float Locations'
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def create_profile_viz(data_df):
    """Create depth profile visualization."""
    if len(data_df) == 0:
        st.warning("No data to display profiles.")
        return
    
    # Create subplots for temperature and salinity
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Temperature Profiles', 'Salinity Profiles'),
        shared_yaxes=True
    )
    
    colors = px.colors.qualitative.Set3
    
    for i, platform in enumerate(data_df['platform_number'].unique()[:5]):  # Limit to 5 floats
        platform_data = data_df[data_df['platform_number'] == platform]
        color = colors[i % len(colors)]
        
        # Temperature profile
        fig.add_trace(
            go.Scatter(
                x=platform_data['temperature'],
                y=platform_data['pressure'],
                mode='lines+markers',
                name=f'Float {platform}',
                line=dict(color=color),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Salinity profile
        fig.add_trace(
            go.Scatter(
                x=platform_data['salinity'],
                y=platform_data['pressure'],
                mode='lines+markers',
                name=f'Float {platform}',
                line=dict(color=color),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Salinity (PSU)", row=1, col=2)
    fig.update_yaxes(title_text="Pressure (dbar)", autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    
    fig.update_layout(height=500, title='ARGO Float Depth Profiles')
    st.plotly_chart(fig, use_container_width=True)

def create_scatter_viz(data_df):
    """Create scatter plot visualization."""
    if len(data_df) == 0:
        st.warning("No data to display.")
        return
    
    fig = px.scatter(
        data_df,
        x='temperature',
        y='salinity',
        color='region',
        size='pressure',
        hover_data=['platform_number', 'date'],
        title='Temperature vs Salinity (bubble size = depth)'
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def create_stats_viz(data_df):
    """Create statistical visualizations."""
    if len(data_df) == 0:
        st.warning("No data for statistics.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature distribution
        fig_temp = px.histogram(
            data_df, x='temperature', nbins=30,
            title='Temperature Distribution'
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        # Salinity distribution
        fig_sal = px.histogram(
            data_df, x='salinity', nbins=30,
            title='Salinity Distribution'
        )
        st.plotly_chart(fig_sal, use_container_width=True)

# Import plotly subplots
from plotly.subplots import make_subplots

if __name__ == "__main__":
    main()
