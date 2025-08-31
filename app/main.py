"""Main Streamlit application for FloatChat."""

import streamlit as st
import pandas as pd
import json
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import asyncio

# Try to import httpx, fall back to requests if not available
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    import requests
    HAS_HTTPX = False

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.dashboard.visualizations import ArgoVisualizer, render_data_table, create_summary_cards, display_query_info
from config.settings import settings

# Configure page
st.set_page_config(
    page_title="FloatChat - ARGO Data Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize visualizer
@st.cache_resource
def get_visualizer():
    return ArgoVisualizer()

visualizer = get_visualizer()

# API client
class FloatChatAPI:
    """Client for FloatChat API."""
    
    def __init__(self, base_url: str = f"http://{settings.app_host}:{settings.app_port}"):
        self.base_url = base_url
    
    async def query(self, query_text: str, include_viz: bool = True) -> Dict[str, Any]:
        """Send query to API."""
        if HAS_HTTPX:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/query",
                    json={
                        "query": query_text,
                        "include_visualization": include_viz,
                        "max_results": 1000
                    },
                    timeout=60.0
                )
                return response.json()
        else:
            # Fallback to requests (synchronous)
            response = requests.post(
                f"{self.base_url}/query",
                json={
                    "query": query_text,
                    "include_visualization": include_viz,
                    "max_results": 1000
                },
                timeout=60.0
            )
            return response.json()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get data statistics."""
        if HAS_HTTPX:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/data/stats")
                return response.json()
        else:
            response = requests.get(f"{self.base_url}/data/stats")
            return response.json()
    
    async def get_floats(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Get list of floats."""
        if HAS_HTTPX:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/floats",
                    params={"limit": limit, "offset": offset}
                )
                return response.json()
        else:
            response = requests.get(
                f"{self.base_url}/floats",
                params={"limit": limit, "offset": offset}
            )
            return response.json()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        try:
            if HAS_HTTPX:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.base_url}/health", timeout=5.0)
                    return response.json()
            else:
                response = requests.get(f"{self.base_url}/health", timeout=5.0)
                return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

@st.cache_resource
def get_api_client():
    return FloatChatAPI()

api = get_api_client()

# Helper functions
def run_async(coro):
    """Run async function in Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# Main app
def main():
    """Main application function."""
    
    # Header
    st.title("🌊 FloatChat - ARGO Data Explorer")
    st.markdown("*AI-powered conversational interface for oceanographic data discovery*")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "💬 Chat Interface", "📊 Data Explorer", "🗺️ Map View", "⚙️ Settings"]
        )
        
        # API Status
        st.subheader("System Status")
        health = run_async(api.health_check())
        if health.get("status") == "healthy":
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.error(health.get("error", "Unknown error"))
    
    # Route to different pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "💬 Chat Interface":
        show_chat_interface()
    elif page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "🗺️ Map View":
        show_map_view()
    elif page == "⚙️ Settings":
        show_settings()

def show_home_page():
    """Display home page with overview and statistics."""
    
    st.header("Welcome to FloatChat")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About ARGO Floats
        
        The Argo program is a global array of autonomous profiling floats that measure temperature, 
        salinity, and other properties of the ocean. These floats drift with ocean currents and 
        periodically dive to collect vertical profiles of the water column.
        
        ### Features
        
        - **🤖 Natural Language Queries**: Ask questions in plain English
        - **📊 Interactive Visualizations**: Maps, profiles, and time series
        - **🔍 Smart Search**: AI-powered data discovery
        - **📈 Real-time Analysis**: Instant insights from oceanographic data
        
        ### Getting Started
        
        1. Go to the **Chat Interface** to ask questions about ARGO data
        2. Use the **Data Explorer** to browse available datasets
        3. View **Map View** for geographic data exploration
        
        ### Example Queries
        
        - "Show me salinity profiles near the equator in March 2023"
        - "What are the nearest ARGO floats to 15°N, 65°E?"
        - "Compare temperature data in the Arabian Sea over the last 6 months"
        """)
    
    with col2:
        st.subheader("Data Overview")
        
        # Get and display statistics
        try:
            stats = run_async(api.get_stats())
            create_summary_cards(stats)
            
            # Recent activity
            st.subheader("Coverage")
            if stats.get('temporal_coverage'):
                temp_cov = stats['temporal_coverage']
                if temp_cov.get('earliest_date') and temp_cov.get('latest_date'):
                    earliest = pd.to_datetime(temp_cov['earliest_date'])
                    latest = pd.to_datetime(temp_cov['latest_date'])
                    st.write(f"**Time Range:** {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
            
            if stats.get('spatial_coverage'):
                spatial = stats['spatial_coverage']
                lat_range = spatial.get('latitude_range', [None, None])
                lon_range = spatial.get('longitude_range', [None, None])
                if lat_range[0] and lat_range[1]:
                    st.write(f"**Latitude:** {lat_range[0]:.1f}° to {lat_range[1]:.1f}°")
                if lon_range[0] and lon_range[1]:
                    st.write(f"**Longitude:** {lon_range[0]:.1f}° to {lon_range[1]:.1f}°")
        
        except Exception as e:
            st.error(f"Could not load statistics: {e}")

def show_chat_interface():
    """Display chat interface for natural language queries."""
    
    st.header("💬 Chat Interface")
    st.markdown("Ask questions about ARGO float data in natural language")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display visualizations if available
            if message.get("visualization_data"):
                display_visualization(message["visualization_data"])
    
    # Chat input
    if prompt := st.chat_input("Ask about ARGO data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query and display response
        with st.chat_message("assistant"):
            with st.spinner("Processing your query..."):
                try:
                    result = run_async(api.query(prompt, include_viz=True))
                    
                    if result.get("success"):
                        # Display response
                        response_text = result.get("response", "I found some data for your query.")
                        st.markdown(response_text)
                        
                        # Display query info
                        display_query_info(result)
                        
                        # Display visualization
                        viz_data = result.get("visualization_data")
                        if viz_data and viz_data.get("type") != "none":
                            display_visualization(viz_data)
                        
                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "visualization_data": viz_data
                        })
                    
                    else:
                        error_msg = f"Sorry, I couldn't process your query: {result.get('error', 'Unknown error')}"
                        st.error(error_msg)
                        
                        if result.get("suggestions"):
                            st.subheader("Suggestions:")
                            for suggestion in result["suggestions"]:
                                st.info(suggestion)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

def show_data_explorer():
    """Display data exploration interface."""
    
    st.header("📊 Data Explorer")
    
    # Query input
    st.subheader("Query Data")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query_text = st.text_input(
            "Enter your query:",
            placeholder="e.g., Show temperature profiles in the Indian Ocean",
            help="Ask questions about ARGO data in natural language"
        )
    
    with col2:
        include_viz = st.checkbox("Include Visualizations", value=True)
    
    if st.button("Execute Query", type="primary"):
        if query_text:
            with st.spinner("Processing query..."):
                try:
                    result = run_async(api.query(query_text, include_viz=include_viz))
                    
                    if result.get("success"):
                        st.success("Query executed successfully!")
                        
                        # Display query information
                        display_query_info(result)
                        
                        # Display results
                        query_results = result.get("query_results", {})
                        if query_results.get("data"):
                            df = pd.DataFrame(query_results["data"])
                            
                            st.subheader("Results")
                            render_data_table(df, max_rows=100)
                            
                            # Display visualization
                            viz_data = result.get("visualization_data")
                            if viz_data and viz_data.get("type") != "none":
                                st.subheader("Visualization")
                                display_visualization(viz_data)
                        
                        else:
                            st.info("No data found for your query.")
                    
                    else:
                        st.error(f"Query failed: {result.get('error', 'Unknown error')}")
                        if result.get("suggestions"):
                            for suggestion in result["suggestions"]:
                                st.info(suggestion)
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a query.")
    
    # Browse floats
    st.subheader("Browse Available Floats")
    
    try:
        floats_data = run_async(api.get_floats(limit=20))
        if floats_data.get("floats"):
            floats_df = pd.DataFrame(floats_data["floats"])
            st.dataframe(floats_df, use_container_width=True)
        else:
            st.info("No float data available.")
    
    except Exception as e:
        st.error(f"Could not load float data: {e}")

def show_map_view():
    """Display map-based data exploration."""
    
    st.header("🗺️ Map View")
    st.markdown("Geographic exploration of ARGO float data")
    
    # Map controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        region = st.selectbox(
            "Select Region:",
            ["Global", "Indian Ocean", "Pacific Ocean", "Atlantic Ocean", "Arabian Sea", "Mediterranean"]
        )
    
    with col2:
        time_period = st.selectbox(
            "Time Period:",
            ["Last Month", "Last 3 Months", "Last 6 Months", "Last Year", "All Time"]
        )
    
    with col3:
        parameter = st.selectbox(
            "Parameter:",
            ["All", "Temperature", "Salinity", "Pressure"]
        )
    
    if st.button("Load Map Data"):
        # Construct query based on selections
        query_parts = []
        
        if region != "Global":
            query_parts.append(f"in the {region}")
        
        if time_period != "All Time":
            query_parts.append(f"from the {time_period.lower()}")
        
        if parameter != "All":
            query_parts.append(f"with {parameter.lower()} measurements")
        
        if query_parts:
            query = f"Show ARGO float locations {' '.join(query_parts)}"
        else:
            query = "Show all ARGO float locations"
        
        with st.spinner("Loading map data..."):
            try:
                result = run_async(api.query(query, include_viz=True))
                
                if result.get("success"):
                    viz_data = result.get("visualization_data")
                    if viz_data and viz_data.get("type") == "map":
                        display_visualization(viz_data)
                    else:
                        st.info("No geographic data found for the selected criteria.")
                else:
                    st.error(f"Could not load map data: {result.get('error')}")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

def show_settings():
    """Display application settings."""
    
    st.header("⚙️ Settings")
    
    st.subheader("API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("API Host", value=settings.app_host, disabled=True)
        st.text_input("API Port", value=str(settings.app_port), disabled=True)
    
    with col2:
        google_key = "Configured" if settings.google_api_key else "Not configured"
        st.text_input("Google API Key", value=google_key, disabled=True)
    
    st.subheader("Data Paths")
    st.text_input("ARGO Data Path", value=settings.argo_data_path, disabled=True)
    st.text_input("Processed Data Path", value=settings.processed_data_path, disabled=True)
    
    st.subheader("System Information")
    st.info(f"Debug Mode: {'Enabled' if settings.debug else 'Disabled'}")
    st.info(f"Log Level: {settings.log_level}")
    
    # Clear cache button
    if st.button("Clear Application Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared successfully!")

def display_visualization(viz_data: Dict[str, Any]):
    """Display visualization based on data type."""
    
    viz_type = viz_data.get("type")
    data = viz_data.get("data")
    config = viz_data.get("config", {})
    
    if not data:
        st.info("No visualization data available.")
        return
    
    try:
        if viz_type == "map":
            # Display map
            map_obj = visualizer.create_map_visualization(data, config)
            st.components.v1.html(map_obj._repr_html_(), height=500)
        
        elif viz_type == "profile":
            # Display profile plot
            fig = visualizer.create_profile_plot(data, config)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "time_series":
            # Display time series plot
            fig = visualizer.create_time_series_plot(data, config)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "table":
            # Display data table
            df = pd.DataFrame(data)
            render_data_table(df, max_rows=config.get("max_rows", 100))
        
        else:
            st.info(f"Visualization type '{viz_type}' not supported yet.")
    
    except Exception as e:
        st.error(f"Error displaying visualization: {str(e)}")

if __name__ == "__main__":
    main()
