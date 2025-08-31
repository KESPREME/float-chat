"""Visualization components for ARGO data dashboard."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import streamlit as st

# Try to import folium, provide fallback if not available
try:
    import folium
    from folium import plugins
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    # Create dummy folium module for fallback
    class DummyFolium:
        class Map:
            def __init__(self, *args, **kwargs):
                pass
            def _repr_html_(self):
                return "<div>Map visualization requires folium package. Please install: pip install folium</div>"
        class plugins:
            @staticmethod
            def Fullscreen():
                return None
            @staticmethod
            def MeasureControl():
                return None
    folium = DummyFolium()


class ArgoVisualizer:
    """Main visualization class for ARGO float data."""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def create_map_visualization(self, data: List[Dict], config: Dict):
        """Create interactive map with ARGO float locations.
        
        Args:
            data: List of data points with lat/lon coordinates
            config: Map configuration (center, zoom, etc.)
            
        Returns:
            Folium map object or fallback message
        """
        if not HAS_FOLIUM:
            return folium.Map()
        
        # Initialize map
        center_lat = config.get('center_lat', 0)
        center_lon = config.get('center_lon', 0)
        zoom = config.get('zoom', 2)
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles='OpenStreetMap'
        )
        
        # Add tile layers
        folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
        
        # Group points by platform for better organization
        platform_groups = {}
        for point in data:
            platform = point.get('platform_number', 'Unknown')
            if platform not in platform_groups:
                platform_groups[platform] = []
            platform_groups[platform].append(point)
        
        # Add markers for each platform
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
                 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 
                 'gray', 'black', 'lightgray']
        
        for i, (platform, points) in enumerate(platform_groups.items()):
            color = colors[i % len(colors)]
            
            # Create feature group for this platform
            fg = folium.FeatureGroup(name=f'Float {platform}')
            
            for point in points:
                folium.CircleMarker(
                    location=[point['lat'], point['lon']],
                    radius=6,
                    popup=folium.Popup(point.get('popup_text', ''), max_width=300),
                    color='black',
                    weight=1,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(fg)
            
            fg.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Add measure tool
        plugins.MeasureControl().add_to(m)
        
        return m
    
    def create_profile_plot(self, data: Dict[str, List], config: Dict) -> go.Figure:
        """Create depth profile plots for temperature/salinity.
        
        Args:
            data: Dictionary with platform data
            config: Plot configuration
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Temperature Profile', 'Salinity Profile'),
            shared_yaxes=True,
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, (platform, profile_data) in enumerate(data.items()):
            color = colors[i % len(colors)]
            
            pressure = profile_data.get('pressure', [])
            temperature = profile_data.get('temperature', [])
            salinity = profile_data.get('salinity', [])
            
            # Filter out None values
            valid_temp_idx = [i for i, (p, t) in enumerate(zip(pressure, temperature)) 
                             if p is not None and t is not None]
            valid_sal_idx = [i for i, (p, s) in enumerate(zip(pressure, salinity)) 
                            if p is not None and s is not None]
            
            # Temperature profile
            if valid_temp_idx:
                temp_pressure = [pressure[i] for i in valid_temp_idx]
                temp_values = [temperature[i] for i in valid_temp_idx]
                
                fig.add_trace(
                    go.Scatter(
                        x=temp_values,
                        y=temp_pressure,
                        mode='lines+markers',
                        name=f'Float {platform}',
                        line=dict(color=color),
                        marker=dict(size=4),
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Salinity profile
            if valid_sal_idx:
                sal_pressure = [pressure[i] for i in valid_sal_idx]
                sal_values = [salinity[i] for i in valid_sal_idx]
                
                fig.add_trace(
                    go.Scatter(
                        x=sal_values,
                        y=sal_pressure,
                        mode='lines+markers',
                        name=f'Float {platform}',
                        line=dict(color=color),
                        marker=dict(size=4),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # Update layout
        fig.update_layout(
            title='ARGO Float Depth Profiles',
            height=600,
            hovermode='closest'
        )
        
        # Update x-axes
        fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_xaxes(title_text="Salinity (PSU)", row=1, col=2)
        
        # Update y-axes (invert for depth)
        fig.update_yaxes(
            title_text="Pressure (dbar)",
            autorange="reversed",
            row=1, col=1
        )
        fig.update_yaxes(
            autorange="reversed",
            row=1, col=2
        )
        
        return fig
    
    def create_time_series_plot(self, data: Dict[str, List], config: Dict) -> go.Figure:
        """Create time series plot for ARGO data.
        
        Args:
            data: Dictionary with time series data
            config: Plot configuration
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, (platform, series_data) in enumerate(data.items()):
            color = colors[i % len(colors)]
            
            dates = series_data.get('dates', [])
            values = series_data.get('values', [])
            
            # Filter out None values
            valid_data = [(d, v) for d, v in zip(dates, values) if v is not None]
            if valid_data:
                dates_clean, values_clean = zip(*valid_data)
                
                fig.add_trace(
                    go.Scatter(
                        x=dates_clean,
                        y=values_clean,
                        mode='lines+markers',
                        name=f'Float {platform}',
                        line=dict(color=color),
                        marker=dict(size=6)
                    )
                )
        
        # Update layout
        fig.update_layout(
            title='ARGO Float Time Series',
            xaxis_title='Date',
            yaxis_title='Value',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, 
                           color_col: Optional[str] = None) -> go.Figure:
        """Create scatter plot for ARGO data relationships.
        
        Args:
            df: DataFrame with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            color_col: Optional column for color coding
            
        Returns:
            Plotly figure object
        """
        if color_col and color_col in df.columns:
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_col,
                title=f'{y_col} vs {x_col}',
                hover_data=['platform_number'] if 'platform_number' in df.columns else None
            )
        else:
            fig = px.scatter(
                df, x=x_col, y=y_col,
                title=f'{y_col} vs {x_col}',
                hover_data=['platform_number'] if 'platform_number' in df.columns else None
            )
        
        fig.update_layout(height=500)
        return fig
    
    def create_histogram(self, df: pd.DataFrame, column: str, 
                        bins: int = 30) -> go.Figure:
        """Create histogram for data distribution.
        
        Args:
            df: DataFrame with data
            column: Column to plot
            bins: Number of bins
            
        Returns:
            Plotly figure object
        """
        fig = px.histogram(
            df, x=column, nbins=bins,
            title=f'Distribution of {column}',
            marginal='box'
        )
        
        fig.update_layout(height=500)
        return fig
    
    def create_heatmap(self, df: pd.DataFrame, x_col: str, y_col: str, 
                      z_col: str, bins: int = 20) -> go.Figure:
        """Create 2D heatmap for spatial/temporal data.
        
        Args:
            df: DataFrame with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            z_col: Column for color values
            bins: Number of bins for aggregation
            
        Returns:
            Plotly figure object
        """
        # Create bins for x and y
        x_bins = np.linspace(df[x_col].min(), df[x_col].max(), bins)
        y_bins = np.linspace(df[y_col].min(), df[y_col].max(), bins)
        
        # Create 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            df[x_col], df[y_col], bins=[x_bins, y_bins],
            weights=df[z_col]
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=hist.T,
            x=x_edges[:-1],
            y=y_edges[:-1],
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'{z_col} Heatmap ({x_col} vs {y_col})',
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=500
        )
        
        return fig
    
    def create_3d_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, 
                         z_col: str, color_col: Optional[str] = None) -> go.Figure:
        """Create 3D scatter plot for multi-dimensional data.
        
        Args:
            df: DataFrame with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            z_col: Column for z-axis
            color_col: Optional column for color coding
            
        Returns:
            Plotly figure object
        """
        if color_col and color_col in df.columns:
            fig = px.scatter_3d(
                df, x=x_col, y=y_col, z=z_col, color=color_col,
                title=f'3D Scatter: {x_col}, {y_col}, {z_col}',
                hover_data=['platform_number'] if 'platform_number' in df.columns else None
            )
        else:
            fig = px.scatter_3d(
                df, x=x_col, y=y_col, z=z_col,
                title=f'3D Scatter: {x_col}, {y_col}, {z_col}',
                hover_data=['platform_number'] if 'platform_number' in df.columns else None
            )
        
        fig.update_layout(height=600)
        return fig


def render_data_table(df: pd.DataFrame, max_rows: int = 100):
    """Render data table with Streamlit.
    
    Args:
        df: DataFrame to display
        max_rows: Maximum number of rows to show
    """
    if len(df) > max_rows:
        st.warning(f"Showing first {max_rows} rows of {len(df)} total rows")
        df_display = df.head(max_rows)
    else:
        df_display = df
    
    st.dataframe(
        df_display,
        use_container_width=True,
        height=400
    )
    
    # Add download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download full dataset as CSV",
        data=csv,
        file_name=f"argo_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def create_summary_cards(stats: Dict[str, Any]):
    """Create summary statistic cards.
    
    Args:
        stats: Dictionary with summary statistics
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Floats",
            value=stats.get('total_floats', 0)
        )
    
    with col2:
        st.metric(
            label="Total Profiles",
            value=f"{stats.get('total_profiles', 0):,}"
        )
    
    with col3:
        if stats.get('temporal_coverage', {}).get('earliest_date'):
            earliest = pd.to_datetime(stats['temporal_coverage']['earliest_date'])
            st.metric(
                label="Data Since",
                value=earliest.strftime('%Y-%m-%d')
            )
        else:
            st.metric(label="Data Since", value="N/A")
    
    with col4:
        if stats.get('spatial_coverage', {}).get('latitude_range'):
            lat_range = stats['spatial_coverage']['latitude_range']
            lat_span = abs(lat_range[1] - lat_range[0]) if lat_range[0] and lat_range[1] else 0
            st.metric(
                label="Latitude Span",
                value=f"{lat_span:.1f}°"
            )
        else:
            st.metric(label="Latitude Span", value="N/A")


def display_query_info(query_result: Dict[str, Any]):
    """Display information about the executed query.
    
    Args:
        query_result: Result from RAG pipeline
    """
    if query_result.get('success'):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Results Found",
                value=query_result.get('row_count', 0)
            )
        
        with col2:
            exec_time = query_result.get('execution_time', 0)
            st.metric(
                label="Query Time",
                value=f"{exec_time:.3f}s"
            )
        
        with col3:
            if query_result.get('sql_query'):
                st.metric(
                    label="SQL Generated",
                    value="✓"
                )
        
        # Show SQL query in expander
        if query_result.get('sql_query'):
            with st.expander("View Generated SQL Query"):
                st.code(query_result['sql_query'], language='sql')
        
        # Show suggestions if any
        if query_result.get('suggestions'):
            with st.expander("Query Optimization Suggestions"):
                for suggestion in query_result['suggestions']:
                    st.info(suggestion)
    
    else:
        st.error(f"Query failed: {query_result.get('error', 'Unknown error')}")
        if query_result.get('suggestions'):
            st.subheader("Suggestions:")
            for suggestion in query_result['suggestions']:
                st.info(suggestion)
