"""Sample data loader for testing FloatChat with synthetic ARGO data."""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from data.storage.database import DatabaseManager
from data.storage.vector_store import create_vector_store, ArgoVectorManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_argo_data(num_floats: int = 5, profiles_per_float: int = 50) -> pd.DataFrame:
    """Generate synthetic ARGO float data for testing.
    
    Args:
        num_floats: Number of floats to simulate
        profiles_per_float: Number of profiles per float
        
    Returns:
        DataFrame with synthetic ARGO data
    """
    logger.info(f"Generating {num_floats} floats with {profiles_per_float} profiles each...")
    
    data = []
    
    # Define regions for realistic geographic distribution
    regions = [
        {"name": "Arabian Sea", "lat_range": (10, 25), "lon_range": (50, 80)},
        {"name": "Bay of Bengal", "lat_range": (5, 22), "lon_range": (80, 100)},
        {"name": "Equatorial Indian Ocean", "lat_range": (-10, 10), "lon_range": (50, 100)},
        {"name": "Southern Indian Ocean", "lat_range": (-40, -10), "lon_range": (20, 120)},
        {"name": "Western Pacific", "lat_range": (-20, 20), "lon_range": (120, 160)}
    ]
    
    for float_id in range(1, num_floats + 1):
        platform_number = f"590{float_id:04d}"  # Realistic platform number format
        
        # Choose a region for this float
        region = random.choice(regions)
        base_lat = random.uniform(*region["lat_range"])
        base_lon = random.uniform(*region["lon_range"])
        
        # Generate deployment date (last 2 years)
        start_date = datetime.now() - timedelta(days=random.randint(30, 730))
        
        for profile_id in range(profiles_per_float):
            # Float drifts over time
            lat_drift = random.gauss(0, 0.5)  # Small random drift
            lon_drift = random.gauss(0, 0.5)
            
            profile_lat = base_lat + lat_drift * profile_id * 0.1
            profile_lon = base_lon + lon_drift * profile_id * 0.1
            
            # Profile date (every 10 days typical cycle)
            profile_date = start_date + timedelta(days=profile_id * 10)
            
            # Generate depth profile (0-2000m typical)
            depths = np.array([0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 
                              600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1750, 2000])
            
            # Realistic temperature profile (warmer at surface, colder at depth)
            surface_temp = 25 + 5 * np.sin(np.pi * (profile_date.month - 3) / 6)  # Seasonal variation
            temp_profile = surface_temp * np.exp(-depths / 1000) + 2  # Exponential decay + deep water temp
            temp_profile += np.random.normal(0, 0.5, len(depths))  # Add noise
            
            # Realistic salinity profile
            surface_salinity = 35 + 2 * np.random.normal(0, 0.5)  # Surface salinity variation
            salinity_profile = surface_salinity + (depths / 1000) * 0.5  # Slight increase with depth
            salinity_profile += np.random.normal(0, 0.1, len(depths))  # Add noise
            
            # Create records for each depth
            for i, depth in enumerate(depths):
                data.append({
                    'platform_number': platform_number,
                    'cycle_number': profile_id + 1,
                    'data_centre': 'IN',  # Indian data center
                    'data_mode': 'R',  # Real-time
                    'date': profile_date,
                    'position_qc': '1',  # Good quality
                    'latitude': profile_lat,
                    'longitude': profile_lon,
                    'pressure': depth,
                    'temperature': round(temp_profile[i], 3),
                    'salinity': round(salinity_profile[i], 3),
                    'pressure_qc': '1',
                    'temperature_qc': '1',
                    'salinity_qc': '1',
                    'file_path': f'synthetic_data/float_{platform_number}_profile_{profile_id:03d}.nc',
                    'processed_at': datetime.now()
                })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} measurement records")
    return df


def load_sample_data():
    """Load sample data into database and vector store."""
    logger.info("Loading sample ARGO data...")
    
    try:
        # Generate sample data
        df = generate_sample_argo_data(num_floats=10, profiles_per_float=30)
        
        # Initialize database
        db_manager = DatabaseManager(settings.database_url)
        
        # Convert DataFrame to records
        records = df.to_dict('records')
        
        # Insert into database
        logger.info("Inserting data into database...")
        inserted_count = db_manager.insert_profiles_batch(records)
        logger.info(f"Inserted {inserted_count} profile records")
        
        # Update summaries for each float
        logger.info("Updating float summaries...")
        platforms = df['platform_number'].unique()
        for platform in platforms:
            db_manager.update_data_summary(platform)
        
        # Initialize vector store and index summaries
        logger.info("Indexing data in vector store...")
        vector_store = create_vector_store(
            store_type="chroma",
            persist_directory=settings.chroma_persist_directory,
            collection_name="argo_summaries"
        )
        vector_manager = ArgoVectorManager(vector_store)
        
        # Get summaries and index them
        summaries = []
        for platform in platforms:
            summary = db_manager.get_float_summary(platform)
            if summary:
                summaries.append(summary)
        
        if summaries:
            vector_manager.index_float_summaries(summaries)
            logger.info(f"Indexed {len(summaries)} float summaries in vector store")
        
        logger.info("✅ Sample data loaded successfully!")
        
        # Print summary statistics
        logger.info("\n📊 Data Summary:")
        logger.info(f"  • Total floats: {len(platforms)}")
        logger.info(f"  • Total profiles: {len(df)}")
        logger.info(f"  • Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  • Latitude range: {df['latitude'].min():.2f}° to {df['latitude'].max():.2f}°")
        logger.info(f"  • Longitude range: {df['longitude'].min():.2f}° to {df['longitude'].max():.2f}°")
        logger.info(f"  • Temperature range: {df['temperature'].min():.2f}°C to {df['temperature'].max():.2f}°C")
        logger.info(f"  • Salinity range: {df['salinity'].min():.2f} to {df['salinity'].max():.2f} PSU")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load sample data: {e}")
        return False


def main():
    """Main function to load sample data."""
    logger.info("🌊 Loading Sample ARGO Data for FloatChat")
    logger.info("=" * 50)
    
    success = load_sample_data()
    
    logger.info("=" * 50)
    if success:
        logger.info("🎉 Sample data loaded successfully!")
        logger.info("\nYou can now:")
        logger.info("1. Start the API server: python -m app.api.main")
        logger.info("2. Start the Streamlit app: streamlit run app/main.py")
        logger.info("3. Try queries like:")
        logger.info("   • 'Show me temperature profiles in the Arabian Sea'")
        logger.info("   • 'What floats are active in the Indian Ocean?'")
        logger.info("   • 'Compare salinity data from the last 6 months'")
    else:
        logger.error("❌ Failed to load sample data")
        sys.exit(1)


if __name__ == "__main__":
    main()
