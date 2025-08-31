"""Initialize database and setup script for FloatChat."""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from data.storage.database import DatabaseManager
from data.storage.vector_store import create_vector_store, ArgoVectorManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database():
    """Initialize PostgreSQL database tables."""
    logger.info("Initializing database...")
    
    try:
        db_manager = DatabaseManager(settings.database_url)
        db_manager.create_tables()
        logger.info("✅ Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        return False


def init_vector_store():
    """Initialize vector database."""
    logger.info("Initializing vector store...")
    
    try:
        # Create ChromaDB vector store
        vector_store = create_vector_store(
            store_type="chroma",
            persist_directory=settings.chroma_persist_directory,
            collection_name="argo_summaries"
        )
        
        # Initialize manager
        vector_manager = ArgoVectorManager(vector_store)
        
        # Add some sample regional data
        sample_regions = [
            {
                "region_name": "Arabian Sea",
                "bounds": {"min_lat": 10, "max_lat": 25, "min_lon": 50, "max_lon": 80},
                "description": "Semi-enclosed sea in the northwestern Indian Ocean, known for strong monsoon winds and upwelling"
            },
            {
                "region_name": "Equatorial Pacific",
                "bounds": {"min_lat": -10, "max_lat": 10, "min_lon": 120, "max_lon": -70},
                "description": "Tropical Pacific region important for ENSO dynamics and climate variability"
            },
            {
                "region_name": "North Atlantic",
                "bounds": {"min_lat": 30, "max_lat": 70, "min_lon": -80, "max_lon": 20},
                "description": "Northern Atlantic Ocean region with deep water formation and Gulf Stream circulation"
            }
        ]
        
        for region in sample_regions:
            vector_manager.add_region_summary(
                region["region_name"],
                region["bounds"],
                region["description"]
            )
        
        logger.info("✅ Vector store initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize vector store: {e}")
        return False


def create_sample_env():
    """Create .env file from example if it doesn't exist."""
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    
    if not env_file.exists() and env_example.exists():
        logger.info("Creating .env file from example...")
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        logger.info("✅ .env file created. Please update with your configuration.")
    elif env_file.exists():
        logger.info("✅ .env file already exists")
    else:
        logger.warning("⚠️ No .env.example file found")


def main():
    """Main initialization function."""
    logger.info("🌊 Initializing FloatChat Application")
    logger.info("=" * 50)
    
    # Create .env file if needed
    create_sample_env()
    
    # Initialize database
    db_success = init_database()
    
    # Initialize vector store
    vector_success = init_vector_store()
    
    # Summary
    logger.info("=" * 50)
    if db_success and vector_success:
        logger.info("🎉 FloatChat initialization completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Update .env file with your OpenAI API key and database credentials")
        logger.info("2. Start the API server: python -m app.api.main")
        logger.info("3. Start the Streamlit app: streamlit run app/main.py")
        logger.info("4. Upload some ARGO NetCDF data using the /data/upload endpoint")
    else:
        logger.error("❌ Initialization failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
