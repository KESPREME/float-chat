"""Configuration settings for FloatChat application."""

import os
from pathlib import Path
from typing import Optional

# Try to import pydantic, provide fallback if not available
try:
    from pydantic import Field
    from pydantic_settings import BaseSettings
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    # Create simple settings class without pydantic
    class BaseSettings:
        pass
    def Field(default=None, env=None):
        return default


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    def __init__(self, **kwargs):
        if HAS_PYDANTIC:
            super().__init__(**kwargs)
        else:
            # Manual environment loading without pydantic
            self._load_env_vars()
        self._create_directories()
    
    def _load_env_vars(self):
        """Load environment variables manually when pydantic not available."""
        # Database Configuration
        self.database_url = os.getenv("DATABASE_URL", "postgresql://floatchat_user:password@localhost:5432/floatchat")
        self.postgres_user = os.getenv("POSTGRES_USER", "floatchat_user")
        self.postgres_password = os.getenv("POSTGRES_PASSWORD", "password")
        self.postgres_db = os.getenv("POSTGRES_DB", "floatchat")
        
        # Google Gemini Configuration
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        
        # Vector Database
        self.chroma_persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
        self.faiss_index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
        
        # Application Settings
        self.app_host = os.getenv("APP_HOST", "localhost")
        self.app_port = int(os.getenv("APP_PORT", "8000"))
        self.streamlit_port = int(os.getenv("STREAMLIT_PORT", "8501"))
        self.debug = os.getenv("DEBUG", "True").lower() == "true"
        
        # Data Paths
        self.argo_data_path = os.getenv("ARGO_DATA_PATH", "./data/argo_netcdf/")
        self.processed_data_path = os.getenv("PROCESSED_DATA_PATH", "./data/processed/")
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = os.getenv("LOG_FILE", "./logs/floatchat.log")
    
    # Database Configuration (only used if pydantic available)
    if HAS_PYDANTIC:
        database_url: str = Field(
            default="postgresql://floatchat_user:password@localhost:5432/floatchat",
            env="DATABASE_URL"
        )
        postgres_user: str = Field(default="floatchat_user", env="POSTGRES_USER")
        postgres_password: str = Field(default="password", env="POSTGRES_PASSWORD")
        postgres_db: str = Field(default="floatchat", env="POSTGRES_DB")
        
        # Google Gemini Configuration
        google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
        gemini_model: str = Field(default="gemini-1.5-pro", env="GEMINI_MODEL")
        
        # Vector Database
        chroma_persist_directory: str = Field(
            default="./data/chroma_db", 
            env="CHROMA_PERSIST_DIRECTORY"
        )
        faiss_index_path: str = Field(
            default="./data/faiss_index", 
            env="FAISS_INDEX_PATH"
        )
        
        # Application Settings
        app_host: str = Field(default="localhost", env="APP_HOST")
        app_port: int = Field(default=8000, env="APP_PORT")
        streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
        debug: bool = Field(default=True, env="DEBUG")
        
        # Data Paths
        argo_data_path: str = Field(default="./data/argo_netcdf/", env="ARGO_DATA_PATH")
        processed_data_path: str = Field(default="./data/processed/", env="PROCESSED_DATA_PATH")
        
        # Logging
        log_level: str = Field(default="INFO", env="LOG_LEVEL")
        log_file: str = Field(default="./logs/floatchat.log", env="LOG_FILE")
        
        class Config:
            env_file = ".env"
            case_sensitive = False
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.chroma_persist_directory,
            self.faiss_index_path,
            self.argo_data_path,
            self.processed_data_path,
            os.path.dirname(self.log_file)
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
