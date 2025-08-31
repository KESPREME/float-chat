"""Database models and operations for ARGO float data."""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class ArgoProfile(Base):
    """ARGO profile data model."""
    
    __tablename__ = "argo_profiles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform_number = Column(String(50), nullable=False, index=True)
    cycle_number = Column(Integer, nullable=False)
    data_centre = Column(String(10))
    data_mode = Column(String(1))
    date = Column(DateTime, nullable=False, index=True)
    position_qc = Column(String(1))
    
    # Geospatial
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    
    # Measurements
    pressure = Column(Float)
    temperature = Column(Float)
    salinity = Column(Float)
    
    # Quality control
    pressure_qc = Column(String(1))
    temperature_qc = Column(String(1))
    salinity_qc = Column(String(1))
    
    # Metadata
    file_path = Column(Text)
    processed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)


class ArgoFloat(Base):
    """ARGO float metadata model."""
    
    __tablename__ = "argo_floats"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform_number = Column(String(50), unique=True, nullable=False, index=True)
    wmo_id = Column(String(20), index=True)
    
    # Deployment info
    deployment_date = Column(DateTime)
    deployment_latitude = Column(Float)
    deployment_longitude = Column(Float)
    
    # Status
    status = Column(String(20))  # active, inactive, dead
    last_location_date = Column(DateTime)
    last_latitude = Column(Float)
    last_longitude = Column(Float)
    
    # Technical specs
    float_type = Column(String(50))
    sensor_types = Column(Text)  # JSON array of sensor types
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DataSummary(Base):
    """Summary statistics for vector database indexing."""
    
    __tablename__ = "data_summaries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform_number = Column(String(50), nullable=False, index=True)
    
    # Temporal range
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Spatial range
    min_latitude = Column(Float, nullable=False)
    max_latitude = Column(Float, nullable=False)
    min_longitude = Column(Float, nullable=False)
    max_longitude = Column(Float, nullable=False)
    
    # Data statistics
    total_profiles = Column(Integer, nullable=False)
    depth_range_min = Column(Float)
    depth_range_max = Column(Float)
    temperature_range_min = Column(Float)
    temperature_range_max = Column(Float)
    salinity_range_min = Column(Float)
    salinity_range_max = Column(Float)
    
    # Text summary for RAG
    summary_text = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DatabaseManager:
    """Database connection and operations manager."""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def insert_profiles_batch(self, profiles_data: list) -> int:
        """Insert multiple profiles in batch.
        
        Args:
            profiles_data: List of profile dictionaries
            
        Returns:
            Number of profiles inserted
        """
        session = self.get_session()
        try:
            profiles = []
            for data in profiles_data:
                profile = ArgoProfile(**data)
                profiles.append(profile)
            
            session.add_all(profiles)
            session.commit()
            
            logger.info(f"Inserted {len(profiles)} profiles")
            return len(profiles)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error inserting profiles: {e}")
            raise
        finally:
            session.close()
    
    def get_profiles_by_region(self, 
                              min_lat: float, max_lat: float,
                              min_lon: float, max_lon: float,
                              start_date: datetime = None,
                              end_date: datetime = None,
                              limit: int = 1000) -> list:
        """Get profiles within a geographic region and time range.
        
        Args:
            min_lat, max_lat: Latitude bounds
            min_lon, max_lon: Longitude bounds  
            start_date, end_date: Optional time bounds
            limit: Maximum number of results
            
        Returns:
            List of profile records
        """
        session = self.get_session()
        try:
            query = session.query(ArgoProfile).filter(
                ArgoProfile.latitude >= min_lat,
                ArgoProfile.latitude <= max_lat,
                ArgoProfile.longitude >= min_lon,
                ArgoProfile.longitude <= max_lon
            )
            
            if start_date:
                query = query.filter(ArgoProfile.date >= start_date)
            if end_date:
                query = query.filter(ArgoProfile.date <= end_date)
            
            return query.limit(limit).all()
            
        finally:
            session.close()
    
    def get_float_summary(self, platform_number: str) -> dict:
        """Get summary statistics for a specific float.
        
        Args:
            platform_number: Float platform number
            
        Returns:
            Dictionary with summary statistics
        """
        session = self.get_session()
        try:
            summary = session.query(DataSummary).filter(
                DataSummary.platform_number == platform_number
            ).first()
            
            if summary:
                return {
                    'platform_number': summary.platform_number,
                    'temporal_range': {
                        'start': summary.start_date,
                        'end': summary.end_date
                    },
                    'spatial_range': {
                        'latitude': [summary.min_latitude, summary.max_latitude],
                        'longitude': [summary.min_longitude, summary.max_longitude]
                    },
                    'total_profiles': summary.total_profiles,
                    'depth_range': [summary.depth_range_min, summary.depth_range_max],
                    'temperature_range': [summary.temperature_range_min, summary.temperature_range_max],
                    'salinity_range': [summary.salinity_range_min, summary.salinity_range_max],
                    'summary_text': summary.summary_text
                }
            return None
            
        finally:
            session.close()
    
    def update_data_summary(self, platform_number: str):
        """Update summary statistics for a float platform.
        
        Args:
            platform_number: Float platform number to update
        """
        session = self.get_session()
        try:
            # Calculate statistics from profiles
            from sqlalchemy import func
            
            stats = session.query(
                func.count(ArgoProfile.id).label('total_profiles'),
                func.min(ArgoProfile.date).label('start_date'),
                func.max(ArgoProfile.date).label('end_date'),
                func.min(ArgoProfile.latitude).label('min_lat'),
                func.max(ArgoProfile.latitude).label('max_lat'),
                func.min(ArgoProfile.longitude).label('min_lon'),
                func.max(ArgoProfile.longitude).label('max_lon'),
                func.min(ArgoProfile.pressure).label('min_depth'),
                func.max(ArgoProfile.pressure).label('max_depth'),
                func.min(ArgoProfile.temperature).label('min_temp'),
                func.max(ArgoProfile.temperature).label('max_temp'),
                func.min(ArgoProfile.salinity).label('min_sal'),
                func.max(ArgoProfile.salinity).label('max_sal')
            ).filter(ArgoProfile.platform_number == platform_number).first()
            
            if stats.total_profiles > 0:
                # Generate summary text
                summary_text = f"""
                Float {platform_number} collected {stats.total_profiles} profiles 
                from {stats.start_date.strftime('%Y-%m-%d')} to {stats.end_date.strftime('%Y-%m-%d')}.
                Geographic coverage: {stats.min_lat:.2f}°N to {stats.max_lat:.2f}°N, 
                {stats.min_lon:.2f}°E to {stats.max_lon:.2f}°E.
                Depth range: {stats.min_depth:.1f}m to {stats.max_depth:.1f}m.
                Temperature range: {stats.min_temp:.2f}°C to {stats.max_temp:.2f}°C.
                Salinity range: {stats.min_sal:.2f} to {stats.max_sal:.2f} PSU.
                """.strip()
                
                # Update or create summary
                existing = session.query(DataSummary).filter(
                    DataSummary.platform_number == platform_number
                ).first()
                
                if existing:
                    existing.total_profiles = stats.total_profiles
                    existing.start_date = stats.start_date
                    existing.end_date = stats.end_date
                    existing.min_latitude = stats.min_lat
                    existing.max_latitude = stats.max_lat
                    existing.min_longitude = stats.min_lon
                    existing.max_longitude = stats.max_lon
                    existing.depth_range_min = stats.min_depth
                    existing.depth_range_max = stats.max_depth
                    existing.temperature_range_min = stats.min_temp
                    existing.temperature_range_max = stats.max_temp
                    existing.salinity_range_min = stats.min_sal
                    existing.salinity_range_max = stats.max_sal
                    existing.summary_text = summary_text
                    existing.updated_at = datetime.utcnow()
                else:
                    summary = DataSummary(
                        platform_number=platform_number,
                        total_profiles=stats.total_profiles,
                        start_date=stats.start_date,
                        end_date=stats.end_date,
                        min_latitude=stats.min_lat,
                        max_latitude=stats.max_lat,
                        min_longitude=stats.min_lon,
                        max_longitude=stats.max_lon,
                        depth_range_min=stats.min_depth,
                        depth_range_max=stats.max_depth,
                        temperature_range_min=stats.min_temp,
                        temperature_range_max=stats.max_temp,
                        salinity_range_min=stats.min_sal,
                        salinity_range_max=stats.max_sal,
                        summary_text=summary_text
                    )
                    session.add(summary)
                
                session.commit()
                logger.info(f"Updated summary for platform {platform_number}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating summary for {platform_number}: {e}")
            raise
        finally:
            session.close()
