"""FastAPI backend for FloatChat application."""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
import asyncio

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import settings
from rag.rag_pipeline import create_rag_pipeline, QueryCache
from data.storage.database import DatabaseManager
from data.ingestion.netcdf_processor import ArgoNetCDFProcessor

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FloatChat API",
    description="AI-powered conversational interface for ARGO ocean data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
rag_pipeline = None
query_cache = QueryCache(max_size=50)
db_manager = None


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    include_visualization: bool = True
    max_results: int = 1000


class QueryResponse(BaseModel):
    success: bool
    user_query: str
    response: Optional[str] = None
    sql_query: Optional[str] = None
    row_count: Optional[int] = None
    execution_time: Optional[float] = None
    visualization_data: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    error: Optional[str] = None


class DataUploadRequest(BaseModel):
    file_paths: List[str]
    process_immediately: bool = True


class FloatSummaryResponse(BaseModel):
    platform_number: str
    total_profiles: int
    temporal_range: Dict[str, str]
    spatial_range: Dict[str, List[float]]
    summary_text: str


# Dependency to get RAG pipeline
async def get_rag_pipeline():
    global rag_pipeline
    if rag_pipeline is None:
        if not settings.google_api_key:
            raise HTTPException(
                status_code=500, 
                detail="Google API key not configured"
            )
        
        vector_config = {
            "store_type": "chroma",
            "persist_directory": settings.chroma_persist_directory,
            "collection_name": "argo_summaries"
        }
        
        rag_pipeline = create_rag_pipeline(
            database_url=settings.database_url,
            vector_store_config=vector_config,
            google_api_key=settings.google_api_key
        )
    
    return rag_pipeline


# Dependency to get database manager
async def get_db_manager():
    global db_manager
    if db_manager is None:
        try:
            db_manager = DatabaseManager(settings.database_url)
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            # Return None for demo mode without database
            return None
    return db_manager


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting FloatChat API...")
    
    # Initialize database tables (optional for demo mode)
    try:
        db = DatabaseManager(settings.database_url)
        db.create_tables()
        logger.info("Database tables initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        logger.info("Running in demo mode without database")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "FloatChat API",
        "version": "1.0.0",
        "description": "AI-powered conversational interface for ARGO ocean data",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection if available
        db = await get_db_manager()
        if db is not None:
            try:
                session = db.get_session()
                session.execute("SELECT 1")
                session.close()
                db_status = "connected"
            except Exception:
                db_status = "disconnected"
        else:
            db_status = "disconnected (demo mode)"
        
        return {
            "status": "healthy",
            "database": db_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy (demo mode)",
                "database": "disconnected",
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, pipeline=Depends(get_rag_pipeline)):
    """Process natural language query through RAG pipeline."""
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Check cache first
        import hashlib
        query_hash = hashlib.md5(request.query.encode()).hexdigest()
        cached_result = query_cache.get(query_hash)
        
        if cached_result:
            logger.info("Returning cached result")
            return QueryResponse(**cached_result)
        
        # Process query through RAG pipeline
        result = pipeline.process_query(request.query)
        
        # Prepare response
        response_data = {
            "success": result["success"],
            "user_query": result["user_query"],
            "response": result.get("response"),
            "sql_query": result.get("sql_query"),
            "suggestions": result.get("suggestions", [])
        }
        
        if result["success"]:
            query_results = result.get("query_results", {})
            response_data.update({
                "row_count": query_results.get("row_count", 0),
                "execution_time": query_results.get("execution_time", 0),
                "visualization_data": result.get("visualization_data") if request.include_visualization else None
            })
        else:
            response_data["error"] = result.get("error")
        
        # Cache successful results
        if result["success"]:
            query_cache.put(query_hash, response_data)
        
        return QueryResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/floats")
async def list_floats(
    limit: int = 50,
    offset: int = 0,
    db=Depends(get_db_manager)
):
    """List available ARGO floats with basic information."""
    try:
        # Return demo data if no database connection
        if db is None:
            demo_floats = []
            for i in range(min(limit, 20)):
                platform_num = f"590{1000 + i}"
                demo_floats.append({
                    "platform_number": platform_num,
                    "profile_count": 45 + (i * 3),
                    "first_date": "2023-01-15",
                    "last_date": "2024-08-30",
                    "avg_lat": 15.5 + (i * 0.5),
                    "avg_lon": 68.2 + (i * 0.3),
                    "status": "active"
                })
            
            return {
                "floats": demo_floats,
                "total_count": 150,
                "limit": limit,
                "offset": offset,
                "data_source": "demo_mode"
            }
        
        session = db.get_session()
        
        # Get distinct platform numbers with basic stats
        from sqlalchemy import text
        query = text("""
        SELECT 
            platform_number,
            COUNT(*) as profile_count,
            MIN(date) as first_date,
            MAX(date) as last_date,
            AVG(latitude) as avg_lat,
            AVG(longitude) as avg_lon
        FROM argo_profiles 
        GROUP BY platform_number
        ORDER BY platform_number
        LIMIT :limit OFFSET :offset
        """)
        
        result = session.execute(query, {"limit": limit, "offset": offset})
        floats = []
        
        for row in result:
            floats.append({
                "platform_number": row[0],
                "profile_count": row[1],
                "first_date": row[2].isoformat() if row[2] else None,
                "last_date": row[3].isoformat() if row[3] else None,
                "average_position": {
                    "latitude": float(row[4]) if row[4] else None,
                    "longitude": float(row[5]) if row[5] else None
                }
            })
        
        session.close()
        
        return {
            "floats": floats,
            "total": len(floats),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing floats: {e}")
        # Return demo data on error
        demo_floats = []
        for i in range(min(limit, 10)):
            platform_num = f"590{2000 + i}"
            demo_floats.append({
                "platform_number": platform_num,
                "profile_count": 35 + (i * 2),
                "first_date": "2023-06-01",
                "last_date": "2024-08-31",
                "avg_lat": 12.0 + (i * 0.8),
                "avg_lon": 65.0 + (i * 0.4),
                "status": "active"
            })
        
        return {
            "floats": demo_floats,
            "total_count": 150,
            "limit": limit,
            "offset": offset,
            "data_source": "demo_mode_fallback"
        }


@app.get("/floats/{platform_number}/summary", response_model=FloatSummaryResponse)
async def get_float_summary(platform_number: str, db=Depends(get_db_manager)):
    """Get detailed summary for a specific float."""
    try:
        # Return demo data if no database connection
        if db is None:
            return FloatSummaryResponse(
                platform_number=platform_number,
                total_profiles=78,
                temporal_range={
                    "start": "2023-03-15",
                    "end": "2024-08-30"
                },
                spatial_range={
                    "latitude": [12.5, 18.2],
                    "longitude": [65.1, 72.8]
                },
                summary_text=f"ARGO float {platform_number} has collected 78 profiles over 17 months in the Arabian Sea region, measuring temperature, salinity, and pressure from surface to 2000m depth."
            )
        
        summary = db.get_float_summary(platform_number)
        
        if not summary:
            raise HTTPException(
                status_code=404, 
                detail=f"Float {platform_number} not found"
            )
        
        return FloatSummaryResponse(
            platform_number=summary["platform_number"],
            total_profiles=summary["total_profiles"],
            temporal_range={
                "start": summary["temporal_range"]["start"].isoformat(),
                "end": summary["temporal_range"]["end"].isoformat()
            },
            spatial_range=summary["spatial_range"],
            summary_text=summary["summary_text"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting float summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/upload")
async def upload_netcdf_data(
    request: DataUploadRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db_manager)
):
    """Upload and process NetCDF files."""
    try:
        if request.process_immediately:
            # Process in background
            background_tasks.add_task(
                process_netcdf_files,
                request.file_paths,
                db
            )
            
            return {
                "message": f"Processing {len(request.file_paths)} files in background",
                "file_count": len(request.file_paths),
                "status": "processing"
            }
        else:
            # Process synchronously (for small datasets)
            result = await process_netcdf_files(request.file_paths, db)
            return result
            
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_netcdf_files(file_paths: List[str], db: DatabaseManager):
    """Process NetCDF files and store in database."""
    try:
        processor = ArgoNetCDFProcessor(settings.argo_data_path)
        
        total_processed = 0
        for file_path in file_paths:
            logger.info(f"Processing {file_path}")
            
            # Process single file
            result = processor.process_profile_file(file_path)
            if result:
                # Convert to DataFrame and insert into database
                df = processor.to_dataframe([result])
                
                # Convert DataFrame to list of dictionaries
                records = df.to_dict('records')
                
                # Clean up records for database insertion
                clean_records = []
                for record in records:
                    clean_record = {}
                    for key, value in record.items():
                        if value is not None and str(value) != 'nan':
                            if key == 'date' and isinstance(value, str):
                                clean_record[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                            else:
                                clean_record[key] = value
                    clean_records.append(clean_record)
                
                # Insert into database
                inserted = db.insert_profiles_batch(clean_records)
                total_processed += inserted
                
                # Update summary statistics
                if clean_records:
                    platform_number = clean_records[0].get('platform_number')
                    if platform_number:
                        db.update_data_summary(platform_number)
        
        logger.info(f"Successfully processed {total_processed} profiles from {len(file_paths)} files")
        
        return {
            "message": "Files processed successfully",
            "files_processed": len(file_paths),
            "profiles_inserted": total_processed,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Error processing NetCDF files: {e}")
        raise


@app.get("/data/stats")
async def get_data_statistics(db=Depends(get_db_manager)):
    """Get overall data statistics."""
    try:
        # Return demo data if no database connection
        if db is None:
            return {
                "total_floats": 150,
                "total_profiles": 12500,
                "temporal_range": {
                    "start": "2020-01-01",
                    "end": "2024-08-31"
                },
                "spatial_range": {
                    "latitude": [-60.0, 60.0],
                    "longitude": [30.0, 120.0]
                },
                "data_source": "demo_mode"
            }
        
        session = db.get_session()
        
        # Get basic statistics
        from sqlalchemy import text
        stats_query = text("""
        SELECT 
            COUNT(DISTINCT platform_number) as total_floats,
            COUNT(*) as total_profiles,
            MIN(date) as earliest_date,
            MAX(date) as latest_date,
            MIN(latitude) as min_lat,
            MAX(latitude) as max_lat,
            MIN(longitude) as min_lon,
            MAX(longitude) as max_lon
        FROM argo_profiles
        """)
        
        result = session.execute(stats_query).fetchone()
        
        stats = {
            "total_floats": result[0] or 0,
            "total_profiles": result[1] or 0,
            "temporal_coverage": {
                "earliest_date": result[2].isoformat() if result[2] else None,
                "latest_date": result[3].isoformat() if result[3] else None
            },
            "spatial_coverage": {
                "latitude_range": [float(result[4]) if result[4] else None, 
                                 float(result[5]) if result[5] else None],
                "longitude_range": [float(result[6]) if result[6] else None, 
                                  float(result[7]) if result[7] else None]
            }
        }
        
        session.close()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting data statistics: {e}")
        # Return demo data on error
        return {
            "total_floats": 150,
            "total_profiles": 12500,
            "temporal_range": {
                "start": "2020-01-01",
                "end": "2024-08-31"
            },
            "spatial_range": {
                "latitude": [-60.0, 60.0],
                "longitude": [30.0, 120.0]
            },
            "data_source": "demo_mode_fallback"
        }


@app.delete("/cache")
async def clear_cache():
    """Clear query cache."""
    query_cache.clear()
    return {"message": "Cache cleared successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug
    )
