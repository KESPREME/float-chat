"""NetCDF file processing for ARGO float data."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ArgoNetCDFProcessor:
    """Process ARGO NetCDF files and extract structured data."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def process_profile_file(self, file_path: str) -> Optional[Dict]:
        """Process a single ARGO profile NetCDF file.
        
        Args:
            file_path: Path to the NetCDF file
            
        Returns:
            Dictionary containing processed profile data
        """
        try:
            with xr.open_dataset(file_path) as ds:
                # Extract metadata
                metadata = self._extract_metadata(ds)
                
                # Extract profile data
                profiles = self._extract_profiles(ds)
                
                # Extract geospatial information
                geospatial = self._extract_geospatial(ds)
                
                return {
                    'metadata': metadata,
                    'profiles': profiles,
                    'geospatial': geospatial,
                    'file_path': file_path,
                    'processed_at': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def _extract_metadata(self, ds: xr.Dataset) -> Dict:
        """Extract metadata from NetCDF dataset."""
        metadata = {}
        
        # Platform information
        if 'PLATFORM_NUMBER' in ds.variables:
            metadata['platform_number'] = str(ds['PLATFORM_NUMBER'].values)
        
        # Cycle information
        if 'CYCLE_NUMBER' in ds.variables:
            metadata['cycle_number'] = int(ds['CYCLE_NUMBER'].values)
        
        # Data center and mode
        if 'DATA_CENTRE' in ds.variables:
            metadata['data_centre'] = str(ds['DATA_CENTRE'].values)
        
        if 'DATA_MODE' in ds.variables:
            metadata['data_mode'] = str(ds['DATA_MODE'].values)
        
        # Dates
        if 'JULD' in ds.variables:
            juld = ds['JULD'].values
            if not np.isnan(juld):
                # Convert JULD (days since 1950-01-01) to datetime
                reference_date = pd.Timestamp('1950-01-01')
                metadata['date'] = (reference_date + pd.Timedelta(days=float(juld))).isoformat()
        
        # Quality control flags
        if 'POSITION_QC' in ds.variables:
            metadata['position_qc'] = str(ds['POSITION_QC'].values)
        
        return metadata
    
    def _extract_profiles(self, ds: xr.Dataset) -> List[Dict]:
        """Extract profile data (temperature, salinity, pressure)."""
        profiles = []
        
        # Get dimensions
        n_levels = ds.dims.get('N_LEVELS', 0)
        n_prof = ds.dims.get('N_PROF', 1)
        
        for prof_idx in range(n_prof):
            profile_data = {}
            
            # Pressure
            if 'PRES' in ds.variables:
                pres = ds['PRES'].values
                if pres.ndim > 1:
                    pres = pres[prof_idx, :]
                profile_data['pressure'] = pres.tolist()
            
            # Temperature
            if 'TEMP' in ds.variables:
                temp = ds['TEMP'].values
                if temp.ndim > 1:
                    temp = temp[prof_idx, :]
                profile_data['temperature'] = temp.tolist()
            
            # Salinity
            if 'PSAL' in ds.variables:
                psal = ds['PSAL'].values
                if psal.ndim > 1:
                    psal = psal[prof_idx, :]
                profile_data['salinity'] = psal.tolist()
            
            # Quality control flags
            if 'PRES_QC' in ds.variables:
                pres_qc = ds['PRES_QC'].values
                if pres_qc.ndim > 1:
                    pres_qc = pres_qc[prof_idx, :]
                profile_data['pressure_qc'] = [str(x) for x in pres_qc]
            
            if 'TEMP_QC' in ds.variables:
                temp_qc = ds['TEMP_QC'].values
                if temp_qc.ndim > 1:
                    temp_qc = temp_qc[prof_idx, :]
                profile_data['temperature_qc'] = [str(x) for x in temp_qc]
            
            if 'PSAL_QC' in ds.variables:
                psal_qc = ds['PSAL_QC'].values
                if psal_qc.ndim > 1:
                    psal_qc = psal_qc[prof_idx, :]
                profile_data['salinity_qc'] = [str(x) for x in psal_qc]
            
            profiles.append(profile_data)
        
        return profiles
    
    def _extract_geospatial(self, ds: xr.Dataset) -> Dict:
        """Extract geospatial information."""
        geospatial = {}
        
        # Latitude
        if 'LATITUDE' in ds.variables:
            lat = ds['LATITUDE'].values
            if hasattr(lat, '__len__') and len(lat) > 0:
                geospatial['latitude'] = float(lat[0]) if lat.ndim > 0 else float(lat)
            else:
                geospatial['latitude'] = float(lat)
        
        # Longitude
        if 'LONGITUDE' in ds.variables:
            lon = ds['LONGITUDE'].values
            if hasattr(lon, '__len__') and len(lon) > 0:
                geospatial['longitude'] = float(lon[0]) if lon.ndim > 0 else float(lon)
            else:
                geospatial['longitude'] = float(lon)
        
        return geospatial
    
    def process_directory(self, directory_path: str) -> List[Dict]:
        """Process all NetCDF files in a directory.
        
        Args:
            directory_path: Path to directory containing NetCDF files
            
        Returns:
            List of processed profile dictionaries
        """
        directory = Path(directory_path)
        processed_data = []
        
        # Find all NetCDF files
        netcdf_files = list(directory.glob("*.nc"))
        
        logger.info(f"Found {len(netcdf_files)} NetCDF files to process")
        
        for file_path in netcdf_files:
            logger.info(f"Processing {file_path.name}")
            result = self.process_profile_file(str(file_path))
            if result:
                processed_data.append(result)
        
        logger.info(f"Successfully processed {len(processed_data)} files")
        return processed_data
    
    def to_dataframe(self, processed_data: List[Dict]) -> pd.DataFrame:
        """Convert processed data to pandas DataFrame.
        
        Args:
            processed_data: List of processed profile dictionaries
            
        Returns:
            Flattened DataFrame with profile measurements
        """
        rows = []
        
        for data in processed_data:
            metadata = data['metadata']
            geospatial = data['geospatial']
            
            for profile in data['profiles']:
                # Get the length of arrays (should be consistent)
                n_measurements = len(profile.get('pressure', []))
                
                for i in range(n_measurements):
                    row = {
                        # Metadata
                        'platform_number': metadata.get('platform_number'),
                        'cycle_number': metadata.get('cycle_number'),
                        'data_centre': metadata.get('data_centre'),
                        'data_mode': metadata.get('data_mode'),
                        'date': metadata.get('date'),
                        'position_qc': metadata.get('position_qc'),
                        
                        # Geospatial
                        'latitude': geospatial.get('latitude'),
                        'longitude': geospatial.get('longitude'),
                        
                        # Measurements
                        'pressure': profile.get('pressure', [None] * n_measurements)[i],
                        'temperature': profile.get('temperature', [None] * n_measurements)[i],
                        'salinity': profile.get('salinity', [None] * n_measurements)[i],
                        
                        # Quality control
                        'pressure_qc': profile.get('pressure_qc', [None] * n_measurements)[i],
                        'temperature_qc': profile.get('temperature_qc', [None] * n_measurements)[i],
                        'salinity_qc': profile.get('salinity_qc', [None] * n_measurements)[i],
                        
                        # File info
                        'file_path': data['file_path'],
                        'processed_at': data['processed_at']
                    }
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_to_parquet(self, df: pd.DataFrame, output_path: str):
        """Save DataFrame to Parquet format.
        
        Args:
            df: DataFrame to save
            output_path: Path for output Parquet file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_file, index=False)
        logger.info(f"Saved {len(df)} records to {output_file}")
