"""Natural language to SQL query translation using LLMs."""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import re
import json
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseMessage

logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """Context information for query translation."""
    available_tables: List[str]
    table_schemas: Dict[str, List[str]]
    sample_data: Optional[Dict] = None
    geographic_regions: Optional[Dict] = None


class NLToSQLTranslator:
    """Translates natural language queries to SQL for ARGO data."""
    
    def __init__(self, google_api_key: str, model_name: str = "gemini-1.5-pro"):
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=google_api_key,
            model=model_name,
            temperature=0.1
        )
        
        # Define geographic regions for common queries
        self.geographic_regions = {
            "equator": {"min_lat": -5, "max_lat": 5, "min_lon": -180, "max_lon": 180},
            "equatorial": {"min_lat": -10, "max_lat": 10, "min_lon": -180, "max_lon": 180},
            "arabian sea": {"min_lat": 10, "max_lat": 25, "min_lon": 50, "max_lon": 80},
            "indian ocean": {"min_lat": -40, "max_lat": 30, "min_lon": 20, "max_lon": 120},
            "pacific": {"min_lat": -60, "max_lat": 60, "min_lon": 120, "max_lon": -70},
            "atlantic": {"min_lat": -60, "max_lat": 70, "min_lon": -80, "max_lon": 20},
            "mediterranean": {"min_lat": 30, "max_lat": 46, "min_lon": -6, "max_lon": 36},
            "north atlantic": {"min_lat": 30, "max_lat": 70, "min_lon": -80, "max_lon": 20},
            "south pacific": {"min_lat": -60, "max_lat": 0, "min_lon": 120, "max_lon": -70}
        }
        
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for SQL translation."""
        return """You are an expert SQL query generator for ARGO oceanographic float data. 
Your task is to convert natural language queries into precise SQL queries.

AVAILABLE TABLES:
1. argo_profiles: Individual measurement records
   - Columns: id, platform_number, cycle_number, date, latitude, longitude, 
             pressure, temperature, salinity, pressure_qc, temperature_qc, salinity_qc
2. argo_floats: Float metadata
   - Columns: id, platform_number, wmo_id, deployment_date, status, last_location_date
3. data_summaries: Aggregated statistics per float
   - Columns: platform_number, start_date, end_date, total_profiles, 
             min_latitude, max_latitude, min_longitude, max_longitude

IMPORTANT GUIDELINES:
1. Always use proper date formatting (YYYY-MM-DD) for date comparisons
2. Latitude: -90 to 90 (negative = South, positive = North)
3. Longitude: -180 to 180 (negative = West, positive = East)
4. Pressure in decibars (roughly equivalent to depth in meters)
5. Temperature in Celsius, Salinity in PSU (Practical Salinity Units)
6. Use appropriate LIMIT clauses to prevent excessive results
7. Include quality control filters when appropriate (qc flags: '1'=good, '2'=probably good)

GEOGRAPHIC REGIONS (use these bounds when mentioned):
- Equator: latitude between -5 and 5
- Arabian Sea: lat 10-25, lon 50-80
- Indian Ocean: lat -40-30, lon 20-120
- Pacific: lat -60-60, lon 120 to -70 (crosses dateline)
- Atlantic: lat -60-70, lon -80-20

RESPONSE FORMAT:
Return only valid PostgreSQL SQL queries. Do not include explanations or markdown formatting.
Use appropriate JOINs when querying multiple tables.
Always consider performance - use indexes on platform_number, date, latitude, longitude.

EXAMPLES:
Query: "Show salinity profiles near the equator in March 2023"
SQL: SELECT platform_number, latitude, longitude, date, pressure, salinity 
     FROM argo_profiles 
     WHERE latitude BETWEEN -5 AND 5 
     AND date BETWEEN '2023-03-01' AND '2023-03-31'
     AND salinity IS NOT NULL 
     AND salinity_qc IN ('1', '2')
     ORDER BY date, pressure 
     LIMIT 1000;

Query: "What are the nearest floats to 15°N, 65°E?"
SQL: SELECT platform_number, latitude, longitude, 
     SQRT(POW(latitude - 15, 2) + POW(longitude - 65, 2)) as distance
     FROM (SELECT DISTINCT platform_number, 
           AVG(latitude) as latitude, AVG(longitude) as longitude
           FROM argo_profiles 
           GROUP BY platform_number) as float_positions
     ORDER BY distance 
     LIMIT 10;
"""
    
    def translate_query(self, natural_query: str, context: Optional[QueryContext] = None) -> Dict:
        """Translate natural language query to SQL.
        
        Args:
            natural_query: Natural language query from user
            context: Optional context information
            
        Returns:
            Dictionary with SQL query and metadata
        """
        try:
            # Preprocess the query to extract temporal and spatial hints
            processed_query = self._preprocess_query(natural_query)
            
            # Create the prompt
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                HumanMessagePromptTemplate.from_template(
                    "Convert this natural language query to SQL: {query}\n\n"
                    "Additional context: {context}"
                )
            ])
            
            # Prepare context information
            context_info = ""
            if context:
                context_info = f"Available tables: {', '.join(context.available_tables)}"
            
            # Generate SQL
            messages = prompt.format_messages(
                query=processed_query,
                context=context_info
            )
            
            response = self.llm(messages)
            sql_query = response.content.strip()
            
            # Clean up the SQL query
            sql_query = self._clean_sql_query(sql_query)
            
            # Extract query metadata
            metadata = self._extract_query_metadata(natural_query, sql_query)
            
            return {
                'sql': sql_query,
                'original_query': natural_query,
                'processed_query': processed_query,
                'metadata': metadata,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error translating query '{natural_query}': {e}")
            return {
                'sql': None,
                'original_query': natural_query,
                'error': str(e),
                'success': False
            }
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query to normalize temporal and spatial references."""
        query_lower = query.lower()
        
        # Handle relative dates
        current_date = datetime.now()
        
        # Replace "last X months" with specific dates
        if "last" in query_lower and "month" in query_lower:
            months_match = re.search(r"last (\d+) months?", query_lower)
            if months_match:
                months = int(months_match.group(1))
                start_date = current_date - timedelta(days=months * 30)
                query = query.replace(
                    months_match.group(0),
                    f"from {start_date.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}"
                )
        
        # Handle "this year", "last year"
        if "this year" in query_lower:
            year = current_date.year
            query = query.replace("this year", f"in {year}")
        
        if "last year" in query_lower:
            year = current_date.year - 1
            query = query.replace("last year", f"in {year}")
        
        # Normalize geographic references
        for region, bounds in self.geographic_regions.items():
            if region in query_lower:
                query = query.replace(
                    region,
                    f"{region} (lat {bounds['min_lat']}-{bounds['max_lat']}, lon {bounds['min_lon']}-{bounds['max_lon']})"
                )
        
        return query
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """Clean and validate the generated SQL query."""
        # Remove markdown formatting if present
        sql_query = re.sub(r'```sql\n?', '', sql_query)
        sql_query = re.sub(r'```\n?', '', sql_query)
        
        # Remove extra whitespace
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()
        
        # Ensure query ends with semicolon
        if not sql_query.endswith(';'):
            sql_query += ';'
        
        return sql_query
    
    def _extract_query_metadata(self, natural_query: str, sql_query: str) -> Dict:
        """Extract metadata about the query for context."""
        metadata = {
            'query_type': 'unknown',
            'tables_used': [],
            'has_spatial_filter': False,
            'has_temporal_filter': False,
            'parameters_requested': []
        }
        
        sql_lower = sql_query.lower()
        natural_lower = natural_query.lower()
        
        # Determine query type
        if 'count(' in sql_lower or 'sum(' in sql_lower:
            metadata['query_type'] = 'aggregation'
        elif 'order by' in sql_lower and 'distance' in sql_lower:
            metadata['query_type'] = 'spatial_search'
        elif 'select' in sql_lower:
            metadata['query_type'] = 'data_retrieval'
        
        # Extract tables used
        for table in ['argo_profiles', 'argo_floats', 'data_summaries']:
            if table in sql_lower:
                metadata['tables_used'].append(table)
        
        # Check for spatial filters
        spatial_keywords = ['latitude', 'longitude', 'between', 'near']
        metadata['has_spatial_filter'] = any(keyword in sql_lower for keyword in spatial_keywords)
        
        # Check for temporal filters
        temporal_keywords = ['date', 'year', 'month', 'time']
        metadata['has_temporal_filter'] = any(keyword in sql_lower for keyword in temporal_keywords)
        
        # Extract requested parameters
        parameter_keywords = {
            'temperature': 'temperature',
            'salinity': 'salinity', 
            'pressure': 'pressure/depth',
            'profile': 'profiles',
            'float': 'floats'
        }
        
        for keyword, param in parameter_keywords.items():
            if keyword in natural_lower:
                metadata['parameters_requested'].append(param)
        
        return metadata
    
    def suggest_query_improvements(self, query_result: Dict) -> List[str]:
        """Suggest improvements for query optimization."""
        suggestions = []
        
        if not query_result['success']:
            suggestions.append("Query translation failed. Try rephrasing with more specific terms.")
            return suggestions
        
        sql = query_result['sql'].lower()
        metadata = query_result['metadata']
        
        # Check for missing LIMIT clause
        if 'limit' not in sql and metadata['query_type'] == 'data_retrieval':
            suggestions.append("Consider adding a LIMIT clause to prevent large result sets.")
        
        # Check for quality control filters
        if 'argo_profiles' in metadata['tables_used'] and '_qc' not in sql:
            suggestions.append("Consider adding quality control filters (e.g., temperature_qc IN ('1', '2')).")
        
        # Suggest spatial optimization
        if not metadata['has_spatial_filter'] and 'near' in query_result['original_query'].lower():
            suggestions.append("For location-based queries, specify latitude/longitude bounds for better performance.")
        
        # Suggest temporal optimization  
        if not metadata['has_temporal_filter'] and any(word in query_result['original_query'].lower() 
                                                      for word in ['recent', 'latest', 'current']):
            suggestions.append("Consider adding specific date ranges for more precise results.")
        
        return suggestions


class QueryValidator:
    """Validates generated SQL queries for safety and correctness."""
    
    def __init__(self):
        self.allowed_keywords = {
            'select', 'from', 'where', 'join', 'inner', 'left', 'right', 'outer',
            'group', 'by', 'having', 'order', 'limit', 'offset', 'as', 'and', 'or',
            'not', 'in', 'between', 'like', 'ilike', 'is', 'null', 'distinct',
            'count', 'sum', 'avg', 'min', 'max', 'sqrt', 'pow', 'abs'
        }
        
        self.forbidden_keywords = {
            'drop', 'delete', 'insert', 'update', 'create', 'alter', 'truncate',
            'grant', 'revoke', 'exec', 'execute', 'sp_', 'xp_'
        }
    
    def validate_query(self, sql_query: str) -> Tuple[bool, List[str]]:
        """Validate SQL query for safety and basic correctness.
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        sql_lower = sql_query.lower()
        
        # Check for forbidden operations
        for forbidden in self.forbidden_keywords:
            if forbidden in sql_lower:
                issues.append(f"Forbidden operation detected: {forbidden}")
        
        # Check for basic SQL structure
        if not sql_lower.strip().startswith('select'):
            issues.append("Query must start with SELECT")
        
        if 'from' not in sql_lower:
            issues.append("Query must contain FROM clause")
        
        # Check for reasonable LIMIT
        limit_match = re.search(r'limit\s+(\d+)', sql_lower)
        if limit_match:
            limit_value = int(limit_match.group(1))
            if limit_value > 10000:
                issues.append(f"LIMIT value too high: {limit_value} (max recommended: 10000)")
        
        # Check for table names
        valid_tables = ['argo_profiles', 'argo_floats', 'data_summaries']
        has_valid_table = any(table in sql_lower for table in valid_tables)
        if not has_valid_table:
            issues.append("Query must reference valid ARGO tables")
        
        return len(issues) == 0, issues
