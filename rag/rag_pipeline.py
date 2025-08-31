"""RAG (Retrieval-Augmented Generation) pipeline for ARGO data queries."""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.llm.query_translator import NLToSQLTranslator, QueryValidator
from data.storage.vector_store import ArgoVectorManager
from data.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class ArgoRAGPipeline:
    """Complete RAG pipeline for ARGO oceanographic data queries."""
    
    def __init__(self, 
                 database_manager: DatabaseManager,
                 vector_manager: ArgoVectorManager,
                 google_api_key: str,
                 llm_model: str = "gemini-1.5-pro"):
        
        self.db_manager = database_manager
        self.vector_manager = vector_manager
        self.query_translator = NLToSQLTranslator(google_api_key, llm_model)
        self.query_validator = QueryValidator()
        
        # Response generation LLM
        from langchain_google_genai import ChatGoogleGenerativeAI
        self.response_llm = ChatGoogleGenerativeAI(
            google_api_key=google_api_key,
            model=llm_model,
            temperature=0.3
        )
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process a complete user query through the RAG pipeline.
        
        Args:
            user_query: Natural language query from user
            
        Returns:
            Dictionary containing query results and response
        """
        try:
            # Step 1: Retrieve relevant context from vector store
            logger.info(f"Processing query: {user_query}")
            relevant_context = self.vector_manager.search_relevant_floats(user_query, k=5)
            
            # Step 2: Translate natural language to SQL
            translation_result = self.query_translator.translate_query(user_query)
            
            if not translation_result['success']:
                return {
                    'success': False,
                    'error': translation_result['error'],
                    'user_query': user_query,
                    'suggestions': ['Try rephrasing your query with more specific terms']
                }
            
            sql_query = translation_result['sql']
            
            # Step 3: Validate SQL query
            is_valid, validation_issues = self.query_validator.validate_query(sql_query)
            
            if not is_valid:
                return {
                    'success': False,
                    'error': f"Query validation failed: {'; '.join(validation_issues)}",
                    'user_query': user_query,
                    'sql_query': sql_query,
                    'suggestions': validation_issues
                }
            
            # Step 4: Execute SQL query
            query_results = self._execute_sql_query(sql_query)
            
            # Step 5: Generate natural language response
            response = self._generate_response(
                user_query=user_query,
                sql_query=sql_query,
                query_results=query_results,
                relevant_context=relevant_context,
                translation_metadata=translation_result['metadata']
            )
            
            # Step 6: Prepare visualization data if applicable
            viz_data = self._prepare_visualization_data(query_results, translation_result['metadata'])
            
            return {
                'success': True,
                'user_query': user_query,
                'sql_query': sql_query,
                'query_results': query_results,
                'response': response,
                'visualization_data': viz_data,
                'relevant_context': relevant_context,
                'metadata': translation_result['metadata'],
                'suggestions': self.query_translator.suggest_query_improvements(translation_result)
            }
            
        except Exception as e:
            logger.error(f"Error processing query '{user_query}': {e}")
            return {
                'success': False,
                'error': str(e),
                'user_query': user_query,
                'suggestions': ['An error occurred. Please try a simpler query.']
            }
    
    def _execute_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query against the database.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Dictionary with query results and metadata
        """
        session = self.db_manager.get_session()
        try:
            start_time = datetime.now()
            
            # Execute query
            result = session.execute(sql_query)
            rows = result.fetchall()
            columns = result.keys()
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Convert to list of dictionaries
            data = []
            for row in rows:
                data.append(dict(zip(columns, row)))
            
            return {
                'data': data,
                'row_count': len(data),
                'columns': list(columns),
                'execution_time': execution_time,
                'executed_at': end_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return {
                'data': [],
                'row_count': 0,
                'columns': [],
                'error': str(e),
                'execution_time': 0
            }
        finally:
            session.close()
    
    def _generate_response(self, 
                          user_query: str,
                          sql_query: str,
                          query_results: Dict,
                          relevant_context: List[Dict],
                          translation_metadata: Dict) -> str:
        """Generate natural language response using LLM.
        
        Args:
            user_query: Original user query
            sql_query: Generated SQL query
            query_results: Results from SQL execution
            relevant_context: Relevant floats from vector search
            translation_metadata: Metadata from query translation
            
        Returns:
            Natural language response
        """
        try:
            # Prepare context for response generation
            context_summary = self._summarize_context(relevant_context)
            results_summary = self._summarize_results(query_results)
            
            # Create prompt for response generation
            prompt = f"""
            You are an expert oceanographer assistant helping users understand ARGO float data.
            
            User Query: {user_query}
            
            Query Results Summary:
            {results_summary}
            
            Relevant Context:
            {context_summary}
            
            Query Metadata:
            - Query Type: {translation_metadata.get('query_type', 'unknown')}
            - Parameters: {', '.join(translation_metadata.get('parameters_requested', []))}
            - Spatial Filter: {translation_metadata.get('has_spatial_filter', False)}
            - Temporal Filter: {translation_metadata.get('has_temporal_filter', False)}
            
            Please provide a clear, informative response that:
            1. Directly answers the user's question
            2. Summarizes key findings from the data
            3. Provides relevant oceanographic context
            4. Mentions any limitations or caveats
            5. Suggests follow-up questions if appropriate
            
            Keep the response conversational and accessible to both experts and non-experts.
            """
            
            response = self.response_llm.predict(prompt)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I found {query_results.get('row_count', 0)} results for your query about ARGO float data. The data shows measurements from various oceanographic profiles."
    
    def _summarize_context(self, relevant_context: List[Dict]) -> str:
        """Summarize relevant context from vector search."""
        if not relevant_context:
            return "No specific float context found."
        
        summary_parts = []
        for ctx in relevant_context[:3]:  # Top 3 most relevant
            platform = ctx['platform_number']
            profiles = ctx.get('total_profiles', 'unknown')
            summary_parts.append(f"Float {platform} ({profiles} profiles)")
        
        return f"Relevant floats: {', '.join(summary_parts)}"
    
    def _summarize_results(self, query_results: Dict) -> str:
        """Summarize query results for LLM context."""
        if query_results.get('error'):
            return f"Query execution failed: {query_results['error']}"
        
        row_count = query_results.get('row_count', 0)
        columns = query_results.get('columns', [])
        
        if row_count == 0:
            return "No data found matching the query criteria."
        
        # Sample some data points for context
        data = query_results.get('data', [])
        sample_size = min(3, len(data))
        
        summary = f"Found {row_count} records with columns: {', '.join(columns)}."
        
        if sample_size > 0:
            summary += f" Sample data includes measurements from {sample_size} records."
            
            # Add specific insights based on data
            if data and isinstance(data[0], dict):
                first_record = data[0]
                
                if 'latitude' in first_record and 'longitude' in first_record:
                    summary += f" Geographic range includes locations like {first_record['latitude']:.2f}°N, {first_record['longitude']:.2f}°E."
                
                if 'date' in first_record:
                    summary += f" Time range includes dates like {first_record['date']}."
                
                if 'temperature' in first_record and first_record['temperature']:
                    summary += f" Temperature measurements include values like {first_record['temperature']:.2f}°C."
        
        return summary
    
    def _prepare_visualization_data(self, query_results: Dict, metadata: Dict) -> Dict[str, Any]:
        """Prepare data for visualization components.
        
        Args:
            query_results: Results from SQL query
            metadata: Query metadata
            
        Returns:
            Dictionary with visualization-ready data
        """
        viz_data = {
            'type': 'none',
            'data': None,
            'config': {}
        }
        
        if query_results.get('row_count', 0) == 0:
            return viz_data
        
        data = query_results['data']
        columns = query_results['columns']
        
        # Determine visualization type based on data and query
        if 'latitude' in columns and 'longitude' in columns:
            viz_data['type'] = 'map'
            viz_data['data'] = self._prepare_map_data(data)
            viz_data['config'] = {
                'center_lat': sum(d['latitude'] for d in data if d.get('latitude')) / len([d for d in data if d.get('latitude')]),
                'center_lon': sum(d['longitude'] for d in data if d.get('longitude')) / len([d for d in data if d.get('longitude')]),
                'zoom': 5
            }
        
        elif 'pressure' in columns and ('temperature' in columns or 'salinity' in columns):
            viz_data['type'] = 'profile'
            viz_data['data'] = self._prepare_profile_data(data)
            viz_data['config'] = {
                'x_axis': 'temperature' if 'temperature' in columns else 'salinity',
                'y_axis': 'pressure',
                'invert_y': True  # Depth increases downward
            }
        
        elif 'date' in columns and len(set(d.get('platform_number') for d in data)) > 1:
            viz_data['type'] = 'time_series'
            viz_data['data'] = self._prepare_time_series_data(data)
            viz_data['config'] = {
                'x_axis': 'date',
                'group_by': 'platform_number'
            }
        
        else:
            viz_data['type'] = 'table'
            viz_data['data'] = data[:100]  # Limit table size
            viz_data['config'] = {
                'columns': columns,
                'max_rows': 100
            }
        
        return viz_data
    
    def _prepare_map_data(self, data: List[Dict]) -> List[Dict]:
        """Prepare data for map visualization."""
        map_data = []
        for record in data:
            if record.get('latitude') and record.get('longitude'):
                point = {
                    'lat': record['latitude'],
                    'lon': record['longitude'],
                    'popup_text': f"Platform: {record.get('platform_number', 'Unknown')}"
                }
                
                if record.get('date'):
                    point['popup_text'] += f"<br>Date: {record['date']}"
                if record.get('temperature'):
                    point['popup_text'] += f"<br>Temperature: {record['temperature']:.2f}°C"
                if record.get('salinity'):
                    point['popup_text'] += f"<br>Salinity: {record['salinity']:.2f} PSU"
                
                map_data.append(point)
        
        return map_data
    
    def _prepare_profile_data(self, data: List[Dict]) -> Dict[str, List]:
        """Prepare data for profile plots."""
        profiles = {}
        
        for record in data:
            platform = record.get('platform_number', 'Unknown')
            if platform not in profiles:
                profiles[platform] = {
                    'pressure': [],
                    'temperature': [],
                    'salinity': []
                }
            
            if record.get('pressure') is not None:
                profiles[platform]['pressure'].append(record['pressure'])
                profiles[platform]['temperature'].append(record.get('temperature'))
                profiles[platform]['salinity'].append(record.get('salinity'))
        
        return profiles
    
    def _prepare_time_series_data(self, data: List[Dict]) -> Dict[str, List]:
        """Prepare data for time series plots."""
        time_series = {}
        
        for record in data:
            platform = record.get('platform_number', 'Unknown')
            if platform not in time_series:
                time_series[platform] = {
                    'dates': [],
                    'values': []
                }
            
            if record.get('date'):
                time_series[platform]['dates'].append(record['date'])
                # Use temperature as default value, fallback to salinity
                value = record.get('temperature') or record.get('salinity')
                time_series[platform]['values'].append(value)
        
        return time_series


class QueryCache:
    """Simple in-memory cache for query results."""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, query_hash: str) -> Optional[Dict]:
        """Get cached result for query."""
        if query_hash in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(query_hash)
            self.access_order.append(query_hash)
            return self.cache[query_hash]
        return None
    
    def put(self, query_hash: str, result: Dict):
        """Cache query result."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[query_hash] = result
        self.access_order.append(query_hash)
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()
        self.access_order.clear()


def create_rag_pipeline(database_url: str, 
                       vector_store_config: Dict,
                       google_api_key: str) -> ArgoRAGPipeline:
    """Factory function to create a complete RAG pipeline.
    
    Args:
        database_url: PostgreSQL database connection URL
        vector_store_config: Configuration for vector store
        google_api_key: Google API key for Gemini
        
    Returns:
        Configured ArgoRAGPipeline instance
    """
    from data.storage.database import DatabaseManager
    from data.storage.vector_store import create_vector_store, ArgoVectorManager
    
    # Initialize components
    db_manager = DatabaseManager(database_url)
    vector_store = create_vector_store(**vector_store_config)
    vector_manager = ArgoVectorManager(vector_store)
    
    # Create pipeline
    pipeline = ArgoRAGPipeline(
        database_manager=db_manager,
        vector_manager=vector_manager,
        google_api_key=google_api_key
    )
    
    return pipeline
