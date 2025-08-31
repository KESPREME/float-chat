"""Tests for RAG pipeline functionality."""

import pytest
import asyncio
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime

from rag.rag_pipeline import ArgoRAGPipeline
from rag.llm.query_translator import NLToSQLTranslator, QueryValidator
from data.storage.vector_store import ArgoVectorManager


class TestNLToSQLTranslator:
    """Test natural language to SQL translation."""
    
    @pytest.fixture
    def translator(self):
        """Mock translator for testing."""
        with patch('rag.llm.query_translator.ChatGoogleGenerativeAI'):
            return NLToSQLTranslator("fake-api-key")
    
    def test_preprocess_query(self, translator):
        """Test query preprocessing."""
        query = "Show me data from last 6 months in the Arabian Sea"
        processed = translator._preprocess_query(query)
        
        assert "arabian sea" in processed.lower()
        assert "months" in processed.lower()
    
    def test_clean_sql_query(self, translator):
        """Test SQL query cleaning."""
        dirty_sql = "```sql\nSELECT * FROM argo_profiles;\n```"
        clean_sql = translator._clean_sql_query(dirty_sql)
        
        assert clean_sql == "SELECT * FROM argo_profiles;"
        assert "```" not in clean_sql


class TestQueryValidator:
    """Test SQL query validation."""
    
    @pytest.fixture
    def validator(self):
        return QueryValidator()
    
    def test_valid_query(self, validator):
        """Test validation of valid query."""
        query = "SELECT * FROM argo_profiles WHERE latitude > 0 LIMIT 100;"
        is_valid, issues = validator.validate_query(query)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_invalid_query_forbidden_operation(self, validator):
        """Test validation catches forbidden operations."""
        query = "DROP TABLE argo_profiles;"
        is_valid, issues = validator.validate_query(query)
        
        assert not is_valid
        assert any("forbidden" in issue.lower() for issue in issues)
    
    def test_invalid_query_no_from(self, validator):
        """Test validation catches missing FROM clause."""
        query = "SELECT platform_number;"
        is_valid, issues = validator.validate_query(query)
        
        assert not is_valid
        assert any("from" in issue.lower() for issue in issues)


class TestArgoRAGPipeline:
    """Test complete RAG pipeline."""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        mock = Mock()
        mock.get_session.return_value = Mock()
        return mock
    
    @pytest.fixture
    def mock_vector_manager(self):
        """Mock vector manager."""
        mock = Mock()
        mock.search_relevant_floats.return_value = [
            {
                'platform_number': '5901234',
                'relevance_score': 0.95,
                'summary_text': 'Test float in Arabian Sea',
                'total_profiles': 100
            }
        ]
        return mock
    
    @pytest.fixture
    def rag_pipeline(self, mock_db_manager, mock_vector_manager):
        """Mock RAG pipeline."""
        with patch('rag.rag_pipeline.ChatGoogleGenerativeAI'):
            pipeline = ArgoRAGPipeline(
                database_manager=mock_db_manager,
                vector_manager=mock_vector_manager,
                google_api_key="fake-key"
            )
            return pipeline
    
    def test_process_query_success(self, rag_pipeline):
        """Test successful query processing."""
        # Mock the translator
        with patch.object(rag_pipeline.query_translator, 'translate_query') as mock_translate:
            mock_translate.return_value = {
                'success': True,
                'sql': 'SELECT * FROM argo_profiles LIMIT 10;',
                'metadata': {'query_type': 'data_retrieval'}
            }
            
            # Mock SQL execution
            with patch.object(rag_pipeline, '_execute_sql_query') as mock_execute:
                mock_execute.return_value = {
                    'data': [{'platform_number': '5901234', 'temperature': 25.5}],
                    'row_count': 1,
                    'columns': ['platform_number', 'temperature'],
                    'execution_time': 0.1
                }
                
                # Mock response generation
                with patch.object(rag_pipeline.response_llm, 'predict') as mock_llm:
                    mock_llm.return_value = "Found 1 temperature measurement."
                    
                    result = rag_pipeline.process_query("Show me temperature data")
                    
                    assert result['success']
                    assert result['row_count'] == 1
                    assert 'response' in result
    
    def test_process_query_translation_failure(self, rag_pipeline):
        """Test query processing with translation failure."""
        with patch.object(rag_pipeline.query_translator, 'translate_query') as mock_translate:
            mock_translate.return_value = {
                'success': False,
                'error': 'Could not translate query'
            }
            
            result = rag_pipeline.process_query("Invalid query")
            
            assert not result['success']
            assert 'error' in result


if __name__ == "__main__":
    pytest.main([__file__])
