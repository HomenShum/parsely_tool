# tests/test_query.py

import unittest
from unittest.mock import patch, AsyncMock, MagicMock
from parsely_tool.query import QueryEngine
from parsely_tool.config import Config
from io import BytesIO
import asyncio
import os

class TestQueryEngine(unittest.TestCase):
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_openai_api_key',
        'PARSE_API_URL': 'http://localhost:8000/parse',
        'QDRANT_API_KEY': 'test_qdrant_api_key',
        'QDRANT_URL': 'http://localhost:6333',
        'COHERE_API_KEY': 'test_cohere_api_key',
        'AOAIKey': 'test_azure_openai_key',
        'AOAIEndpoint': 'test_azure_openai_endpoint'
    })
    @patch('parsely_tool.query.openai.Embedding.create')
    @patch('parsely_tool.query.QdrantClient')
    def setUp(self, mock_qdrant_client, mock_openai_embedding):
        """
        Set up the test environment by mocking external dependencies.
        """
        # Mock OpenAI Embedding response
        mock_openai_embedding.return_value = {
            'data': [{'embedding': [0.1] * 1536}]
        }
        
        # Mock QdrantClient instance and its methods
        self.mock_qdrant = mock_qdrant_client.return_value
        self.mock_qdrant.search.return_value = [
            {'id': 'doc1', 'score': 0.95, 'payload': {'filename': 'test.pdf', 'date_uploaded': 1234567890}}
        ]
        
        # Initialize Config and QueryEngine with mocked environment variables
        config = Config(username='test_user', verbose=False)
        self.query_engine = QueryEngine(config)
    
    def test_embed_text_success(self):
        """
        Test that embed_text successfully returns the mocked embedding vector.
        """
        text = "Sample query"
        vector = self.query_engine.embed_text(text)
        expected_vector = [0.1] * 1536
        self.assertEqual(vector, expected_vector, "The embedding vector does not match the expected value.")
    
    def test_query_success(self):
        """
        Test that the query method returns the expected search results.
        """
        query_text = "What is the content about?"
        results = self.query_engine.query(query_text)
        
        # Assert that one result is returned
        self.assertEqual(len(results), 1, "The number of search results returned is not as expected.")
        
        # Assert the content of the first result
        first_result = results[0]
        self.assertEqual(first_result['id'], 'doc1', "The document ID does not match the expected value.")
        self.assertEqual(first_result['score'], 0.95, "The score does not match the expected value.")
        self.assertEqual(first_result['payload']['filename'], 'test.pdf', "The filename does not match the expected value.")
        self.assertEqual(first_result['payload']['date_uploaded'], 1234567890, "The date_uploaded does not match the expected value.")
    
    @patch('parsely_tool.query.openai.Embedding.create', side_effect=Exception("Embedding Error"))
    def test_embed_text_failure(self, mock_openai_embedding):
        """
        Test that embed_text handles exceptions and returns a fallback vector.
        """
        text = "Sample query"
        vector = self.query_engine.embed_text(text)
        expected_vector = [0.0] * 1536
        self.assertEqual(vector, expected_vector, "The fallback embedding vector does not match the expected value.")
    
    @patch('parsely_tool.query.openai.Embedding.create')
    @patch('parsely_tool.query.QdrantClient')
    def test_query_no_results(self, mock_qdrant_client, mock_openai_embedding):
        """
        Test that the query method handles cases with no search results.
        """
        # Setup mocks
        mock_openai_embedding.return_value = {
            'data': [{'embedding': [0.2] * 1536}]
        }
        mock_qdrant_client.return_value.search.return_value = []  # No results
        
        # Initialize QueryEngine with mocked dependencies
        config = Config(username='test_user', verbose=False)
        query_engine = QueryEngine(config)
        
        query_text = "Non-existent content"
        results = query_engine.query(query_text)
        
        # Assert that no results are returned
        self.assertEqual(len(results), 0, "The query should return no results.")
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_openai_api_key',
        'PARSE_API_URL': 'http://localhost:8000/parse',
        'QDRANT_API_KEY': 'test_qdrant_api_key',
        'QDRANT_URL': 'http://localhost:6333',
        'COHERE_API_KEY': 'test_cohere_api_key',
        'AOAIKey': 'test_azure_openai_key',
        'AOAIEndpoint': 'test_azure_openai_endpoint'
    })
    @patch('parsely_tool.query.openai.Embedding.create')
    @patch('parsely_tool.query.QdrantClient')
    def test_query_multiple_results(self, mock_qdrant_client, mock_openai_embedding):
        """
        Test that the query method correctly handles multiple search results.
        """
        # Setup mocks
        mock_openai_embedding.return_value = {
            'data': [{'embedding': [0.3] * 1536}]
        }
        mock_qdrant_client.return_value.search.return_value = [
            {'id': 'doc1', 'score': 0.95, 'payload': {'filename': 'test1.pdf', 'date_uploaded': 1234567890}},
            {'id': 'doc2', 'score': 0.90, 'payload': {'filename': 'test2.pdf', 'date_uploaded': 1234567891}},
            {'id': 'doc3', 'score': 0.85, 'payload': {'filename': 'test3.pdf', 'date_uploaded': 1234567892}},
        ]
        
        # Initialize Config and QueryEngine with mocked environment variables
        config = Config(username='test_user', verbose=False)
        query_engine = QueryEngine(config)
        
        query_text = "Relevant content"
        results = query_engine.query(query_text, top_k=3)
        
        # Assert that three results are returned
        self.assertEqual(len(results), 3, "The number of search results returned is not as expected.")
        
        # Assert the content of each result
        for i, result in enumerate(results, start=1):
            self.assertEqual(result['id'], f'doc{i}', f"The document ID for result {i} does not match.")
            self.assertEqual(result['score'], 0.95 - 0.05 * (i - 1), f"The score for result {i} does not match.")
            self.assertEqual(result['payload']['filename'], f'test{i}.pdf', f"The filename for result {i} does not match.")
            self.assertEqual(result['payload']['date_uploaded'], 1234567890 + i - 1, f"The date_uploaded for result {i} does not match.")
    
    @patch('parsely_tool.query.openai.Embedding.create')
    @patch('parsely_tool.query.QdrantClient')
    def test_embed_text_invalid_response(self, mock_qdrant_client, mock_openai_embedding):
        """
        Test that embed_text handles invalid responses from OpenAI API gracefully.
        """
        # Setup mock to return invalid data
        mock_openai_embedding.return_value = {
            'invalid_key': 'invalid_value'
        }
        
        text = "Sample query"
        vector = self.query_engine.embed_text(text)
        expected_vector = [0.0] * 1536
        self.assertEqual(vector, expected_vector, "The fallback embedding vector does not match the expected value when response is invalid.")
    
    def tearDown(self):
        """
        Clean up after each test method.
        """
        pass  # Add any necessary cleanup here

if __name__ == '__main__':
    unittest.main()
