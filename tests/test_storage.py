# tests/test_storage.py

import unittest
from parsely_tool.storage import Storage
from parsely_tool.config import Config
from unittest.mock import patch, MagicMock
import os

class TestStorage(unittest.TestCase):
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_openai_api_key',
        'PARSE_API_URL': 'http://localhost:8000/parse',
        'QDRANT_API_KEY': 'test_qdrant_api_key',
        'QDRANT_URL': 'http://localhost:6333',
        'COHERE_API_KEY': 'test_cohere_api_key',
        'AOAIKey': 'test_azure_openai_key',
        'AOAIEndpoint': 'test_azure_openai_endpoint'
    })
    @patch('parsely_tool.storage.QdrantClient')
    @patch('parsely_tool.storage.openai.Embedding.create')
    def setUp(self, mock_openai_embedding, mock_qdrant_client):
        # Mock OpenAI Embedding response
        mock_openai_embedding.return_value = {
            'data': [{'embedding': [0.1] * 1536}]
        }
        
        # Mock QdrantClient methods
        self.mock_qdrant = mock_qdrant_client.return_value
        self.mock_qdrant.collection_exists.return_value = False
        self.mock_qdrant.create_collection.return_value = None
        self.mock_qdrant.upsert.return_value = None

        config = Config(username='test_user', verbose=False)
        self.storage = Storage(config)

    def test_store_documents(self):
        # Create mock documents
        documents = {
            'doc1': {
                'filename': 'test.pdf',
                'id': 'doc1',
                'text': 'Mocked extracted text from PDF.',
                'date_uploaded': 1234567890,
            }
        }
        self.storage.store_documents(documents)
        
        # Assert that create_collection was called since it didn't exist
        self.mock_qdrant.collection_exists.assert_called_with(self.storage.collection_name)
        self.mock_qdrant.create_collection.assert_called_once()
        
        # Assert that upsert was called with the correct parameters
        expected_points = [{
            'id': 'doc1',
            'vector': [0.1] * 1536,
            'payload': {
                'filename': 'test.pdf',
                'date_uploaded': 1234567890,
            }
        }]
        self.mock_qdrant.upsert.assert_called_with(
            collection_name=self.storage.collection_name,
            points=expected_points,
        )
        
        # Assert that parsed_documents were stored correctly
        parsed_docs = self.storage.get_collection_info()
        # Depending on how get_collection_info is implemented, adjust assertions accordingly

    # Add more tests as needed

if __name__ == '__main__':
    unittest.main()
