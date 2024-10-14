import unittest
from unittest.mock import patch, AsyncMock
from parsely_tool.parsers import Parser
from parsely_tool.config import Config
from io import BytesIO
import os

class TestParser(unittest.IsolatedAsyncioTestCase):
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_openai_api_key',
        'PARSE_API_URL': 'http://localhost:8000/parse',
        'QDRANT_API_KEY': 'test_qdrant_api_key',
        'QDRANT_URL': 'http://localhost:6333',
        'COHERE_API_KEY': 'test_cohere_api_key',
        'AOAIKey': 'test_azure_openai_key',
        'AOAIEndpoint': 'test_azure_openai_endpoint'
    })
    @patch('parsely_tool.parsers.aiohttp.ClientSession.post', new_callable=AsyncMock)
    async def asyncSetUp(self, mock_post):
        """
        Mock the aiohttp.ClientSession.post to avoid actual HTTP calls.
        """
        # Define the mock response for the parsing API
        mock_response = AsyncMock()
        mock_response.text = AsyncMock(return_value='{"extracted_text": {"0": ["Mocked extracted text from PDF."]}}')
        mock_post.return_value.__aenter__.return_value = mock_response

        # Initialize Config and Parser with mocked environment variables
        config = Config(username='test_user', verbose=False)
        self.parser = Parser(config)
        self.mock_post = mock_post  # Store mock for further verification if needed

    async def test_parse_pdf(self):
        """
        Test the PDF parsing function with a mocked HTTP response.
        """
        # Create a mock PDF file
        file_content = b"%PDF-1.4 mock pdf content"
        file_obj = BytesIO(file_content)
        file_obj.name = "test.pdf"

        # Run the parser
        await self.parser.parse_pdf(file_obj)

        # Retrieve parsed documents
        parsed_docs = self.parser.get_parsed_documents()

        # Assert that the parsed document exists with correct content
        self.assertIn("test.pdf_0", parsed_docs)
        self.assertEqual(parsed_docs["test.pdf_0"]["text"], "Mocked extracted text from PDF.")

        # Ensure the mock POST request was called once with correct parameters
        self.mock_post.assert_called_once_with(
            'http://localhost:8000/parse',
            data=unittest.mock.ANY,
            timeout=unittest.mock.ANY
        )

    # Add more async tests for other file types as needed

if __name__ == '__main__':
    unittest.main()
