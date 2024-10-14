# parsely_tool/config.py

import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv(dotenv_path='.env')

class Config:
    def __init__(self):
        # Username configuration
        self.username = os.getenv("USERNAME", "user123")

        # OpenAI configurations
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        # Azure OpenAI configurations
        self.use_azure_openai = os.getenv("USE_AZURE_OPENAI", "False").lower() == "true"
        self.azure_api_key = os.getenv("AOAIKey")
        self.azure_endpoint = os.getenv("AOAIEndpoint")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.azure_embedding_model = os.getenv("AOAIEMB3sm", "text-embedding-3-small")
        self.azure_embedding_deployment = os.getenv("AOAIEMB3sm", "text-embedding-3-small")
        self.azure_completion_model = os.getenv("AOAIGPT4oStructuredOutput", "gpt-4o-2024-08-06")  # Default model


        if self.use_azure_openai:
            if not self.azure_api_key or not self.azure_endpoint:
                raise ValueError("Azure OpenAI configuration missing. Ensure AOAIKey and AOAIEndpoint are set.")

        # Qdrant configurations
        self.qdrant_url = os.getenv("qdrant_url")
        self.qdrant_api_key = os.getenv("qdrant_api_key")
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("QDRANT configuration missing. Ensure qdrant_url and qdrant_api_key are set.")

        # Cohere API configuration
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        if not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY is not set.")

        # Optional API for parsing
        self.parse_api_url = os.getenv("parse_api_url")
        if not self.parse_api_url:
            raise ValueError("PARSE_API_URL is not set.")

        # Spider Scraper API configuration
        self.spider_scraper_api_key = os.getenv("spider_scraper_api_key")
        if not self.spider_scraper_api_key:
            raise ValueError("Spider Scraper API key is not set.")
