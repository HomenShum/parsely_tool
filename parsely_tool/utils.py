# parsely_tool/utils.py

import logging
from openai import OpenAI, AzureOpenAI
from spider import Spider

class Utils:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.openai_api_key = config.openai_api_key
        self.azure_api_key = config.azure_api_key
        self.azure_endpoint = config.azure_endpoint
        self.azure_api_version = config.azure_api_version

        # Initialize clients based on config
        self.openai_client = None
        self.azure_openai_client = None

        if config.use_azure_openai:
            self.azure_openai_client = AzureOpenAI(
                api_key=self.azure_api_key,
                api_version=self.azure_api_version,
                azure_endpoint=self.azure_endpoint
            )
        else:
            self.openai_client = OpenAI(api_key=self.openai_api_key)

        self.spider_scraper_api_key = config.spider_scraper_api_key

    def setup_logging(self, verbose=False):
        """
        Setup logging configuration.
        """
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def generate_user_needs(self, user_question, model="gpt-4o-mini"):
        """
        Generate user needs from a user question using OpenAI or Azure OpenAI.
        """
        try:
            if self.azure_openai_client:
                response = self.azure_openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Rephrase the User Input. Extract key topics."},
                        {"role": "user", "content": f"User Input: {user_question}"}
                    ],
                    seed=42,
                )
            else:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Rephrase the User Input. Extract key topics."},
                        {"role": "user", "content": f"User Input: {user_question}"}
                    ],
                    seed=42,
                )

            user_needs = response.choices[0].message.content.strip()
            return user_needs
        except Exception as e:
            self.logger.error(f"Error generating user needs: {str(e)}")
            return ""

    def generate_final_response(self, user_question, user_needs, search_results, model="gpt-4o"):
        """
        Generate a final response using OpenAI or Azure OpenAI based on search results.
        """
        try:
            combined_results = {
                "User Input": user_question,
                "User Needs": user_needs,
                "Search Results": search_results,
            }
            assistant_content = f"{combined_results}"

            if self.azure_openai_client:
                response = self.azure_openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Provide a detailed answer based on search results."
                        },
                        {
                            "role": "user",
                            "content": assistant_content
                        }
                    ],
                    seed=42,
                )
            else:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Provide a detailed answer based on search results."
                        },
                        {
                            "role": "user",
                            "content": assistant_content
                        }
                    ],
                    seed=42,
                )

            final_response = response.choices[0].message.content.strip()
            return final_response
        except Exception as e:
            self.logger.error(f"Error generating final response: {str(e)}")
            return ""

    def embed_text(self, text, model="text-embedding-3-small"):
        """
        Generate embeddings for the query text using OpenAI or Azure OpenAI.
        """
        try:
            if self.azure_openai_client:
                response = self.azure_openai_client.embeddings.create(
                    input=text,
                    model=model
                )
            else:
                response = self.openai_client.embeddings.create(
                    input=text,
                    model=model
                )
            # vector = response['data'][0]['embedding']
            vector = response.data[0].embedding
            return vector
        except Exception as e:
            self.logger.error(f"Error generating embeddings for text: {e}")
            return [0.0] * 1536  # Return a zero vector as a fallback

    def spider_cloud_scrape(self, url):
        """
        Scrape the content from the given URL using the Spider API.
        """
        # Initialize the Spider with your API key
        spider_scraper_app = Spider(api_key=self.spider_scraper_api_key)

        # Crawl the entity
        crawler_params = {
            "limit": 1,
            "proxy_enabled": True,
            "store_data": False,
            "metadata": False,
            "request": "http",
            "return_format": "markdown",
        }

        try:
            scraped_data = spider_scraper_app.crawl_url(url, params=crawler_params)
            print(f"Scraped data found for URL: {url}")
            markdown = scraped_data[0]["content"]
        except Exception as e:
            print(f"Error scraping URL {url}: {e}")
            markdown = ""

        print(f"Scraped content: {markdown}")

        return markdown
