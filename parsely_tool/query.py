# parsely_tool/query.py

from qdrant_client.http import models
from qdrant_client import QdrantClient, AsyncQdrantClient
import logging
import cohere
import Stemmer
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from .utils import Utils
from typing import Any, Type
from pydantic import BaseModel
from openai import OpenAI as OpenAI_original
# from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding


class QueryEngine:
    def __init__(self, config, collection_name=None):
        self.config = config
        self.collection_name = collection_name or f"{config.username}_collection"
        self.qdrant_client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )
        self.qdrant_async_client = AsyncQdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )
        self.logger = logging.getLogger(__name__)
        self.utils = Utils(config)

        # Initialize OpenAI or Azure OpenAI embeddings based on config
        # if config.use_azure_openai:
        #     self.embed_model = AzureOpenAIEmbedding(
        #         model=config.azure_embedding_model,
        #         deployment_name=config.azure_embedding_deployment,
        #         api_key=config.azure_api_key,
        #         azure_endpoint=config.azure_endpoint,
        #         api_version=config.azure_api_version,
        #     )
        # else:
        #     self.embed_model = OpenAIEmbedding(api_key=config.openai_api_key)
        # if not os.path.exists("./bge_m3_onnx"):
        #     OptimumEmbedding.create_and_save_optimum_model(
        #         # "BAAI/bge-small-en-v1.5", "./bge_onnx"
        #         "BAAI/bge-m3", "./bge_m3_onnx"
        #     )

        # self.embed_model = OptimumEmbedding(folder_name="./bge_m3_onnx")
        self.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-large-en-v1.5")


        # if config.use_azure_openai:
        #     self.llm_client = AzureOpenAI(
        #         engine="gpt-4o-2024-08-06",
        #         model="gpt-4o-2024-08-06",
        #         api_key=config.azure_api_key,
        #         api_version=config.azure_api_version,
        #         azure_endpoint=config.azure_endpoint
        #     )
        # else:
        self.structured_output_llm_client = OpenAI_original(api_key=config.openai_api_key)


    def load_index(self):
        """
        Load the VectorStoreIndex from Qdrant.
        """
        vector_store = QdrantVectorStore(
            collection_name=self.collection_name,
            client=self.qdrant_client,
            aclient=self.qdrant_async_client,
            enable_hybrid=True,
            prefer_grpc=True,            
        )

        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model,
            use_async=True,            
        )

    def query(self, user_question, similarity_top_k=2, sparse_top_k=12):
        """
        Perform a hybrid query using the loaded index.
        """
        if not hasattr(self, 'index'):
            self.load_index()

        # Initialize the query engine
        query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            sparse_top_k=sparse_top_k,
            vector_store_query_mode="hybrid",
        )

        # Perform the query
        response = query_engine.query(user_question)
        self.logger.info(f"Query response: {response}")

        # Access the top similar documents
        source_nodes = response.source_nodes

        return response, source_nodes

    async def aquery(self, user_question, similarity_top_k=2, sparse_top_k=12):
        """
        Perform a hybrid query using the loaded index.
        """
        if not hasattr(self, 'index'):
            self.load_index()

        # Initialize the query engine
        query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            sparse_top_k=sparse_top_k,
            vector_store_query_mode="hybrid",
            use_async=True,            
        )

        # Perform the query
        response = await query_engine.aquery(user_question)
        self.logger.info(f"Query response: {response}")

        # Access the top similar documents
        source_nodes = response.source_nodes

        return response, source_nodes



    def structured_output(self, user_question: str, user_needs: str, search_result: str, response_model: Type[BaseModel]):
        """
        Generate structured output based on a single search result using OpenAI's structured data extraction.

        Args:
            user_question (str): The user's original question.
            user_needs (str): The user's needs extracted from the question.
            search_result (str): The text of a single search result.
            response_model (Type[BaseModel]): The Pydantic model defining the expected output schema.

        Returns:
            An instance of the response_model or None if an error occurs.
        """
        # Construct the prompt
        schema_json = response_model.schema_json(indent=2)
        prompt = f"""
                You are an expert at structured data extraction. Given the user's question and needs, and the search result, extract the exercise information and provide a rationale, formatted as per the provided JSON schema.

                User Question:
                {user_question}

                User Needs:
                {user_needs}

                Search Result:
                {search_result}

                Extract the exercise and output it as a JSON object in the following format:
                {schema_json}

                Ensure that the output strictly adheres to the schema.
                """

        try:
            completion = self.structured_output_llm_client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are an expert at structured data extraction."},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_model,
            )
            structured_response = completion.choices[0].message.parsed
            return structured_response
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            return None


    def embed_text(self, text):
        """
        Generate embeddings for the query text using OpenAI.
        """
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-3-small"
            )
            vector = response['data'][0]['embedding']
            return vector
        except Exception as e:
            self.logger.error(f"Error generating embeddings for query: {e}")
            return [0.0] * 1536  # Return a zero vector as a fallback

    def build_metadata_filters(self, filters):
        """
        Build metadata filters to apply to the query.
        """
        if not filters:
            return None

        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        conditions = []
        for key, value in filters.items():
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions)

    def sd_query(self, query, documents=None):
        """
        Performs a sparse-dense query using LlamaIndex components.

        Args:
            query (str): The user's query.
            documents (Optional[List[Document]]): Optional list of documents to query.
                If not provided, uses documents stored in Qdrant.

        Returns:
            list: A list of dictionaries containing the query results.
        """
        try:
            # If documents are provided, use them; else use Qdrant collection
            if documents is not None:
                llama_index_documents = documents
            else:
                if not self.collection_name:
                    logging.error("No collection available for querying.")
                    return []
                # Retrieve all documents from Qdrant
                retrieved_documents = self.qdrant_client.scroll(self.collection_name)
                llama_index_documents = [
                    Document(text=doc.payload["text"], metadata=doc.payload)
                    for doc in retrieved_documents
                ]

            # Create document store and add documents
            docstore = SimpleDocumentStore()
            docstore.add_documents(llama_index_documents)

            # Create BM25 retriever
            similarity_top_k_value = min(10, len(llama_index_documents))
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=docstore,
                similarity_top_k=similarity_top_k_value,
                stemmer=Stemmer.Stemmer("english"),
                language="english",
            )

            # Perform sparse retrieval (BM25)
            sparse_results = bm25_retriever.retrieve(query)

            # Prepare documents for Cohere reranking
            cohere_api_key = self.config.cohere_api_key
            if not cohere_api_key:
                logging.error("COHERE_API_KEY environment variable not set.")
                return []
            cohere_client = cohere.Client(cohere_api_key)
            docs_for_rerank = [node.get_content() for node in sparse_results]

            # Perform dense reranking
            rerank_response = cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=docs_for_rerank,
                top_n=min(10, len(docs_for_rerank)),
                return_documents=True
            )

            # Convert results to output format
            output_data = [
                {
                    "Result": result,
                }
                for result in rerank_response.results
            ]

            return output_data
        except Exception as e:
            logging.error(f"Error in sd_query: {str(e)}")
            return []

    def dense_rerank(self, query, documents):
        """
        Performs dense reranking of documents using Cohere's rerank model.

        Args:
            query (str): The query string.
            documents (List[dict]): List of documents to rerank.

        Returns:
            List[dict]: Reranked list of documents.
        """
        try:
            cohere_api_key = self.config.cohere_api_key
            if not cohere_api_key:
                logging.error("COHERE_API_KEY environment variable not set.")
                return documents

            cohere_client = cohere.Client(cohere_api_key)
            doc_texts = [doc["text"] for doc in documents]
            rerank_response = cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=doc_texts,
                top_n=min(10, len(documents)),
                return_documents=True
            )
            reranked_docs = rerank_response.results
            return reranked_docs
        except Exception as e:
            logging.error(f"No results from rerank: {str(e)}")
            return documents  # Return original documents if reranking fails

    def generate_response(self, user_question, user_needs, sparse_results, dense_results, hybrid_response):
        """
        Generates a detailed answer based on the search results.

        Args:
            user_question (str): The original user question.
            user_needs (str): Rephrased question with extracted key topics.
            sparse_results (list): Results from the sparse search.
            dense_results (list): Results after dense reranking.
            hybrid_response (str): Response from the hybrid search.

        Returns:
            str: The final generated response.
        """
        try:
            combined_results = {
                "sparse_results": sparse_results,
                "dense_results": dense_results,
                "hybrid_response": str(hybrid_response),
            }
            # Prepare the content for the assistant
            assistant_content = f"""
            User Input: {user_question}
            User Needs: {user_needs}
            Combined Search Results: {json.dumps(combined_results, indent=2)}
            """
            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Provide a detailed answer based on the search results. Begin with a question summary, key topics, and provide the answer with source evidence and confidence level."
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
            logging.error(f"Error generating response: {str(e)}")
            return ""  # Return empty string if response generation fails

    def inspect_collection(self):
        """
        Inspect the current Qdrant collection.
        """
        try:
            collection_info = self.qdrant_client.get_collection(collection_name=self.collection_name)
            print(f"Collection Name: {self.collection_name}")
            print(f"Collection Info: {collection_info}")
            return collection_info
        except Exception as e:
            logging.error(f"Error inspecting collection: {str(e)}")
            return None

    def delete_collection(self, collection_name=None):
        """
        Delete a specific Qdrant collection.

        Args:
            collection_name (str, optional): Name of the collection to delete. If None, deletes the current collection.
        """
        try:
            if collection_name is None:
                collection_name = self.collection_name
            self.qdrant_client.delete_collection(collection_name)
            self.logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logging.error(f"Error deleting collection {collection_name}: {str(e)}")

    def delete_all_collections(self):
        """
        Delete all Qdrant collections.
        """
        try:
            collections = self.qdrant_client.get_collections().collections
            for collection in collections:
                self.qdrant_client.delete_collection(collection.name)
                self.logger.info(f"Deleted collection: {collection.name}")
        except Exception as e:
            logging.error(f"Error deleting all collections: {str(e)}")