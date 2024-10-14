# parsely_tool/storage.py

from qdrant_client import QdrantClient, AsyncQdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
import logging
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from .utils import Utils

class Storage:
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

        if not os.path.exists("./bge_m3_onnx"):
            OptimumEmbedding.create_and_save_optimum_model(
                # "BAAI/bge-small-en-v1.5", "./bge_onnx"
                "BAAI/bge-m3", "./bge_m3_onnx"
            )

        self.embed_model = OptimumEmbedding(folder_name="./bge_m3_onnx")
        

    def ensure_collection_exists(self, collection_name):
        """
        Ensure a Qdrant collection exists. Create it if it does not.
        Returns True if collection was created, False if it already exists.
        """
        if not self.qdrant_client.collection_exists(collection_name=collection_name):
            self.qdrant_client.recreate_collection(
                collection_name=collection_name,
                shard_number=4,
                vectors_config={
                    "text-dense": models.VectorParams(
                        size=1024,  # OpenAI Embeddings
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "text-sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                },
            )
            self.logger.info(f"Collection {collection_name} created.")
            return True  # Collection was created
        else:
            self.logger.info(f"Collection {collection_name} already exists.")
            return False  # Collection already exists


    def store_documents(self, documents, overwrite=False):
        """
        Store Document objects into Qdrant using VectorStoreIndex.
        If overwrite is False and the collection already exists, indexing is skipped.
        """
        collection_created = self.ensure_collection_exists(self.collection_name)

        if not collection_created and not overwrite:
            self.logger.info(f"Collection {self.collection_name} already exists. Skipping indexing as overwrite=False.")
            return  # Do not proceed with indexing

        # Proceed with indexing
        vector_store = QdrantVectorStore(
            collection_name=self.collection_name,
            client=self.qdrant_client,
            aclient=self.qdrant_async_client,
            enable_hybrid=True,
            prefer_grpc=True,
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
            use_async=False,
        )

        self.logger.info(f"Stored {len(documents)} documents in Qdrant.")

    def astore_documents(self, documents, overwrite=False):
        """
        Asynchronously store Document objects into Qdrant using VectorStoreIndex.
        If overwrite is False and the collection already exists, indexing is skipped.
        """
        collection_created = self.ensure_collection_exists(self.collection_name)

        if not collection_created and not overwrite:
            self.logger.info(f"Collection {self.collection_name} already exists. Skipping indexing as overwrite=False.")
            return  # Do not proceed with indexing

        # Proceed with indexing
        vector_store = QdrantVectorStore(
            collection_name=self.collection_name,
            client=self.qdrant_client,
            aclient=self.qdrant_async_client,
            enable_hybrid=True,
            prefer_grpc=True,
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
            use_async=True,
        )

        self.logger.info(f"Stored {len(documents)} documents in Qdrant.")


    def get_collection_info(self):
        """
        Retrieve information about the Qdrant collection.
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            self.logger.info(f"Collection Info: {collection_info}")
            return collection_info
        except Exception as e:
            self.logger.error(f"Error retrieving collection info: {e}")
            return None

    def delete_collection(self):
        """
        Delete the Qdrant collection.
        """
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            self.logger.info(f"Deleted collection {self.collection_name}.")
        except Exception as e:
            self.logger.error(f"Error deleting collection {self.collection_name}: {e}")

    def store_metadata(self, metadata):
        """
        Store metadata for the documents in Qdrant.
        """
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.logger.error(f"Collection {self.collection_name} does not exist. Cannot store metadata.")
            return

        # Prepare metadata for insertion
        points = []
        for doc_id, meta in metadata.items():
            points.append({
                'id': doc_id,
                'payload': meta
            })

        # Update metadata in Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        self.logger.info(f"Stored metadata for {len(points)} documents in Qdrant.")

    def list_all_collections(self):
        """
        List all collections in Qdrant.
        """
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            self.logger.info(f"Available collections: {collection_names}")
            return collection_names
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            return []

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
            self.logger.error(f"Error deleting all collections: {str(e)}")