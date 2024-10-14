# examples/example_usage_2.py

import nest_asyncio

nest_asyncio.apply()

import asyncio
from parsely_tool.parsers import Parser
from parsely_tool.storage import Storage
from parsely_tool.query import QueryEngine
from parsely_tool.utils import Utils
from parsely_tool.config import Config  # Import Config from parsely_tool.config
from io import BytesIO
import os
import glob
import logging
import argparse
import time

async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Example usage of parsely_tool.')
    parser.add_argument('--use-azure', action='store_true', help='Use Azure OpenAI instead of OpenAI')
    args = parser.parse_args()

    # Setup configuration
    config = Config()  # Use the imported Config class
    config.verbose = True
    config.use_azure_openai = args.use_azure
    Utils(config).setup_logging(config.verbose)

    # Initialize components
    parser_component = Parser(config)
    storage = Storage(config)
    query_engine = QueryEngine(config)
    logger = logging.getLogger(__name__)

    # Read files
    file_directory = './assets/files'
    file_paths = glob.glob(os.path.join(file_directory, '*'))
    file_objects = []
    for file_path in file_paths:
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
                file_obj = BytesIO(file_content)
                file_obj.name = os.path.basename(file_path)
                file_objects.append(file_obj)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")

    # Parse and store files
    await parser_component.parse_files(file_objects)
    documents = parser_component.get_parsed_documents()
    await storage.astore_documents(documents)

    # Perform a query
    start_time = time.perf_counter()
    print("Performing sync query...")
    user_question = "What is the content about?"
    response, source_nodes = query_engine.query(user_question)
    end_time = time.perf_counter()
    print(f"Sync query time: {end_time - start_time:.2f} seconds")

    print(f"Answer: {response}")

    print("Source Documents:")
    for node in source_nodes:
        print(f"Document: {node.node.text}")

    # Perform an async query
    start_time = time.perf_counter()
    print("Performing async query...")
    response, source_nodes = await query_engine.aquery(user_question, similarity_top_k=2, sparse_top_k=12)
    end_time = time.perf_counter()
    print(f"Async query time: {end_time - start_time:.2f} seconds")

    print(f"Answer: {response}")

    print("Source Documents:")
    for node in source_nodes:
        print(f"Document: {node.node.text}")

    # Generate user needs and final response using Utils
    utils = Utils(config)
    user_needs = utils.generate_user_needs(user_question)
    print(f"User Needs: {user_needs}")

    # Let's assume we have search results from the query
    search_results = [node.node.text for node in source_nodes]

    final_response = utils.generate_final_response(user_question, user_needs, search_results)
    print(f"Final Response: {final_response}")

if __name__ == '__main__':
    asyncio.run(main())
