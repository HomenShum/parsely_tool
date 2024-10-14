# examples/example_usage.py

import asyncio
from parsely_tool import Parser, Storage, QueryEngine, Utils, Config
from io import BytesIO
import os
import glob
import logging

def main():
    # Setup configuration
    # Setup logging configuration
    

    config = Config(username='username', verbose=True)
    Utils.setup_logging(config.verbose)

    # Initialize components
    parser = Parser(config)
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
    asyncio.run(parser.parse_files(file_objects))
    storage.store_documents(parser.get_parsed_documents())

    # Perform a query
    user_question = "What is the content about?"
    response, source_nodes = query_engine.query(user_question)

    print(f"Answer: {response}")

    print("Source Documents:")
    for node in source_nodes:
        print(f"Document: {node.node.text}")


if __name__ == '__main__':
    main()

