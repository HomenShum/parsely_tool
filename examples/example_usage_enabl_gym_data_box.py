# examples/example_usage_enabl_gym_data.py

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
import time
import json
from pydantic import BaseModel
from typing import Optional, List

# from llama_index.vector_stores.pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec


# Using CCG authentication
"""
.env file contains:
BOX_CCG_CLIENT_ID
BOX_CCG_CLIENT_SECRET
BOX_CCG_ENTERPRISE_ID
BOX_CCG_USER_ID
"""

from llama_index.readers.box import BoxReader
from box_sdk_gen import CCGConfig, BoxCCGAuth, BoxClient

import os
from typing import List
import dotenv

from box_sdk_gen import CCGConfig, BoxCCGAuth, BoxClient, File
from llama_index.readers.box import (
    BoxReader,
    BoxReaderTextExtraction,
    BoxReaderAIPrompt,
    BoxReaderAIExtract,
)
from llama_index.core.schema import Document


def get_box_client() -> BoxClient:
    dotenv.load_dotenv()

    # Common configurations for Box API authentication
    client_id = os.getenv("BOX_CCG_CLIENT_ID", "YOUR_BOX_CLIENT_ID")
    client_secret = os.getenv("BOX_CCG_CLIENT_SECRET", "YOUR_BOX_CLIENT_SECRET")

    # CCG configurations
    enterprise_id = os.getenv("BOX_CCG_ENTERPRISE_ID", "YOUR_BOX_ENTERPRISE_ID")
    ccg_user_id = os.getenv("BOX_CCG_USER_ID")

    # Initialize the Box CCG configuration
    config = CCGConfig(
        client_id=client_id,
        client_secret=client_secret,
        enterprise_id=enterprise_id,
        user_id=ccg_user_id,
    )

    # Initialize Box Authentication with the config
    auth = BoxCCGAuth(config)
    if config.user_id:
        auth.with_user_subject(config.user_id)

    return BoxClient(auth)


def get_testing_data() -> dict:
    """
    This function returns a dictionary of Box file and folder IDs for testing purposes.
    """
    return {
        "disable_folder_tests": True,
        "test_folder_id": "273980493541",
        "test_doc_id": "1584054722303",
        "test_ppt_id": "1584056661506",
        "test_xls_id": "1584048916472",
        "test_pdf_id": "1584049890463",
        "test_json_id": "1584058432468",
        "test_csv_id": "1584054196674",
        "test_txt_waiver_id": "1514587167701",
        "test_folder_invoice_po_id": "261452450320",
        "test_folder_purchase_order_id": "261457585224",
        "test_txt_invoice_id": "1517629086517",
        "test_txt_po_id": "1517628697289",
    }


def print_docs(label: str, docs: List[Document]):
    """
    Prints the information for a list of Document objects.

    :param label: A label for the printed section.
    :param docs: A list of Document objects to be printed.
    """
    print("------------------------------")
    print(f"{label}: {len(docs)} document(s)")

    for doc in docs:
        print("------------------------------")
        file = File.from_dict(doc.extra_info)
        print(f"File ID: {file.id}\nName: {file.name}\nSize: {file.size} bytes")
        print(f"Text: {doc.text[:100]} ...")  # Print the first 100 characters of the text
    print("------------------------------\n\n\n")


def main():
    """
    Main function to authenticate the Box client, load testing data, and retrieve documents.
    """
    box_client = get_box_client()  # Get the Box client using authentication
    test_data = get_testing_data()  # Retrieve test data (file and folder IDs)

    # Initialize the Box reader and load document data
    reader = BoxReader(box_client=box_client)
    docs = reader.load_data(file_ids=[test_data["test_txt_invoice_id"]])

    # Print the retrieved document data
    print_docs("Box Simple Reader", docs)


if __name__ == "__main__":
    main()

# class ExerciseStructuredOutput(BaseModel):
#     workout_name: str
#     reps: int
#     sets: int
#     rest: str
#     completion: bool 
#     image_url: Optional[str]
#     rationale: str  # How does the exercise help or potentially harm the user?  

# class ExercisesResponse(BaseModel):
#     exercises: List[ExerciseStructuredOutput]


# async def main():
#     # Setup configuration
#     config = Config()  # Use the imported Config class
#     config.verbose = True
#     config.use_azure_openai = True
#     Utils(config).setup_logging(config.verbose)

#     # Initialize components
#     parser_component = Parser(config)
#     # During ingestion for gym data
#     collection_name = "ENABL_gym_exercise_dataset_collection"
#     storage = Storage(config, collection_name=collection_name)
#     query_engine = QueryEngine(config, collection_name=collection_name)
#     logger = logging.getLogger(__name__)


    
#     # # Read files parsely_tool\assets\files\Gym Exercises Dataset.xlsx
#     # file_paths = [r'C:\Users\hshum\OneDrive\Desktop\Python\CafeCorner\parsely_tool\assets\files\Gym Exercises Dataset.xlsx']

#     # # file_directory = './assets/files'
#     # # file_paths = glob.glob(os.path.join(file_directory, '*'))
#     # file_objects = []
#     # for file_path in file_paths:
#     #     try:
#     #         with open(file_path, 'rb') as f:
#     #             file_content = f.read()
#     #             file_obj = BytesIO(file_content)
#     #             file_obj.name = os.path.basename(file_path)
#     #             file_objects.append(file_obj)
#     #     except Exception as e:
#     #         logger.error(f"Error reading file {file_path}: {e}")

#     # # Parse and store files
#     # print("Parsing and storing files...")
#     # await parser_component.parse_files(file_objects)
#     # documents = parser_component.get_parsed_documents()
#     # storage.astore_documents(documents, overwrite=True)

#     user_question = '''
#         You are a world class personal trainer. 
#         Given the following information about a user:
#         {
#            "body_goals": "Build Muscle",
#            "allergies": ["Gluten"],
#            "workout_frequency": "5+ Days",
#            "experience": "Advanced",
#            "muscle_group_target": "Chest and Tricep"
#         }

#         What exercises should the user do?
#         workout_name: str
#         reps: int
#         sets: int
#         rest: str
#         completion: bool 
#         image_url: Optional[str]
#         rationale: str  # How does the exercise help or potentially harm the user?  
#         '''

#     # # Perform a synchronous query
#     # start_time = time.perf_counter()
#     # print("Performing sync query...")
#     # response, source_nodes = query_engine.query(user_question)
#     # end_time = time.perf_counter()
#     # print(f"Sync query time: {end_time - start_time:.2f} seconds")

#     # print(f"Answer: {response}")

#     # print("Source Documents:")
#     # for node in source_nodes:
#     #     print(f"Document: {node.node.text}")

#     # Perform an asynchronous query
#     start_time = time.perf_counter()
#     print("Performing async query...")
#     response, source_nodes = await query_engine.aquery(user_question, similarity_top_k=20, sparse_top_k=50)
#     end_time = time.perf_counter()
#     print(f"Async query time: {end_time - start_time:.2f} seconds")

#     print(f"Answer: {response}")
#     # Extract search results from source nodes
#     search_results = [node.node.text for node in source_nodes]

#     # Limit to the first 10 search results
#     search_results = search_results[:10]

#     # Generate user needs
#     utils = Utils(config)
#     user_needs = utils.generate_user_needs(user_question)
#     print(f"User Needs: {user_needs}")

#     # Initialize a list to collect exercises
#     exercises = []

#     # For each search result, generate structured output
#     for search_result in search_results:
#         # Parse the search result to extract the Description_URL
#         try:
#             result_data = json.loads(search_result)
#             description_url = result_data.get('Description_URL')
#             if description_url:
#                 # Scrape the description from the URL using spider_cloud_scrape
#                 description = utils.spider_cloud_scrape(description_url)
#             else:
#                 description = ""
#         except json.JSONDecodeError:
#             description = ""

#         # add exercise description to search result
#         search_result += f"\nDescription: {description}"

#         # Generate structured output with the scraped description
#         exercise = query_engine.structured_output(user_question, user_needs, search_result, ExerciseStructuredOutput)
#         if exercise:
#             exercises.append(exercise)
#         else:
#             print("Failed to generate structured exercise data for a search result.")


#     final_response = utils.generate_final_response(user_question, user_needs, search_results)
#     print(f"Final Response: {final_response}")

#     # end of script
#     print("End of script.")

#     # Inspect the Qdrant collection
#     print("Inspecting the Qdrant collection...")
#     collection_info = query_engine.inspect_collection()
#     print(f"Collection Name: {collection_name}")
#     print(f"Collection Info: {collection_info}")
#     print("End of script.")

#     # Save all of the results to local file
#     print("Saving all of the results to local file...")
#     with open('example_usage_enabl_gym_data_results.txt', 'w') as f:
#         f.write(f"User Question: {user_question}\n")
#         f.write(f"Initial Response: {response}\n")
#         f.write(f"Search Results: {[node.node.text for node in source_nodes]}\n")
#         f.write("Structured Exercise Data: \n")
#         for exercise in exercises:
#             exercise.completion = False
#             f.write(str(exercise) + "\n")
#         f.write(f"Final Response: {final_response}\n")
#         f.write(f"Collection Name: {collection_name}\n")
#         f.write(f"Collection Info: {collection_info}\n")

# if __name__ == '__main__':
#     asyncio.run(main())

