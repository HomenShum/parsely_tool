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


class ExerciseStructuredOutput(BaseModel):
    workout_name: str
    reps: int
    sets: int
    rest: str
    completion: bool 
    image_url: Optional[str]
    rationale: str  # How does the exercise help or potentially harm the user?  

class ExercisesResponse(BaseModel):
    exercises: List[ExerciseStructuredOutput]


async def main():
    # Setup configuration
    config = Config()  # Use the imported Config class
    config.verbose = True
    config.use_azure_openai = True
    Utils(config).setup_logging(config.verbose)

    # Initialize components
    parser_component = Parser(config)
    # During ingestion for gym data
    collection_name = "ENABL_gym_exercise_dataset_collection"
    storage = Storage(config, collection_name=collection_name)
    query_engine = QueryEngine(config, collection_name=collection_name)
    logger = logging.getLogger(__name__)


    
    # # Read files parsely_tool\assets\files\Gym Exercises Dataset.xlsx
    # file_paths = [r'C:\Users\hshum\OneDrive\Desktop\Python\CafeCorner\parsely_tool\assets\files\Gym Exercises Dataset.xlsx']

    # # file_directory = './assets/files'
    # # file_paths = glob.glob(os.path.join(file_directory, '*'))
    # file_objects = []
    # for file_path in file_paths:
    #     try:
    #         with open(file_path, 'rb') as f:
    #             file_content = f.read()
    #             file_obj = BytesIO(file_content)
    #             file_obj.name = os.path.basename(file_path)
    #             file_objects.append(file_obj)
    #     except Exception as e:
    #         logger.error(f"Error reading file {file_path}: {e}")

    # # Parse and store files
    # print("Parsing and storing files...")
    # await parser_component.parse_files(file_objects)
    # documents = parser_component.get_parsed_documents()
    # storage.astore_documents(documents, overwrite=True)

    user_question = '''
        You are a world class personal trainer. 
        Given the following information about a user:
        {
           "body_goals": "Build Muscle",
           "allergies": ["Gluten"],
           "workout_frequency": "5+ Days",
           "experience": "Advanced",
           "muscle_group_target": "Chest and Tricep"
        }

        What exercises should the user do?
        workout_name: str
        reps: int
        sets: int
        rest: str
        completion: bool 
        image_url: Optional[str]
        rationale: str  # How does the exercise help or potentially harm the user?  
        '''

    # # Perform a synchronous query
    # start_time = time.perf_counter()
    # print("Performing sync query...")
    # response, source_nodes = query_engine.query(user_question)
    # end_time = time.perf_counter()
    # print(f"Sync query time: {end_time - start_time:.2f} seconds")

    # print(f"Answer: {response}")

    # print("Source Documents:")
    # for node in source_nodes:
    #     print(f"Document: {node.node.text}")

    # Perform an asynchronous query
    start_time = time.perf_counter()
    print("Performing async query...")
    response, source_nodes = await query_engine.aquery(user_question, similarity_top_k=20, sparse_top_k=50)
    end_time = time.perf_counter()
    print(f"Async query time: {end_time - start_time:.2f} seconds")

    print(f"Answer: {response}")
    # Extract search results from source nodes
    search_results = [node.node.text for node in source_nodes]

    # Limit to the first 10 search results
    search_results = search_results[:10]

    # Generate user needs
    utils = Utils(config)
    user_needs = utils.generate_user_needs(user_question)
    print(f"User Needs: {user_needs}")

    # Initialize a list to collect exercises
    exercises = []

    # For each search result, generate structured output
    for search_result in search_results:
        # Parse the search result to extract the Description_URL
        try:
            result_data = json.loads(search_result)
            description_url = result_data.get('Description_URL')
            if description_url:
                # Scrape the description from the URL using spider_cloud_scrape
                description = utils.spider_cloud_scrape(description_url)
            else:
                description = ""
        except json.JSONDecodeError:
            description = ""

        # add exercise description to search result
        search_result += f"\nDescription: {description}"

        # Generate structured output with the scraped description
        exercise = query_engine.structured_output(user_question, user_needs, search_result, ExerciseStructuredOutput)
        if exercise:
            exercises.append(exercise)
        else:
            print("Failed to generate structured exercise data for a search result.")


    final_response = utils.generate_final_response(user_question, user_needs, search_results)
    print(f"Final Response: {final_response}")

    # end of script
    print("End of script.")

    # Inspect the Qdrant collection
    print("Inspecting the Qdrant collection...")
    collection_info = query_engine.inspect_collection()
    print(f"Collection Name: {collection_name}")
    print(f"Collection Info: {collection_info}")
    print("End of script.")

    # Save all of the results to local file
    print("Saving all of the results to local file...")
    with open('example_usage_enabl_gym_data_results.txt', 'w') as f:
        f.write(f"User Question: {user_question}\n")
        f.write(f"Initial Response: {response}\n")
        f.write(f"Search Results: {[node.node.text for node in source_nodes]}\n")
        f.write("Structured Exercise Data: \n")
        for exercise in exercises:
            exercise.completion = False
            f.write(str(exercise) + "\n")
        f.write(f"Final Response: {final_response}\n")
        f.write(f"Collection Name: {collection_name}\n")
        f.write(f"Collection Info: {collection_info}\n")

if __name__ == '__main__':
    asyncio.run(main())

