import streamlit as st
import nest_asyncio
import asyncio
import json
import time
from io import BytesIO
from typing import Optional, List
from pydantic import BaseModel

# Apply nest_asyncio to allow nested event loops in Streamlit
nest_asyncio.apply()

# Import parsely_tool components
from parsely_tool.parsers import Parser
from parsely_tool.storage import Storage
from parsely_tool.query import QueryEngine
from parsely_tool.utils import Utils
from parsely_tool.config import Config

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Pydantic models for structured output
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

# Asynchronous function to process user input and query exercises
async def fetch_exercises(user_data: dict) -> ExercisesResponse:
    # Setup configuration
    config = Config()
    config.verbose = True
    config.use_azure_openai = True
    Utils(config).setup_logging(config.verbose)

    # Initialize components
    parser_component = Parser(config)
    collection_name = "ENABL_gym_exercise_dataset_collection"
    storage = Storage(config, collection_name=collection_name)
    query_engine = QueryEngine(config, collection_name=collection_name)

    user_question = f'''
        You are a world-class personal trainer. 
        Given the following information about a user:
        {json.dumps(user_data, indent=4)}

        What exercises should the user do?
        workout_name: str
        reps: int
        sets: int
        rest: str
        completion: bool 
        image_url: Optional[str]
        rationale: str  # How does the exercise help or potentially harm the user?  
    '''

    # Perform an asynchronous query
    start_time = time.perf_counter()
    logger.info("Performing async query...")
    response, source_nodes = await query_engine.aquery(user_question, similarity_top_k=20, sparse_top_k=50)
    end_time = time.perf_counter()
    logger.info(f"Async query time: {end_time - start_time:.2f} seconds")

    logger.info(f"Answer: {response}")

    # Extract search results from source nodes
    search_results = [node.node.text for node in source_nodes]
    search_results = search_results[:10]  # Limit to first 10 results

    # Generate user needs
    utils = Utils(config)
    user_needs = utils.generate_user_needs(user_question)
    logger.info(f"User Needs: {user_needs}")

    # Initialize list to collect exercises
    exercises = []

    # Process each search result to generate structured exercise data
    for search_result in search_results:
        try:
            result_data = json.loads(search_result)
            description_url = result_data.get('Description_URL')
            if description_url:
                description = utils.spider_cloud_scrape(description_url)
            else:
                description = ""
        except json.JSONDecodeError:
            description = ""

        # Append description to search result
        search_result += f"\nDescription: {description}"

        # Generate structured output
        exercise = query_engine.structured_output(user_question, user_needs, search_result, ExerciseStructuredOutput)
        if exercise:
            exercises.append(exercise)
        else:
            logger.warning("Failed to generate structured exercise data for a search result.")

    final_response = utils.generate_final_response(user_question, user_needs, search_results)
    logger.info(f"Final Response: {final_response}")

    # Inspect the Qdrant collection (optional)
    collection_info = query_engine.inspect_collection()
    logger.info(f"Collection Name: {collection_name}")
    logger.info(f"Collection Info: {collection_info}")

    # Return the structured exercises
    return ExercisesResponse(exercises=exercises)

# Streamlit App Layout
def main():
    st.set_page_config(page_title="Personal Trainer Assistant", layout="wide")
    st.title("🏋️ Personal Trainer Assistant")

    st.markdown("""
    ### Get Customized Exercise Recommendations
    Enter your fitness preferences and goals below to receive personalized exercise suggestions.
    """)

    # User Input Form
    with st.form(key='user_input_form'):
        st.header("Your Fitness Profile")
        body_goals = st.selectbox("Body Goals", ["Build Muscle", "Lose Weight", "Increase Endurance", "Improve Flexibility", "Other"])
        allergies = st.text_input("Allergies (comma-separated)", "Gluten")
        workout_frequency = st.selectbox("Workout Frequency", ["1-2 Days", "3-4 Days", "5+ Days"])
        experience = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
        muscle_group_target = st.text_input("Muscle Group Target", "Chest and Tricep")

        submit_button = st.form_submit_button(label='Get Exercises')

    if submit_button:
        with st.spinner('Fetching exercises...'):
            user_data = {
                "body_goals": body_goals,
                "allergies": [a.strip() for a in allergies.split(",") if a.strip()],
                "workout_frequency": workout_frequency,
                "experience": experience,
                "muscle_group_target": muscle_group_target
            }

            # Run the asynchronous function
            exercises_response = asyncio.run(fetch_exercises(user_data))

            if exercises_response.exercises:
                st.success("Here are your recommended exercises:")
                for idx, exercise in enumerate(exercises_response.exercises, 1):
                    st.subheader(f"{idx}. {exercise.workout_name}")
                    cols = st.columns([1, 3])
                    with cols[0]:
                        if exercise.image_url:
                            st.image(exercise.image_url, width=150)
                    with cols[1]:
                        st.markdown(f"**Reps:** {exercise.reps}")
                        st.markdown(f"**Sets:** {exercise.sets}")
                        st.markdown(f"**Rest:** {exercise.rest}")
                        st.markdown(f"**Rationale:** {exercise.rationale}")
                    st.markdown("---")
            else:
                st.warning("No exercises found based on your input. Please try adjusting your preferences.")

    # Optional: Display raw response data
    with st.expander("View Raw Response Data"):
        try:
            if submit_button:
                raw_data = exercises_response.json(indent=4)
                st.json(json.loads(raw_data))
        except Exception as e:
            st.error(f"Error displaying raw data: {e}")

if __name__ == "__main__":
    main()
