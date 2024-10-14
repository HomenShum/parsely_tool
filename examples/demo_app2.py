import streamlit as st
import nest_asyncio
import asyncio
import json
import time
from io import BytesIO
from typing import Optional, List
from pydantic import BaseModel
import os
import glob
import pandas as pd

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

class FilteredRecipe(BaseModel):
    Title: str
    Ingredients: List[str]
    Total_Calories: float
    Total_Fat: float
    Total_Carbohydrates: float
    Total_Sugars: float
    Total_Proteins: float

# Function to ingest and store recipes data (Cached)
async def ingest_recipes_data(recipe_file_paths: List[str], config: Config) -> Optional[Storage]:
    try:
        utils = Utils(config)
        utils.setup_logging(config.verbose)

        parser_component = Parser(config)
        query_engine = QueryEngine(config, collection_name="ENABL_recipes_data_collection")
        storage = Storage(config, collection_name="ENABL_recipes_data_collection")

        # Convert file paths to BytesIO objects
        recipe_files_io = []
        for file_path in recipe_file_paths:
            try:
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                    file_obj = BytesIO(file_content)
                    file_obj.name = os.path.basename(file_path)
                    recipe_files_io.append(file_obj)
            except Exception as e:
                logger.error(f"Error reading recipe file {file_path}: {e}")

        # Parse and store Recipes Data
        logger.info("Parsing and storing recipes data...")
        await parser_component.parse_files(recipe_files_io)
        recipes_documents = parser_component.get_parsed_documents()
        storage.astore_documents(recipes_documents, overwrite=True)

        return storage
    except Exception as e:
        logger.error(f"Error in ingest_recipes_data: {e}")
        return None

# Function to ingest and store nutrition data (Cached)
async def ingest_nutrition_data(nutrition_file_paths: List[str], config: Config) -> Optional[Storage]:
    try:
        utils = Utils(config)
        utils.setup_logging(config.verbose)

        parser_component = Parser(config)
        query_engine = QueryEngine(config, collection_name="ENABL_food_data_group_collection")
        storage = Storage(config, collection_name="ENABL_food_data_group_collection")

        # Convert file paths to BytesIO objects
        nutrition_files_io = []
        for file_path in nutrition_file_paths:
            try:
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                    file_obj = BytesIO(file_content)
                    file_obj.name = os.path.basename(file_path)
                    nutrition_files_io.append(file_obj)
            except Exception as e:
                logger.error(f"Error reading nutrition file {file_path}: {e}")

        # Parse and store Nutrition Data
        logger.info("Parsing and storing nutrition data...")
        await parser_component.parse_files(nutrition_files_io)
        nutrition_documents = parser_component.get_parsed_documents()
        storage.astore_documents(nutrition_documents, overwrite=True)

        return storage
    except Exception as e:
        logger.error(f"Error in ingest_nutrition_data: {e}")
        return None


# Asynchronous function to process user input and query exercises
async def fetch_exercises(user_data: dict) -> Optional[ExercisesResponse]:
    try:
        # Setup configuration
        config = Config()
        config.verbose = True
        config.use_azure_openai = True  # Set to True if using Azure OpenAI
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
    except Exception as e:
        logger.error(f"Error in fetch_exercises: {e}")
        return None

# Asynchronous function to process recipe filtering
async def fetch_filtered_recipes(protein_threshold: float, calorie_limit: float,
                                 dietary_restrictions: List[str], config: Config) -> List[FilteredRecipe]:
    try:
        utils = Utils(config)
        utils.setup_logging(config.verbose)

        # Initialize QueryEngine
        query_engine = QueryEngine(config, collection_name="ENABL_recipes_data_collection")
        query_engine.load_index()  # Ensure the index is loaded

        # Perform Matching of Ingredients and Nutrition Data
        logger.info("Matching ingredients with nutritional information...")
        # Assuming you have a method to fetch matched recipes from storage
        matched_recipes = query_engine.match_ingredients_from_storage()

        # Apply Macro-Based Filtering
        logger.info("Applying macro-based filtering...")
        filtered_recipes = query_engine.filter_recipes_by_macros(
            matched_recipes,
            protein_threshold=protein_threshold,
            calorie_limit=calorie_limit
        )

        # Apply Dietary Restrictions
        final_filtered_recipes = query_engine.apply_dietary_restrictions(filtered_recipes, dietary_restrictions)

        # Convert to Pydantic Models for Structured Output
        structured_recipes = []
        for recipe in final_filtered_recipes:
            try:
                structured_recipes.append(FilteredRecipe(**recipe))
            except Exception as e:
                logger.error(f"Error converting recipe to FilteredRecipe model: {e}")

        return structured_recipes
    except Exception as e:
        logger.error(f"Error in fetch_filtered_recipes: {e}")
        return []


# Function to get list of files from a directory
def get_files_from_directory(directory: str, extensions: List[str] = ["*.csv", "*.xlsx", "*.json"]) -> List[str]:
    files = []
    for ext in extensions:
        files_found = glob.glob(os.path.join(directory, ext))
        files.extend(files_found)
    return files

# Caching the file listings to improve performance
@st.cache_data(show_spinner=False)
def list_dataset_files():
    try:
        food_data_dir = os.path.join("assets", "files", "food_data_group")
        recipes_data_dir = os.path.join("assets", "files", "recipes_data")

        food_files = get_files_from_directory(food_data_dir)
        recipes_files = get_files_from_directory(recipes_data_dir)

        logger.info(f"Found {len(food_files)} nutrition data files.")
        logger.info(f"Found {len(recipes_files)} recipe data files.")

        return food_files, recipes_files
    except Exception as e:
        logger.error(f"Error listing dataset files: {e}")
        return [], []

# Function to load a sample of the dataset for preview
@st.cache_data(show_spinner=False)
def load_dataset_preview(file_path: str, nrows: int = 5) -> Optional[pd.DataFrame]:
    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path, nrows=nrows)
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path, nrows=nrows)
        elif file_path.lower().endswith('.json'):
            df = pd.read_json(file_path, lines=True).head(nrows)
        else:
            st.warning(f"Unsupported file format: {file_path}")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file {file_path}: {e}")
        return None

# Asynchronous function to handle chatbot responses (Updated with sd_query fallback)
async def fetch_chat_response(user_input: str, selected_datasets: List[str]) -> Optional[str]:
    try:
        # Setup configuration
        config = Config()
        config.verbose = True
        config.use_azure_openai = True  # Set to True if using Azure OpenAI
        utils = Utils(config)
        utils.setup_logging(config.verbose)

        # Initialize QueryEngine for each selected collection
        query_engines = []
        if "Gym Exercise Data" in selected_datasets:
            query_engines.append(QueryEngine(config, collection_name="ENABL_gym_exercise_dataset_collection"))
        if "Nutrition Data" in selected_datasets:
            query_engines.append(QueryEngine(config, collection_name="ENABL_food_data_group_collection"))

        if not query_engines:
            return "Please select at least one dataset for me to use in answering your questions."

        # Formulate the user's question
        user_question = f'''
            You are a world-class personal trainer and nutrition expert.
            A user has asked the following question:
            "{user_input}"

            Provide a detailed and helpful response based on the available exercise and nutrition data.
        '''

        # Perform asynchronous queries on all selected collections
        responses = []
        source_nodes_all = []
        for qe in query_engines:
            start_time = time.perf_counter()
            logging.info(f"Chatbot performing async query on {qe.collection_name}...")
            response, source_nodes = await qe.aquery(user_question, similarity_top_k=20, sparse_top_k=50)
            end_time = time.perf_counter()
            logging.info(f"Chatbot async query time on {qe.collection_name}: {end_time - start_time:.2f} seconds")
            responses.append(response)
            source_nodes_all.extend(source_nodes)

        # Check if any results were returned
        if not responses or not any(responses) or not source_nodes_all:
            logging.warning("No data retrieved from the selected datasets.")
            
            # Use sd_query as a fallback
            sd_response_data = []
            for qe in query_engines:
                sd_results = qe.sd_query(user_question)
                sd_response_data.extend(sd_results)

            if not sd_response_data:
                # If sd_query also returns no results
                best_effort_answer = "I'm sorry, but I couldn't find specific information in the selected datasets to fully answer your question. However, based on my general knowledge, here's what I can suggest:"
                return f"**Here is what I understood:** {user_input}\n\n**Hope this can answer your question:**\n{best_effort_answer}"

            # Construct best-effort answer from sd_query results
            best_effort_answer = "I'm sorry, but I couldn't find specific information in the selected datasets to fully answer your question. However, based on my general knowledge, here's what I can suggest:\n\n"
            for idx, result in enumerate(sd_response_data, 1):
                best_effort_answer += f"{idx}. {result.get('Result', 'No information available.')}\n"

            return f"**Here is what I understood:** {user_input}\n\n**Hope this can answer your question:**\n{best_effort_answer}"

        # Combine all valid responses
        combined_response = " ".join([resp.text for resp in responses if resp])
        logging.info(f"Chatbot Combined Answer: {combined_response}")

        # Extract search results from source nodes
        search_results = [node.node.text for node in source_nodes_all]
        search_results = search_results[:10]  # Limit to first 10 results

        # Generate user needs
        user_needs = utils.generate_user_needs(user_question)
        logging.info(f"Chatbot User Needs: {user_needs}")

        # Initialize list to collect relevant information
        relevant_info = []

        # Process each search result to extract useful information
        for search_result in search_results:
            try:
                # Attempt to parse JSON
                result_data = json.loads(search_result)
                description_url = result_data.get('Description_URL', '')
                if description_url:
                    description = utils.spider_cloud_scrape(description_url)
                else:
                    description = ""
            except json.JSONDecodeError:
                description = ""

            # Append description to search result
            search_result += f"\nDescription: {description}"
            relevant_info.append(search_result)

        # Compose the final response
        if relevant_info:
            final_retrieved_info = "\n".join(relevant_info)
            final_response = f"Hope this can answer your question:\n{final_retrieved_info}"
        else:
            best_effort_answer = "I'm sorry, but I couldn't find specific information in the selected datasets to fully answer your question. However, based on my general knowledge, here's what I can suggest:"
            final_response = f"{best_effort_answer}"

        # Combine initial acknowledgment and final response
        full_response = f"**Here is what I understood:** {user_input}\n\n**Hope this can answer your question:**\n{final_response}"

        return full_response

    except Exception as e:
        logging.error(f"Error in fetch_chat_response: {e}")
        return "I'm sorry, but something went wrong while processing your request."



# Streamlit App Layout
def main():
    st.set_page_config(page_title="Personal Trainer & Recipe Assistant", layout="wide")
    st.title("🏋️‍♂️ Personal Trainer & Recipe Assistant")

    # Initialize configuration
    config = Config()
    config.verbose = True
    config.use_azure_openai = True  # Set based on your requirements

    # Initialize utilities
    utils = Utils(config)
    utils.setup_logging(config.verbose)

    # Create three tabs
    tab1, tab2, tab3 = st.tabs(["Personal Trainer", "Recipe Filter", "Chatbot"])

    ############################
    # Tab 1: Personal Trainer #
    ############################
    with tab1:
        st.header("🏋️ Personal Trainer Assistant")

        st.markdown("""
        ### Get Customized Exercise Recommendations
        Enter your fitness preferences and goals below to receive personalized exercise suggestions.
        """)

        # User Input Form for Exercises
        with st.form(key='user_input_form'):
            st.subheader("Your Fitness Profile")
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
                try:
                    exercises_response = asyncio.run(fetch_exercises(user_data))
                except Exception as e:
                    st.error(f"An error occurred while fetching exercises: {e}")
                    exercises_response = None

                if exercises_response and exercises_response.exercises:
                    st.success("Here are your recommended exercises:")
                    for idx, exercise in enumerate(exercises_response.exercises, 1):
                        st.markdown(f"### {idx}. {exercise.workout_name}")
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
                elif exercises_response:
                    st.warning("No exercises found based on your input. Please try adjusting your preferences.")

        # Optional: Display raw response data
        with st.expander("View Raw Exercise Data"):
            try:
                if submit_button and exercises_response:
                    raw_data = exercises_response.json(indent=4)
                    st.json(json.loads(raw_data))
            except Exception as e:
                st.error(f"Error displaying raw data: {e}")

    ###########################
    # Tab 2: Recipe Filter    #
    ###########################
    with tab2:
        st.header("🍲 Recipe Filter")

        st.markdown("""
        ### Get Filtered Recipes Based on Your Dietary Preferences
        Explore and preview datasets from our curated collections, set your nutritional goals and dietary restrictions to receive tailored recipe suggestions.
        """)

        # Automatically list dataset files
        food_files, recipes_files = list_dataset_files()

        # Display the directories being checked
        base_dir = os.path.dirname(os.path.abspath(__file__))
        st.markdown(f"**Base Directory:** {base_dir}")
        st.markdown(f"**Nutrition Data Directory:** {os.path.join(base_dir, 'parsely_tool', 'assets', 'files', 'food_data_group')}")
        st.markdown(f"**Recipes Data Directory:** {os.path.join(base_dir, 'parsely_tool', 'assets', 'files', 'recipes_data')}")

        # Combine all files with labels
        dataset_options = []

        # Add Food Data Group files
        for file in food_files:
            dataset_options.append({
                "name": os.path.basename(file),
                "path": file,
                "type": "Nutrition Data"
            })

        # Add Recipes Data files
        for file in recipes_files:
            dataset_options.append({
                "name": os.path.basename(file),
                "path": file,
                "type": "Recipe Data"
            })

        if not dataset_options:
            st.warning("No datasets found in the specified directories. Please ensure that the directories contain `.csv`, `.xlsx`, or `.json` files.")
        else:
            # Display available datasets for debugging
            st.markdown("**Available Datasets:**")
            for dataset in dataset_options:
                st.markdown(f"- **{dataset['name']}** ({dataset['type']})")

            # Ingestion Step: Ingest data once
            ingest_button = st.button("Ingest Datasets")

            if ingest_button:
                with st.spinner('Ingesting recipes and nutrition data...'):
                    # Ingest Recipes Data
                    recipes_storage = asyncio.run(ingest_recipes_data([d['path'] for d in dataset_options if d['type'] == "Recipe Data"], config))

                    # Ingest Nutrition Data
                    nutrition_storage = asyncio.run(ingest_nutrition_data([d['path'] for d in dataset_options if d['type'] == "Nutrition Data"], config))

                    if recipes_storage and nutrition_storage:
                        st.success("Datasets ingested and stored successfully!")
                    else:
                        st.error("Failed to ingest one or more datasets. Please check the logs for more details.")

            # Dataset Selection for Preview
            st.subheader("Dataset Preview")
            selected_dataset = st.selectbox(
                "Select a dataset to preview:",
                options=dataset_options,
                format_func=lambda x: f"{x['name']} ({x['type']})"
            )

            if selected_dataset:
                st.markdown(f"### Preview of `{selected_dataset['name']}` ({selected_dataset['type']})")
                df_preview = load_dataset_preview(selected_dataset['path'], nrows=5)
                if df_preview is not None:
                    st.dataframe(df_preview)
                else:
                    st.warning("Unable to display preview for the selected dataset.")

            # Filtering Criteria Form
            st.subheader("Set Your Filtering Criteria")

            with st.form(key='recipe_filter_form'):
                protein_threshold = st.number_input("Minimum Protein (grams)", min_value=0.0, value=20.0, step=1.0)
                calorie_limit = st.number_input("Maximum Calories", min_value=0.0, value=500.0, step=50.0)

                st.markdown("**Dietary Restrictions**")
                vegan = st.checkbox("Vegan")
                lactose = st.checkbox("Lactose Intolerant")
                gluten = st.checkbox("Gluten-Free")
                other_restrictions = st.text_input("Other Restrictions (comma-separated)", "")

                filter_submit = st.form_submit_button(label='Get Filtered Recipes')

            if filter_submit:
                if not food_files or not recipes_files:
                    st.error("No datasets available to process. Please ensure that the directories contain the necessary files.")
                else:
                    with st.spinner('Filtering recipes...'):
                        # Prepare dietary restrictions list
                        dietary_restrictions = []
                        if vegan:
                            dietary_restrictions.append('vegan')
                        if lactose:
                            dietary_restrictions.append('lactose')
                        if gluten:
                            dietary_restrictions.append('gluten')
                        if other_restrictions:
                            dietary_restrictions.extend([r.strip().lower() for r in other_restrictions.split(",") if r.strip()])

                        # Run the asynchronous function
                        try:
                            filtered_recipes = asyncio.run(fetch_filtered_recipes(
                                protein_threshold=protein_threshold,
                                calorie_limit=calorie_limit,
                                dietary_restrictions=dietary_restrictions,
                                config=config
                            ))
                        except Exception as e:
                            st.error(f"An error occurred while filtering recipes: {e}")
                            filtered_recipes = []

                        if filtered_recipes:
                            st.success(f"Found {len(filtered_recipes)} recipes matching your criteria:")
                            for idx, recipe in enumerate(filtered_recipes, 1):
                                st.markdown(f"### {idx}. {recipe.Title}")
                                st.markdown("**Ingredients:**")
                                for ingredient in recipe.Ingredients:
                                    st.markdown(f"- {ingredient}")
                                st.markdown(f"**Total Calories:** {recipe.Total_Calories} kcal")
                                st.markdown(f"**Total Fat:** {recipe.Total_Fat} g")
                                st.markdown(f"**Total Carbohydrates:** {recipe.Total_Carbohydrates} g")
                                st.markdown(f"**Total Sugars:** {recipe.Total_Sugars} g")
                                st.markdown(f"**Total Proteins:** {recipe.Total_Proteins} g")
                                st.markdown("---")
                        else:
                            st.warning("No recipes found matching your criteria. Please adjust your filters.")

            # Optional: Display raw recipe data
            with st.expander("View Raw Filtered Recipe Data"):
                try:
                    if filter_submit and filtered_recipes:
                        raw_recipes = [recipe.dict() for recipe in filtered_recipes]
                        st.json(raw_recipes)
                except Exception as e:
                    st.error(f"Error displaying raw data: {e}")

    ###########################
    # Tab 3: Chatbot          #
    ###########################
    with tab3:
        st.header("🤖 Chatbot Assistant")

        st.markdown("""
        ### Ask Your Fitness and Nutrition Questions
        Interact with our chatbot to get personalized advice and information based on your fitness and dietary preferences. You can select which datasets the chatbot should utilize to provide more accurate and relevant responses.
        """)

        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Dataset Selection for Chatbot
        st.subheader("Select Data Sources for Chatbot")
        dataset_options = ["Gym Exercise Data", "Nutrition Data"]
        selected_datasets = st.multiselect(
            "Choose which datasets the chatbot should use:",
            options=dataset_options,
            default=dataset_options  # Default to all selected
        )

        # Chat interface within a form to handle submissions properly
        with st.form(key='chat_form'):
            user_input = st.text_input("You:", key="chat_input")
            submit_chat = st.form_submit_button(label="Send")

        if submit_chat:
            if user_input.strip() == "":
                st.warning("Please enter a question before sending.")
            else:
                # Append user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_input})

                with st.spinner('Chatbot is processing your request...'):
                    # Fetch chatbot response
                    try:
                        chat_response = asyncio.run(fetch_chat_response(user_input, selected_datasets))
                    except Exception as e:
                        chat_response = "I'm sorry, but something went wrong while processing your request."

                if chat_response:
                    # Append chatbot response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": chat_response})

        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### Conversation")
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                elif message["role"] == "assistant":
                    st.markdown(f"**Chatbot:** {message['content']}")

if __name__ == "__main__":
    main()
