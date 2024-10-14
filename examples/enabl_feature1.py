# examples/enabl_feature1.py

r"""
TODO:
100724 Design Flow:
Design an API to ingest 
User Preference: {"body_goals":"Build Muscle","allergies":"Gluten","workout_frequency":"5+ Days","experience":"Advanced"} 
Exercise muscle group target: {"muscle_group_target": "Chest and Tricep"}
API should look through the following excel columns in gym-exercises-dataset C:\Users\hshum\OneDrive\Desktop\Python\CafeCorner\parsely_tool\assets\files\Gym Exercises Dataset.xlsx: 
"Exercise_Name"
"Exercise_Image"
"Exercise_Image1"
"muscle_gp"
"Rating" 
The API output is 
{ 
    "image_url": "exercise_image or exercise_image1", 
    "Exercise_Name": "Flat Bench Press"
}

"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Parsely Gym Exercises API",
    description="API to filter gym exercises based on user preferences and muscle group targets.",
    version="1.0.0"
)

# Define Pydantic models for request and response

class UserPreference(BaseModel):
    body_goals: Optional[str] = Field(None, example="Build Muscle")
    allergies: Optional[str] = Field(None, example="Gluten")
    workout_frequency: Optional[str] = Field(None, example="5+ Days")
    experience: Optional[str] = Field(None, example="Advanced")

class MuscleGroupTarget(BaseModel):
    muscle_group_target: Optional[str] = Field(None, example="Chest and Tricep")

class ExerciseResponse(BaseModel):
    image_url: Optional[str] = Field(None, example="http://example.com/image1.jpg")
    Exercise_Name: str = Field(..., example="Flat Bench Press")

# Load and preprocess the dataset at startup

DATASET_PATH = r'C:\Users\hshum\OneDrive\Desktop\Python\CafeCorner\parsely_tool\assets\files\Gym Exercises Dataset.xlsx'

@app.get("/")
def read_root():
    return {"message": "Welcome to the Parsely Gym Exercises API!"}

@app.on_event("startup")
def load_dataset():
    """
    Load the gym exercises dataset from the specified path at startup.
    
    Checks if the dataset exists at the specified path, and if it does, loads it into a Pandas DataFrame.
    If the dataset does not exist, logs an error and raises a FileNotFoundError.
    
    Also checks if the dataset contains the required columns, and if not, logs an error and raises a ValueError.
    
    If any other exception occurs while loading the dataset, logs an error and raises the exception.
    """
    global df_exercises
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset not found at path: {DATASET_PATH}")
        raise FileNotFoundError(f"Dataset not found at path: {DATASET_PATH}")
    try:
        df_exercises = pd.read_excel(DATASET_PATH)
        # Normalize column names to lowercase for consistency
        df_exercises.columns = df_exercises.columns.str.strip().str.lower()
        
        # Define required columns in lowercase
        required_columns = ["exercise_name", "exercise_image", "exercise_image1", "muscle_gp", "rating"]
        if not all(col in df_exercises.columns for col in required_columns):
            missing = list(set(required_columns) - set(df_exercises.columns))
            logger.error(f"Missing columns in dataset: {missing}")
            raise ValueError(f"Missing columns in dataset: {missing}")
        logger.info("Dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise e

# Helper function to filter exercises


def filter_exercises(user_pref: UserPreference, muscle_target: MuscleGroupTarget) -> List[ExerciseResponse]:
    """
    Filter exercises based on user preferences and muscle group targets.

    Args:
        user_pref (UserPreference): User preferences such as body goals, allergies, workout frequency, and experience.
        muscle_target (MuscleGroupTarget): Muscle group target to filter exercises by.

    Returns:
        List[ExerciseResponse]: A list of ExerciseResponse objects, each containing an exercise name and image URL.
    """
    filtered_df = df_exercises.copy()

    # Apply muscle group target filter
    if muscle_target.muscle_group_target:
        # Split the muscle groups by 'and' and strip whitespace
        targets = [grp.strip().lower() for grp in muscle_target.muscle_group_target.split('and')]
        # Create a regex pattern to match any of the target muscle groups
        pattern = '|'.join(targets)
        filtered_df = filtered_df[filtered_df['muscle_gp'].str.lower().str.contains(pattern, na=False)]
        logger.debug(f"Filtered by muscle groups: {targets}")

    # Further filters based on user preferences can be added here
    # Example: If body_goals is "Build Muscle", filter by 'rating' >= 4
    if user_pref.body_goals:
        if user_pref.body_goals.lower() == "build muscle":
            filtered_df = filtered_df[filtered_df['rating'] >= 4]  # Assuming rating is out of 5
            logger.debug("Filtered by body goals: Build Muscle with rating >= 4")

    # Additional filtering based on other preferences (allergies, workout_frequency, experience) can be implemented here

    # Select top-rated exercises
    top_exercises = filtered_df.sort_values(by='rating', ascending=False).head(20)

    # Prepare the response
    exercises = []
    for _, row in top_exercises.iterrows():
        # Check if either image exists and is not NaN
        image_url = None
        if pd.notna(row['exercise_image']):
            image_url = row['exercise_image']
        elif pd.notna(row['exercise_image1']):
            image_url = row['exercise_image1']

        exercises.append(
            ExerciseResponse(
                image_url=image_url,  # This will be None if no valid image is found
                Exercise_Name=row['exercise_name']
            )
        )
    
    logger.info(f"Found {len(exercises)} exercises matching the criteria.")
    return exercises


# Define the API endpoint

@app.post("/filter_exercises/", response_model=List[ExerciseResponse])
def filter_exercises_endpoint(user_preference: UserPreference, muscle_group: MuscleGroupTarget):
    """
    API endpoint to filter exercises based on user preferences and muscle group.

    Args:
        user_preference (UserPreference): User preferences.
        muscle_group (MuscleGroupTarget): Muscle group to target.

    Returns:
        List[ExerciseResponse]: List of filtered exercises.

    Raises:
        HTTPException: If no exercises are found or an internal server error occurs.
    """
    try:
        exercises = filter_exercises(user_preference, muscle_group)
        if not exercises:
            raise HTTPException(status_code=404, detail="No exercises found matching the criteria.")
        return exercises
    except Exception as e:
        logger.error(f"Error in filter_exercises_endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error.")

# For running via command line: uvicorn examples.enabl_feature1:app --reload

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("enabl_feature1:app", host="0.0.0.0", port=8000, reload=True)
