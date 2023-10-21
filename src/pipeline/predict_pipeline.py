import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            print("Loading model and preprocessor...")
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("Transforming features using the preprocessor...")
            # Ensure `features` is a DataFrame, if not, convert it
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            
            data_scaled = preprocessor.transform(features)

            print("Predicting using the model...")
            preds = model.predict(data_scaled)

            print("Prediction successful.")
            return preds
        
        except Exception as e:
            raise CustomException("Prediction failed", sys)

class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education,
                 lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            # Prepare data as a dictionary
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Create a DataFrame from the dictionary
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException("Failed to convert data to DataFrame", sys)
