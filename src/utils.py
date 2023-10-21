import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.

    Parameters:
        file_path (str): The path to save the object to.
        obj (object): The object to save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple models and return a report of R-squared scores.

    Parameters:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        X_test (numpy.ndarray): Testing features.
        y_test (numpy.ndarray): Testing labels.
        models (dict): Dictionary of models to evaluate.
        param (dict): Dictionary of model parameters.

    Returns:
        dict: A dictionary with model names as keys and R-squared scores as values.
    """
    try:
        report = {}
        for model_name, model in models.items():
            params = param.get(model_name, {})
            gs = GridSearchCV(model, params, cv=3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load an object from a file using pickle.

    Parameters:
        file_path (str): The path to load the object from.

    Returns:
        object: The loaded object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
