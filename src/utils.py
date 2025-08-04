import os
import sys
import dill
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Save a Python object to a file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, models, X_test, y_test, params):
    """
    Train and evaluate multiple models using GridSearchCV.
    Returns a report dictionary with R2 scores on the test data.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training and tuning model: {model_name}")
            param_grid = params.get(model_name, {})

            gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            logging.info(f"{model_name} -> Train R2: {train_model_score:.4f}, Test R2: {test_model_score:.4f}")
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a Python object from a file using dill.
    """
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys)
