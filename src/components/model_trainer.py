import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import numpy as np

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params):
        report = {}
        best_model = None
        best_score = -np.inf

        for model_name, model in models.items():
            try:
                param = params.get(model_name, {})

                if param:
                    gs = GridSearchCV(model, param, cv=3, scoring='r2', verbose=0, error_score='raise')
                    gs.fit(X_train, y_train)
                    model = gs.best_estimator_
                else:
                    model.fit(X_train, y_train)

                y_test_pred = model.predict(X_test)
                test_model_score = r2_score(y_test, y_test_pred)

                report[model_name] = test_model_score

                if test_model_score > best_score:
                    best_score = test_model_score
                    best_model = model

            except Exception as e:
                logging.warning(f"[{model_name}] failed during evaluation: {e}")

        return report, best_model

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False, train_dir=None),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGBoost Regressor": XGBRegressor()
            }

            params = {
                "Random Forest": {
                    'n_estimators': [100],
                    'max_depth': [10],
                    'min_samples_split': [2]
                },
                "Decision Tree": {
                    'max_depth': [10],
                    'min_samples_split': [2]
                },
                "Gradient Boosting": {
                    'n_estimators': [100],
                    'learning_rate': [0.1],
                    'max_depth': [3]
                },
                "Linear Regression": {},
                "K-Neighbours Regressor": {
                    'n_neighbors': [5]
                },
                "CatBoost Regressor": {
                    'iterations': [1000],
                    'learning_rate': [0.1],
                    'depth': [6]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50],
                    'learning_rate': [1.0]
                },
                "XGBoost Regressor": {
                    'n_estimators': [100],
                    'learning_rate': [0.1],
                    'max_depth': [3]
                }
            }

            model_report, best_model = self.evaluate_models(X_train, y_train, X_test, y_test, models, params)

            print(model_report)
            print("\nBest model found:", type(best_model).__name__)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2

        except Exception as e:
            raise CustomException(e, sys)
