from sklearn.model_selection import GridSearchCV
from pipeline import classifier_pipe
import pandas as pd
from config import config

def best_estimator(X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.Series, y_test:pd.Series) -> dict:
    """Function finds best estimator with grid search cv
        Returns dictionary"""
    grid_search = GridSearchCV(estimator=classifier_pipe, param_grid=config.LR_PARAMS, cv=5, n_jobs=3)
    grid_search.fit(X_train, y_train)
    return grid_search
