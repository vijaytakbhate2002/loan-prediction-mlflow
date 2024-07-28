from config import config
import pandas as pd
import numpy as np
import mlflow
import argparse
from train_pipeline import train_pipe
from Best_model import grid_search
from sklearn.model_selection import train_test_split
from mlflow.sklearn import log_model
from sklearn.metrics import recall_score, precision_score, f1_score
from pipeline import classifier_pipe, preprocessing_pipe
import warnings
warnings.filterwarnings('ignore')

def eval(y_true, y_pred) -> dict:
    metrics_dict = {}
    metrics_dict['recall_score'] = recall_score(y_true=y_true, y_pred=y_pred)
    metrics_dict['precision_score'] = precision_score(y_true=y_true, y_pred=y_pred)
    metrics_dict['f1_score'] = f1_score(y_true=y_true, y_pred=y_pred)
    return metrics_dict

def mlflow_run(user_params=None):
    mlflow.set_experiment("Logistic Regression")
    if user_params:
        user_params = vars(user_params)
    with mlflow.start_run() as f:
        Data = pd.read_csv(config.TRAIN_PATH)
        Data_X = preprocessing_pipe.fit_transform(X=Data[config.X_COLS])
        Data_y = [1 if val == 'N' else 0 for val in Data[config.TARGET]]

        X_train, X_test, y_train, y_test = train_test_split(Data_X, Data_y, test_size=0.2, stratify=Data[config.TARGET])
        grids = grid_search.best_estimator(X_train, X_test, y_train, y_test,)
        params= grids.best_params_
        train_score = grids.score(X_train, y_train)
        test_score = grids.score(X_test, y_test)
        y_pred = grids.predict(X_test)

        # parameter logging
        mlflow.log_params(params)

        # metrics logging
        mlflow.log_metrics(eval(y_true=y_test, y_pred=y_pred))
        mlflow.log_metric("train_score", train_score)
        mlflow.log_metric("test_score", test_score)

        # best model logging
        log_model(sk_model=grids, artifact_path='trained_best_model')
        pass
    mlflow.end_run()
    pass

if __name__ == '__main__':
    # arg = argparse.ArgumentParser()
    # arg.add_argument('--penalty', '-p', type=str, default='l2')
    # arg.add_argument('--C', '-c', type=float, default=0.2)
    # arg.add_argument('--max_iter', '-mi', type=int, default=100)
    # parsed_args = arg.parse_args()
    # print(parsed_args)
    mlflow_run()
