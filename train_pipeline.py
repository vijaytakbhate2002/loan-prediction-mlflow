from config import config
import pandas as pd
from config import config
import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline import classifier_pipe

def train_pipe() -> None:
    """Args: None
        Return: None
        func: takes train data, calls fit method of pipeline, dump pipeline"""
    train_X = pd.read_csv(config.TRAIN_PATH)
    train_y = train_X[config.TARGET].map({'N':0, 'Y':1})
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.2, shuffle=True, stratify=train_y)
    classifier_pipe.fit(X_train, y_train)
    train_score = classifier_pipe.score(X_train, y_train)
    test_score = classifier_pipe.score(X_test, y_test)
    return train_score, test_score