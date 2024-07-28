import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

# Data ingesting
class Mode_Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols:list):
        self.cols = cols
        
    def fit(self, X:pd.DataFrame, y=None):
        self.dict = {}
        for col in self.cols:
            self.dict[col] = X[col].mode()[0]
        return self

    def transform(self, X:pd.DataFrame, y=None):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].fillna(self.dict[col])
        return X
    
class Mean_Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols:list):
        self.cols = cols
        
    def fit(self, X:pd.DataFrame, y=None):
        self.dict = {}
        for col in self.cols:
            self.dict[col] = X[col].mean()
        return self

    def transform(self, X:pd.DataFrame, y=None):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].fillna(self.dict[col])
        return X

class Label_Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols:list):
        self.cols = cols

    def fit(self, X, y=None):
        self.encoder = LabelEncoder()
        return self
    
    def transform(self, X:pd.DataFrame, y=None):
        X = X.copy()
        for col in self.cols:
            X[col] = self.encoder.fit_transform(X[col])
        return X

class Log_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols:list):
        self.cols = cols

    def fit(self, X:pd.DataFrame, y=None):
        return self

    def transform(self, X:pd.DataFrame, y=None):
        X = X.copy()
        for col in self.cols:
            X[col] = np.log(X[col])
        return X

class Drop_Columns(BaseEstimator, TransformerMixin):
    def __init__(self, cols:list):
        self.cols = cols

    def fit(self, X:pd.DataFrame, y=None):
        return self

    def transform(self, X:pd.DataFrame, y=None):
        X = X.copy()
        X = X.drop(self.cols, axis='columns', inplace=False)
        return X