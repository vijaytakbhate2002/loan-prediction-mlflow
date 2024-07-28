from config import config
from sklearn.pipeline import Pipeline
from Data_processing.process import Mean_Imputer, Mode_Imputer, Label_Encoder, Log_Transformer, Drop_Columns 
from config import config
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

classifier_pipe = Pipeline([
    ('Mode Imputer', Mode_Imputer(cols=config.CAT_COLS)),
    ('Mean Imputer', Mean_Imputer(cols=config.NUM_COLS)),
    ('Label Encoder', Label_Encoder(cols=config.CAT_COLS)),
    ('Log Transformer', Log_Transformer(cols=config.NUM_COLS)),
    ('Drop columns', Drop_Columns(cols=config.DROP_COLS)),
    ('Logistic Regression', LogisticRegression(random_state=0))
])

preprocessing_pipe = Pipeline([
    ('Mode Imputer', Mode_Imputer(cols=config.CAT_COLS)),
    ('Mean Imputer', Mean_Imputer(cols=config.NUM_COLS)),
    ('Label Encoder', Label_Encoder(cols=config.CAT_COLS)),
    ('Log Transformer', Log_Transformer(cols=config.NUM_COLS)),
    ('Drop columns', Drop_Columns(cols=config.DROP_COLS))
])