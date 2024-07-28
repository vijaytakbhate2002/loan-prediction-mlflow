import os

# folder and file paths
C_PATH = "C:\\Users\\admin\\OneDrive\\Desktop\\ML Ops Workspace\\4-mlflow\\Loan Eligibillity Prediction 3\\config\\config.py"
ROOT_PATH = '\\'.join(__file__.split('\\')[:-2])
README = os.path.join(ROOT_PATH, 'README')
CONTRIBUTING = os.path.join(ROOT_PATH, 'CONTRIBUTING')

# Dataset path
DATA = os.path.join(ROOT_PATH, 'dataset')
TRAIN_PATH = os.path.join(DATA, 'train.csv')
TEST_PATH = os.path.join(DATA, 'test.csv')

# dataset_cols
PROJECT_NAME = 'Loan Eliigibility Prediction'
COLS = ['Gender','Married','Dependents','Education','Self_Employed',
        'ApplicantIncome','LoanAmount',
        'Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']

X_COLS = ['Gender','Married','Dependents','Education','Self_Employed',
          'ApplicantIncome','LoanAmount',
          'Loan_Amount_Term','Credit_History','Property_Area']

TARGET = 'Loan_Status'

CAT_COLS = ['Gender','Married','Dependents','Education','Self_Employed','Credit_History', 'Property_Area']
NUM_COLS = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

DROP_COLS = []

# Grid search CV parameter grid
LR_PARAMS = {
    'Logistic Regression__C': [0.01, 1.0, 10.0, 100.0],
    'Logistic Regression__penalty': ['l2', 'elasticnet', 'none'],
    'Logistic Regression__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
    'Logistic Regression__max_iter': [100, 200, 300]
}