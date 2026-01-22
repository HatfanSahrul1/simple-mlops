import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils import config
from src.data.data_preprocessor import HousePricePreprocessor

def train_model(data_path=config.TRAIN_DATA_PATH):
    print(f"Loading data from {data_path}...")
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File {data_path} not found.")
        sys.exit(1)

    if 'SalePrice' in df.columns:
        if len(df) > 100: 
            df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)
        
        df["SalePrice"] = np.log1p(df["SalePrice"])
        y_train = df.SalePrice.values
        X = df.drop(['SalePrice', 'Id'], axis=1, errors='ignore')
    else:
        print("Error: Target variable 'SalePrice' not found.")
        sys.exit(1)
    
    preprocessor = HousePricePreprocessor()
    
    X_processed = preprocessor.preprocess(X, is_training=True)
    
    model_columns = list(X_processed.columns)
    
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    
    with mlflow.start_run():
        params = {
            'colsample_bytree': 0.46, 'learning_rate': 0.05, 'max_depth': 3,
            'n_estimators': 100,
            'subsample': 0.52, 'random_state': 7
        }
        mlflow.log_params(params)
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_processed, y_train)
        
        print("Saving artifacts to shared volume...")
        joblib.dump(model, config.MODEL_PATH)
        joblib.dump(model_columns, config.COLS_PATH)
        joblib.dump(preprocessor, config.PREPROCESSOR_PATH)
        
        mlflow.log_artifact(config.MODEL_PATH)
        print("Training Finished Successfully.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else config.TRAIN_DATA_PATH
    train_model(path)