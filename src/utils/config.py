import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
LIVE_DATA_PATH = os.path.join(DATA_DIR, "live_data.csv")
RETRAIN_DATA_PATH = os.path.join(DATA_DIR, "retrain_data.csv")

MODEL_PATH = os.path.join(DATA_DIR, "house_price_model.joblib")
COLS_PATH = os.path.join(DATA_DIR, "model_columns.joblib")
PREPROCESSOR_PATH = os.path.join(DATA_DIR, "preprocessor.joblib")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
EXPERIMENT_NAME = "House_Price_Production"

DRIFT_THRESHOLD = 0.05
MIN_DATA_POINTS = 10
CHECK_INTERVAL = 10