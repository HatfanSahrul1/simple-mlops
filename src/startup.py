import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import config
from src.data.data_preprocessor import HousePricePreprocessor

app = FastAPI(title="House Price Prediction API")

# Global State
model = None
model_columns = None
preprocessor = None

class HouseInput(BaseModel):
    # Minimal input sesuai dataset
    MSSubClass: int
    MSZoning: str
    LotArea: int
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    GrLivArea: int
    TotalBsmtSF: int = 0
    FirstFlrSF: int
    SecondFlrSF: int = 0
    GarageCars: int = 0
    GarageArea: int = 0
    
    class Config:
        populate_by_name = True

@app.on_event("startup")
def load_artifacts():
    global model, model_columns, preprocessor
    try:
        if os.path.exists(config.MODEL_PATH):
            model = joblib.load(config.MODEL_PATH)
            model_columns = joblib.load(config.COLS_PATH)
            preprocessor = joblib.load(config.PREPROCESSOR_PATH)
            print("Model loaded from shared volume!")
        else:
            print("WARNING: Model not found. Waiting for training...")
    except Exception as e:
        print(f"Error loading model: {e}")

def save_live_data(data_dict):
    """Simpan data request user ke CSV untuk dimonitor"""
    try:
        df = pd.DataFrame([data_dict])
        df = df.rename(columns={'FirstFlrSF': '1stFlrSF', 'SecondFlrSF': '2ndFlrSF'})
        
        header = not os.path.exists(config.LIVE_DATA_PATH)
        df.to_csv(config.LIVE_DATA_PATH, mode='a', header=header, index=False)
        print(f"Data logged to {config.LIVE_DATA_PATH}")
    except Exception as e:
        print(f"Logging failed: {e}")

@app.post("/predict")
def predict(data: HouseInput, background_tasks: BackgroundTasks):
    background_tasks.add_task(save_live_data, data.dict())
    
    if not model:
        load_artifacts()
        if not model:
            raise HTTPException(status_code=503, detail="Model is training, please wait.")

    try:
        df_input = pd.DataFrame([data.dict()])
        df_input = df_input.rename(columns={'FirstFlrSF': '1stFlrSF', 'SecondFlrSF': '2ndFlrSF'})
        
        df_processed = preprocessor.preprocess(df_input, is_training=False)
        df_final = df_processed.reindex(columns=model_columns, fill_value=0)
        
        pred_log = model.predict(df_final)
        pred_val = np.expm1(pred_log)
        
        return {"price": float(pred_val[0]), "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)