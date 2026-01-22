import pandas as pd
import numpy as np
import time
import os
import sys
from scipy.stats import ks_2samp

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import config

def monitor_service():
    print("=== Monitor Service Started ===")
    
    while True:
        time.sleep(config.CHECK_INTERVAL)
        
        if not os.path.exists(config.LIVE_DATA_PATH):
            continue
            
        try:
            live_data = pd.read_csv(config.LIVE_DATA_PATH)
            
            if len(live_data) < config.MIN_DATA_POINTS:
                print(f"Data collected: {len(live_data)}/{config.MIN_DATA_POINTS}")
                continue
                
            print("Threshold reached. Checking for drift...")
            
            ref_data = pd.read_csv(config.TRAIN_DATA_PATH)
            
            drift_detected = False
            if 'GrLivArea' in live_data.columns and 'GrLivArea' in ref_data.columns:
                stat, p_value = ks_2samp(ref_data['GrLivArea'], live_data['GrLivArea'])
                print(f"KS Test GrLivArea: p-value={p_value:.5f}")
                
                if p_value < config.DRIFT_THRESHOLD:
                    drift_detected = True
                    print("DRIFT DETECTED! Data distribution has changed significantly.")
            
            if drift_detected:
                print("Generating dummy labels for retraining...")
                
                mean_price = ref_data['SalePrice'].mean()
                std_price = ref_data['SalePrice'].std()
                
                live_data['SalePrice'] = np.random.normal(mean_price, std_price, len(live_data))
                live_data['SalePrice'] = live_data['SalePrice'].abs()
                
                live_data.to_csv(config.RETRAIN_DATA_PATH, index=False)
                
                print("🚀 Triggering Retraining...")
                exit_code = os.system(f"python src/models/train.py {config.RETRAIN_DATA_PATH}")
                
                if exit_code == 0:
                    print("VVVVVVVV Retraining Success! New model saved. VVVVVVVV")
                    os.remove(config.LIVE_DATA_PATH)
                    print("Live data cleared.")
                else:
                    print("xxxxxxxxx Retraining Failed. xxxxxxxxx")
            else:
                print("No big drift detected.")
                
        except Exception as e:
            print(f"Monitor Error: {e}")

if __name__ == "__main__":
    time.sleep(10)
    monitor_service()