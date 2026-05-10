"""
Model Retraining Pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from datetime import datetime
from src.train_xgboost import train_xgboost

def retrain_model(new_data_path):
    print(f"Retraining with new data from {new_data_path}")
    new_model = train_xgboost(data_path=new_data_path)
    print("Retraining complete")
    return new_model

if __name__ == "__main__":
    if len(sys.argv) > 1:
        retrain_model(sys.argv[1])
