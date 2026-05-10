"""
Fraud Detection API - With Correct Paths
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
from datetime import datetime

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessor - Using correct paths (going up one level)
print("="*50)
print("Loading Model and Scaler...")
print("="*50)

model = None
scaler = None
label_encoders = None
threshold = 0.5

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Define paths in the parent directory (where models are stored)
model_path = os.path.join(parent_dir, 'models', 'xgboost_fraud_model.pkl')
scaler_path = os.path.join(parent_dir, 'models', 'feature_scaler.pkl')
encoders_path = os.path.join(parent_dir, 'models', 'label_encoders.pkl')
threshold_path = os.path.join(parent_dir, 'models', 'optimal_threshold.txt')

print(f"Looking for model at: {model_path}")

# Load model
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("✅ Model loaded")
else:
    print(f"❌ Model not found at {model_path}")

# Load scaler
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("✅ Scaler loaded")
else:
    print(f"⚠️ Scaler not found at {scaler_path}")

# Load label encoders
if os.path.exists(encoders_path):
    label_encoders = joblib.load(encoders_path)
    print("✅ Label encoders loaded")
else:
    print(f"⚠️ Label encoders not found at {encoders_path}")

# Load threshold
if os.path.exists(threshold_path):
    with open(threshold_path, 'r') as f:
        threshold = float(f.read())
    print(f"✅ Threshold loaded: {threshold:.3f}")
else:
    print(f"⚠️ Using default threshold: {threshold}")

print("="*50)

class Transaction(BaseModel):
    amount: float
    country: str = "US"
    device_risk: float = 0.5
    ip_risk: float = 0.5
    transaction_type: str = "PAYMENT"
    merchant_category: str = "retail"

def encode_value(value, encoder_dict, encoder_name):
    """Encode categorical value using label encoder"""
    try:
        if encoder_dict and encoder_name in encoder_dict:
            encoder = encoder_dict[encoder_name]
            if value in encoder.classes_:
                return int(encoder.transform([value])[0])
            else:
                print(f"⚠️ {value} not found in {encoder_name}, using default")
                return 0
    except Exception as e:
        print(f"⚠️ Encoding error for {encoder_name}: {e}")
        return 0
    return 0

@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoders_loaded": label_encoders is not None,
        "threshold": threshold,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        print("\n" + "="*50)
        print("📊 Processing Transaction")
        print("="*50)
        
        # Get current time features
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        days_from_start = 0
        
        # Encode categorical variables using loaded encoders
        transaction_type_encoded = encode_value(transaction.transaction_type, label_encoders, 'transaction_type')
        merchant_encoded = encode_value(transaction.merchant_category, label_encoders, 'merchant_category')
        country_encoded = encode_value(transaction.country, label_encoders, 'country')
        
        print(f"Input Values:")
        print(f"  Amount: ${transaction.amount}")
        print(f"  Country: {transaction.country} -> {country_encoded}")
        print(f"  Transaction Type: {transaction.transaction_type} -> {transaction_type_encoded}")
        print(f"  Merchant: {transaction.merchant_category} -> {merchant_encoded}")
        print(f"  Device Risk: {transaction.device_risk}")
        print(f"  IP Risk: {transaction.ip_risk}")
        print(f"  Hour: {hour}, Day: {day_of_week}")
        
        # Create raw features in the exact order expected
        raw_features = np.array([[
            float(transaction.amount),
            float(transaction_type_encoded),
            float(merchant_encoded),
            float(country_encoded),
            float(transaction.device_risk),
            float(transaction.ip_risk),
            float(hour),
            float(day_of_week),
            float(days_from_start)
        ]])
        
        print(f"\nRaw features (first 5): {raw_features[0][:5]}")
        
        # Scale features if scaler is available
        if scaler is not None:
            scaled_features = scaler.transform(raw_features)
            print(f"Features scaled successfully")
            features_to_use = scaled_features
        else:
            print("⚠️ No scaler available, using raw features")
            features_to_use = raw_features
        
        # Make prediction
        if model is not None:
            prob_array = model.predict_proba(features_to_use)
            prob = float(prob_array[0, 1])
            print(f"\n✅ Model prediction: {prob:.4f} ({prob*100:.2f}%)")
        else:
            # Fallback rule-based
            prob = 0.0
            if transaction.amount > 5000:
                prob += 0.4
            if transaction.country in ['RU', 'NG', 'BR']:
                prob += 0.3
            prob += transaction.device_risk * 0.15
            prob += transaction.ip_risk * 0.15
            prob = min(prob, 1.0)
            print(f"\n⚠️ Using fallback: {prob:.4f} ({prob*100:.2f}%)")
        
        # Apply threshold
        prediction = "fraud" if prob >= threshold else "normal"
        
        print(f"Threshold: {threshold:.3f}")
        print(f"Decision: {prediction.upper()}")
        print("="*50)
        
        return {
            "prediction": prediction,
            "probability": round(prob, 4),
            "risk_score": round(prob * 100, 2),
            "model_used": "XGBoost" if model else "Rule-Based",
            "timestamp": datetime.now().isoformat(),
            "threshold_used": round(threshold, 3)
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "prediction": "error",
            "probability": 0.5,
            "risk_score": 50.0,
            "model_used": "Error",
            "timestamp": datetime.now().isoformat(),
            "threshold_used": threshold,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 Starting Fraud Detection API")
    print("="*50)
    uvicorn.run(app, host="127.0.0.1", port=8000)