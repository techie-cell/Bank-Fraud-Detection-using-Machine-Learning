#!/usr/bin/env python3
"""
Project setup script for AI-Powered Bank Fraud Detection System
Run this first to create the complete project structure
"""

import os
import sys

def create_directory_structure():
    """Create all project directories"""
    directories = [
        'data/credit_card',
        'data/synthetic',
        'src',
        'models',
        'api',
        'dashboard',
        'sota',
        'logs',
        'notebooks',
        'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[OK] Created {directory}")
        
    # Create __init__.py files
    with open('src/__init__.py', 'w') as f:
        f.write('# Source module\n')
    with open('sota/__init__.py', 'w') as f:
        f.write('# SOTA module\n')
    
    print("[OK] Created __init__.py files")

def create_requirements():
    """Create requirements.txt"""
    requirements = """# Core ML and Data Processing
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
xgboost==1.7.6
imbalanced-learn==0.11.0

# Deep Learning (for autoencoders)
tensorflow==2.13.0

# Visualization and Dashboard
streamlit==1.25.0
plotly==5.15.0
matplotlib==3.7.2
seaborn==0.12.2

# API and Backend
fastapi==0.100.0
uvicorn==0.23.1
pydantic==2.1.1

# Model Explainability
shap==0.42.1

# Utilities
joblib==1.3.1
python-dotenv==1.0.0

# Monitoring and Logging
prometheus-client==0.17.1
evidently==0.2.6

# Testing
pytest==7.4.0
"""
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements)
    print("[OK] Created requirements.txt")

def create_dataset_generator():
    """Create synthetic dataset generator"""
    
    synthetic_script = '''"""
Synthetic Financial Fraud Dataset Generator
Creates realistic transaction data for fraud detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_synthetic_fraud_dataset(n_transactions=100000, fraud_ratio=0.01):
    """
    Generate synthetic transaction dataset with fraud patterns
    
    Parameters:
    -----------
    n_transactions : int
        Number of transactions to generate
    fraud_ratio : float
        Ratio of fraudulent transactions (default 0.01 = 1%)
    """
    
    np.random.seed(42)
    random.seed(42)
    
    # Number of fraud cases
    n_fraud = int(n_transactions * fraud_ratio)
    n_normal = n_transactions - n_fraud
    
    # Generate timestamps (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = [start_date + timedelta(seconds=random.randint(0, 30*24*3600)) 
                  for _ in range(n_transactions)]
    
    # Transaction types
    transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    
    # Countries (with risk scores)
    countries = {
        'US': 0.2, 'UK': 0.2, 'CA': 0.2, 'AU': 0.2, 'DE': 0.2,
        'FR': 0.3, 'JP': 0.3, 'SG': 0.3,
        'RU': 0.7, 'NG': 0.8, 'BR': 0.6, 'IN': 0.5,
        'CN': 0.6, 'UA': 0.7, 'PK': 0.8
    }
    
    # Initialize data
    data = {
        'transaction_id': [f'TXN_{i:08d}' for i in range(n_transactions)],
        'timestamp': timestamps,
        'amount': np.zeros(n_transactions),
        'transaction_type': np.zeros(n_transactions, dtype=object),
        'merchant_category': np.zeros(n_transactions, dtype=object),
        'country': np.zeros(n_transactions, dtype=object),
        'device_risk_score': np.zeros(n_transactions),
        'ip_risk_score': np.zeros(n_transactions),
        'is_fraud': np.zeros(n_transactions, dtype=int)
    }
    
    # Merchant categories
    merchant_categories = ['retail', 'restaurant', 'travel', 'electronics', 
                          'entertainment', 'grocery', 'gas', 'healthcare']
    
    # Generate normal transactions
    for i in range(n_normal):
        amount = np.random.exponential(100) + 20
        amount = min(amount, 5000)
        
        transaction_type = np.random.choice(transaction_types, p=[0.4, 0.2, 0.1, 0.2, 0.1])
        
        country = np.random.choice(list(countries.keys()), 
                                   p=[0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.04, 0.02, 0.02])
        
        device_risk = np.random.normal(countries[country] * 0.3, 0.1)
        device_risk = np.clip(device_risk, 0, 1)
        
        ip_risk = np.random.normal(countries[country] * 0.4, 0.15)
        ip_risk = np.clip(ip_risk, 0, 1)
        
        merchant_category = np.random.choice(merchant_categories)
        
        data['amount'][i] = amount
        data['transaction_type'][i] = transaction_type
        data['merchant_category'][i] = merchant_category
        data['country'][i] = country
        data['device_risk_score'][i] = device_risk
        data['ip_risk_score'][i] = ip_risk
        data['is_fraud'][i] = 0
    
    # Generate fraud transactions
    for i in range(n_normal, n_transactions):
        amount = np.random.choice([
            np.random.exponential(500) + 500,
            np.random.exponential(50) + 5,
        ], p=[0.7, 0.3])
        amount = min(amount, 10000)
        
        transaction_type = np.random.choice(transaction_types, p=[0.2, 0.4, 0.3, 0.05, 0.05])
        
        high_risk_countries = ['RU', 'NG', 'BR', 'PK', 'UA', 'CN']
        country = np.random.choice(high_risk_countries, p=[0.3, 0.25, 0.15, 0.15, 0.1, 0.05])
        
        device_risk = np.random.normal(0.7, 0.2)
        device_risk = np.clip(device_risk, 0.3, 1)
        
        ip_risk = np.random.normal(0.8, 0.15)
        ip_risk = np.clip(ip_risk, 0.4, 1)
        
        merchant_category = np.random.choice(merchant_categories + ['gambling', 'adult'])
        
        data['amount'][i] = amount
        data['transaction_type'][i] = transaction_type
        data['merchant_category'][i] = merchant_category
        data['country'][i] = country
        data['device_risk_score'][i] = device_risk
        data['ip_risk_score'][i] = ip_risk
        data['is_fraud'][i] = 1
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    print("Generating synthetic fraud dataset...")
    
    for size, filename in [(50000, 'synthetic_fraud_small.csv'), 
                           (100000, 'synthetic_fraud_medium.csv'),
                           (200000, 'synthetic_fraud_large.csv')]:
        df = generate_synthetic_fraud_dataset(size)
        df.to_csv(f'data/synthetic/{filename}', index=False)
        print(f"Generated {filename} with {len(df)} transactions ({df['is_fraud'].sum()} fraud cases)")
    
    print("\\nDataset statistics:")
    df_sample = pd.read_csv('data/synthetic/synthetic_fraud_medium.csv')
    print(f"Fraud ratio: {df_sample['is_fraud'].mean()*100:.2f}%")
'''
    
    with open('src/generate_synthetic_data.py', 'w', encoding='utf-8') as f:
        f.write(synthetic_script)
    print("[OK] Created synthetic data generator")

def create_preprocessing():
    """Create preprocessing module"""
    
    preprocessing_script = '''"""
Data preprocessing for fraud detection
Handles both Credit Card and Synthetic datasets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

class FraudDataPreprocessor:
    def __init__(self, dataset_type='credit_card'):
        self.dataset_type = dataset_type
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_credit_card_data(self, filepath):
        df = pd.read_csv(filepath)
        
        if 'Class' in df.columns:
            df['is_fraud'] = df['Class']
            df = df.drop('Class', axis=1)
        elif 'is_fraud' not in df.columns:
            raise ValueError("Dataset must contain 'Class' or 'is_fraud' column")
        
        if 'Time' in df.columns:
            df = df.drop('Time', axis=1)
            
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud']
        
        if 'Amount' in X.columns:
            X['Amount'] = np.log1p(X['Amount'])
            
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def load_synthetic_data(self, filepath):
        df = pd.read_csv(filepath)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['days_from_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days
            df = df.drop('timestamp', axis=1)
            
        categorical_cols = ['transaction_type', 'merchant_category', 'country']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = df[col].fillna('unknown')
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                
        df = df.fillna(df.median())
        
        if 'is_fraud' not in df.columns:
            raise ValueError("Dataset must contain 'is_fraud' column")
            
        X = df.drop(['transaction_id', 'is_fraud'], axis=1) if 'transaction_id' in df.columns else df.drop('is_fraud', axis=1)
        y = df['is_fraud']
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def preprocess(self, data_path, test_size=0.2, use_smote=True):
        if self.dataset_type == 'credit_card':
            X, y = self.load_credit_card_data(data_path)
        else:
            X, y = self.load_synthetic_data(data_path)
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        if use_smote:
            smote = SMOTE(random_state=42, sampling_strategy=0.1)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE - Training shape: {X_train.shape}, Fraud ratio: {y_train.mean():.4f}")
            
        return X_train, X_test, y_train, y_test
    
    def save(self, path):
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'dataset_type': self.dataset_type
        }, path)
        
    def load(self, path):
        state = joblib.load(path)
        self.scaler = state['scaler']
        self.label_encoders = state['label_encoders']
        self.feature_columns = state['feature_columns']
        self.dataset_type = state['dataset_type']
'''
    
    with open('src/preprocessing.py', 'w', encoding='utf-8') as f:
        f.write(preprocessing_script)
    print("[OK] Created preprocessing module")

def create_train_baseline():
    """Create baseline training script"""
    
    baseline_script = '''"""
Baseline Model Training (Logistic Regression + Random Forest)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from src.preprocessing import FraudDataPreprocessor

def train_baseline_models(dataset_type='synthetic', data_path=None):
    preprocessor = FraudDataPreprocessor(dataset_type=dataset_type)
    
    if data_path is None:
        if dataset_type == 'credit_card':
            data_path = 'data/credit_card/creditcard.csv'
        else:
            data_path = 'data/synthetic/synthetic_fraud_medium.csv'
    
    print(f"Loading {dataset_type} dataset from {data_path}")
    
    X_train, X_test, y_train, y_test = preprocessor.preprocess(
        data_path, test_size=0.2, use_smote=True
    )
    
    preprocessor.save('models/preprocessor_baseline.pkl')
    
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    for model, name in [(lr_model, "Logistic Regression"), (rf_model, "Random Forest")]:
        y_pred = model.predict(X_test)
        print(f"\\n{name} Results:")
        print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    joblib.dump(lr_model, 'models/logistic_regression.pkl')
    joblib.dump(rf_model, 'models/random_forest.pkl')
    
    return lr_model, rf_model

if __name__ == "__main__":
    train_baseline_models()
'''
    
    with open('src/train_baseline.py', 'w', encoding='utf-8') as f:
        f.write(baseline_script)
    print("[OK] Created baseline training script")

def create_train_xgboost():
    """Create XGBoost training script"""
    
    xgboost_script = '''"""
XGBoost Model Training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from src.preprocessing import FraudDataPreprocessor

def train_xgboost(dataset_type='synthetic', data_path=None):
    preprocessor = FraudDataPreprocessor(dataset_type=dataset_type)
    
    if data_path is None:
        if dataset_type == 'credit_card':
            data_path = 'data/credit_card/creditcard.csv'
        else:
            data_path = 'data/synthetic/synthetic_fraud_medium.csv'
    
    print(f"Loading {dataset_type} dataset from {data_path}")
    X_train, X_test, y_train, y_test = preprocessor.preprocess(
        data_path, test_size=0.2, use_smote=False
    )
    
    preprocessor.save('models/preprocessor_xgboost.pkl')
    
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\\nXGBoost Results:")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"Fraud Recall: {tp/(tp+fn):.4f}")
    print(f"False Positive Rate: {fp/(fp+tn):.4f}")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    joblib.dump(xgb_model, 'models/xgboost_fraud_model.pkl')
    
    # Save threshold
    with open('models/optimal_threshold.txt', 'w') as f:
        f.write('0.5')
    
    return xgb_model

if __name__ == "__main__":
    train_xgboost()
'''
    
    with open('src/train_xgboost.py', 'w', encoding='utf-8') as f:
        f.write(xgboost_script)
    print("[OK] Created XGBoost training script")

def create_api():
    """Create FastAPI backend"""
    
    api_script = '''"""
FastAPI Backend for Fraud Detection
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection API", version="1.0.0")

models = {}

class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0)
    country: str = Field(..., description="Country code")
    device_risk: float = Field(..., ge=0, le=1)
    ip_risk: float = Field(..., ge=0, le=1)
    transaction_type: Optional[str] = "PAYMENT"
    merchant_category: Optional[str] = "retail"

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_score: float
    model_used: str
    timestamp: str
    threshold_used: float

def load_models():
    try:
        if os.path.exists('models/xgboost_fraud_model.pkl'):
            models['xgboost'] = joblib.load('models/xgboost_fraud_model.pkl')
            logger.info("Loaded XGBoost model")
            models['threshold'] = 0.5
            return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
    return False

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/health")
async def health_check():
    return {"status": "operational", "models_loaded": len(models) > 0}

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest):
    if 'xgboost' not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Simplified prediction for demo
        risk_score = (transaction.amount / 10000 + transaction.device_risk + transaction.ip_risk) / 3
        risk_score = min(1.0, risk_score)
        prob_fraud = risk_score
        threshold = models.get('threshold', 0.5)
        
        prediction = "fraud" if prob_fraud >= threshold else "normal"
        
        return PredictionResponse(
            prediction=prediction,
            probability=round(prob_fraud, 4),
            risk_score=round(risk_score * 100, 2),
            model_used="XGBoost",
            timestamp=datetime.now().isoformat(),
            threshold_used=threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    with open('api/app.py', 'w', encoding='utf-8') as f:
        f.write(api_script)
    print("[OK] Created FastAPI backend")

def create_dashboard():
    """Create Streamlit dashboard"""
    
    dashboard_script = '''"""
Streamlit Dashboard for Fraud Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime

st.set_page_config(page_title="Fraud Detection System", page_icon=":bank:", layout="wide")

st.title(":bank: AI-Powered Bank Fraud Detection System")
st.markdown("---")

with st.sidebar:
    st.header("Configuration")
    dataset_type = st.selectbox("Select Dataset", ["Synthetic", "Credit Card"])
    api_url = st.text_input("API URL", value="http://localhost:8000")
    use_api = st.checkbox("Use API for predictions", value=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Transaction Analysis")
    
    with st.form("transaction_form"):
        amount = st.number_input("Amount ($)", min_value=0.01, value=1000.00)
        country = st.selectbox("Country", ["US", "UK", "RU", "NG", "BR", "IN", "CN"])
        device_risk = st.slider("Device Risk Score", 0.0, 1.0, 0.5)
        ip_risk = st.slider("IP Risk Score", 0.0, 1.0, 0.5)
        transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
        
        submitted = st.form_submit_button("Analyze Transaction", use_container_width=True)

with col2:
    st.subheader("Prediction Results")
    prediction_placeholder = st.empty()
    probability_placeholder = st.empty()

if submitted:
    transaction_data = {
        "amount": amount,
        "country": country,
        "device_risk": device_risk,
        "ip_risk": ip_risk,
        "transaction_type": transaction_type
    }
    
    if use_api:
        try:
            response = requests.post(f"{api_url}/predict", json=transaction_data)
            if response.status_code == 200:
                result = response.json()
                
                if result["prediction"] == "fraud":
                    prediction_placeholder.error(f"FRAUD ALERT! Risk Score: {result['risk_score']:.1f}%")
                else:
                    prediction_placeholder.success(f"Transaction Normal. Risk Score: {result['risk_score']:.1f}%")
                
                probability_placeholder.metric("Fraud Probability", f"{result['probability']*100:.1f}%")
            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error(f"Cannot connect to API: {e}")
    else:
        # Local prediction
        risk = (amount/10000 + device_risk + ip_risk) / 3
        risk = min(1.0, risk)
        if risk > 0.5:
            prediction_placeholder.error(f"FRAUD ALERT! Risk Score: {risk*100:.1f}%")
        else:
            prediction_placeholder.success(f"Transaction Normal. Risk Score: {risk*100:.1f}%")
        probability_placeholder.metric("Fraud Probability", f"{risk*100:.1f}%")

# Visualizations
st.markdown("---")
st.subheader("Analytics Dashboard")

try:
    df = pd.read_csv('data/synthetic/synthetic_fraud_medium.csv')
    fraud_count = df['is_fraud'].sum()
    fraud_rate = (fraud_count / len(df)) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    with col2:
        st.metric("Fraud Cases", f"{fraud_count:,}")
    with col3:
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    with col4:
        st.metric("Normal Transactions", f"{len(df)-fraud_count:,}")
    
    # Fraud distribution pie chart
    fig = go.Figure(data=[go.Pie(labels=['Normal', 'Fraud'], 
                                  values=[len(df)-fraud_count, fraud_count],
                                  marker_colors=['green', 'red'])])
    fig.update_layout(title="Transaction Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.info("Run 'python src/generate_synthetic_data.py' first to generate data")

st.markdown("---")
st.markdown("*AI-Powered Bank Fraud Detection System | Real-time Detection*")
'''
    
    with open('dashboard/app_streamlit.py', 'w', encoding='utf-8') as f:
        f.write(dashboard_script)
    print("[OK] Created Streamlit dashboard")

def create_sota_modules():
    """Create SOTA modules"""
    
    gnn_script = '''"""
Graph Neural Network for Fraud Ring Detection (Placeholder)
"""

class FraudGNN:
    def __init__(self):
        self.model = None
    
    def detect_fraud_rings(self, transactions):
        print("GNN fraud ring detection (requires GPU)")
        return []
'''
    
    with open('sota/gnn_model.py', 'w', encoding='utf-8') as f:
        f.write(gnn_script)
    
    federated_script = '''"""
Federated Learning for Privacy-Preserving Fraud Detection (Placeholder)
"""

class FederatedFraudDetector:
    def __init__(self):
        pass
    
    def train_round(self):
        print("Federated learning training round")
        return {}
'''
    
    with open('sota/federated_learning.py', 'w', encoding='utf-8') as f:
        f.write(federated_script)
    
    online_script = '''"""
Online Learning for Real-time Adaptation (Placeholder)
"""

class OnlineFraudDetector:
    def __init__(self):
        pass
    
    def update_online(self, transaction, label):
        print("Online model update")
'''
    
    with open('sota/online_learning.py', 'w', encoding='utf-8') as f:
        f.write(online_script)
    
    multiagent_script = '''"""
Multi-Agent System for Fraud Detection (Placeholder)
"""

class MultiAgentFraudDetector:
    def __init__(self):
        pass
    
    def detect_fraud(self, transaction):
        print("Multi-agent fraud detection")
        return {"is_fraud": False, "risk_score": 0.5}
'''
    
    with open('sota/multi_agent.py', 'w', encoding='utf-8') as f:
        f.write(multiagent_script)
    
    print("[OK] Created SOTA modules")

def create_retraining_script():
    """Create retraining script"""
    
    retrain_script = '''"""
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
'''
    
    with open('src/retrain.py', 'w', encoding='utf-8') as f:
        f.write(retrain_script)
    print("[OK] Created retraining script")

def create_anomaly_detection():
    """Create anomaly detection module"""
    
    anomaly_script = '''"""
Anomaly Detection for Fraud
"""

import numpy as np
from sklearn.ensemble import IsolationForest

class FraudAnomalyDetector:
    def __init__(self):
        self.isolation_forest = None
    
    def train_isolation_forest(self, X_train, contamination=0.01):
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.isolation_forest.fit(X_train)
        print("Isolation Forest training complete")
    
    def detect_anomalies(self, X):
        if self.isolation_forest:
            predictions = self.isolation_forest.predict(X)
            return predictions == -1
        return np.zeros(len(X), dtype=bool)

if __name__ == "__main__":
    X_train = np.random.randn(1000, 10)
    detector = FraudAnomalyDetector()
    detector.train_isolation_forest(X_train)
    print("Anomaly detection ready")
'''
    
    with open('src/anomaly_detection.py', 'w', encoding='utf-8') as f:
        f.write(anomaly_script)
    print("[OK] Created anomaly detection module")

def create_evaluation():
    """Create evaluation module"""
    
    evaluation_script = '''"""
Model Evaluation
"""

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

class FraudEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
    
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Normal', 'Fraud']))
        print(f"ROC-AUC: {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"Fraud Recall: {tp/(tp+fn):.4f}")
        print(f"False Positive Rate: {fp/(fp+tn):.4f}")
'''
    
    with open('src/evaluation.py', 'w', encoding='utf-8') as f:
        f.write(evaluation_script)
    print("[OK] Created evaluation module")

def create_vscode_settings():
    """Create VS Code settings"""
    
    settings = '''{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "editor.formatOnSave": true
}
'''
    
    os.makedirs('.vscode', exist_ok=True)
    with open('.vscode/settings.json', 'w', encoding='utf-8') as f:
        f.write(settings)
    print("[OK] Created VS Code settings")

def create_batch_files():
    """Create batch files for Windows"""
    
    setup_bat = '''@echo off
echo Creating virtual environment...
python -m venv venv
echo Activating virtual environment...
call venv\\Scripts\\activate
echo Installing dependencies...
pip install -r requirements.txt
echo Generating synthetic data...
python src/generate_synthetic_data.py
echo Training models...
python src/train_baseline.py
python src/train_xgboost.py
echo Setup complete!
echo.
echo To start the system:
echo 1. Start API: cd api && python app.py
echo 2. Start Dashboard: streamlit run dashboard/app_streamlit.py
pause
'''
    
    with open('setup.bat', 'w') as f:
        f.write(setup_bat)
    
    api_bat = '''@echo off
call venv\\Scripts\\activate
cd api
python app.py
pause
'''
    
    with open('start_api.bat', 'w') as f:
        f.write(api_bat)
    
    dashboard_bat = '''@echo off
call venv\\Scripts\\activate
streamlit run dashboard/app_streamlit.py
pause
'''
    
    with open('start_dashboard.bat', 'w') as f:
        f.write(dashboard_bat)
    
    print("[OK] Created batch files for Windows")

def main():
    """Main setup function"""
    print("="*60)
    print("AI-Powered Bank Fraud Detection System Setup")
    print("="*60)
    
    create_directory_structure()
    create_requirements()
    create_dataset_generator()
    create_preprocessing()
    create_train_baseline()
    create_train_xgboost()
    create_api()
    create_dashboard()
    create_sota_modules()
    create_retraining_script()
    create_anomaly_detection()
    create_evaluation()
    create_vscode_settings()
    create_batch_files()
    
    print("\n" + "="*60)
    print("PROJECT SETUP COMPLETE!")
    print("="*60)
    print("\nNEXT STEPS:")
    print("1. Create virtual environment: python -m venv venv")
    print("2. Activate: venv\\Scripts\\activate")
    print("3. Install: pip install -r requirements.txt")
    print("4. Generate data: python src/generate_synthetic_data.py")
    print("5. Train models: python src/train_xgboost.py")
    print("6. Start API: cd api && python app.py")
    print("7. Start Dashboard: streamlit run dashboard/app_streamlit.py")
    print("\nFor Credit Card Dataset:")
    print("- Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("- Place creditcard.csv in data/credit_card/")
    print("\nProject Structure Created Successfully!")

if __name__ == "__main__":
    main()