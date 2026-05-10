"""
Simple training script for credit card fraud detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

def train_simple_model():
    """Train a simple XGBoost model on credit card data"""
    
    # Load credit card data
    data_path = 'data/credit_card/creditcard.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        print("Please place creditcard.csv in data/credit_card/ folder")
        return None
    
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.4f}%)")
    
    # Prepare features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Remove Time column if exists
    if 'Time' in X.columns:
        X = X.drop('Time', axis=1)
    
    # Scale Amount
    if 'Amount' in X.columns:
        scaler = StandardScaler()
        X['Amount'] = scaler.fit_transform(X[['Amount']])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Fraud: {y_train.sum()}")
    print(f"Test set: {X_test.shape}, Fraud: {y_test.sum()}")
    
    # Train XGBoost with class imbalance handling
    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    
    fraud_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_pos_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\nFraud Recall: {fraud_recall:.4f} ({fraud_recall*100:.2f}%)")
    print(f"False Positive Rate: {false_pos_rate:.4f} ({false_pos_rate*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_fraud_model.pkl')
    joblib.dump(scaler, 'models/amount_scaler.pkl')
    
    print("\n✅ Model saved to models/xgboost_fraud_model.pkl")
    
    return model

if __name__ == "__main__":
    train_simple_model()