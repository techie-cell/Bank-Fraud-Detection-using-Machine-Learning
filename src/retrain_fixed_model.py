"""
Retrain XGBoost model with proper scaling
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from datetime import datetime

def retrain_fixed_model():
    """Retrain the fraud detection model with proper scaling"""
    
    print("="*60)
    print("Retraining Fraud Detection Model")
    print("="*60)
    
    # Load synthetic data
    data_path = 'data/synthetic/synthetic_fraud_medium.csv'
    
    if not os.path.exists(data_path):
        print(f"Generating synthetic data first...")
        os.system('python src/generate_synthetic_data.py')
    
    print(f"\nLoading data from {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    
    # Feature engineering
    print("\nPreparing features...")
    
    # Convert timestamp to features
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['days_from_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days
        df = df.drop('timestamp', axis=1)
    
    # Encode categorical variables
    categorical_cols = ['transaction_type', 'merchant_category', 'country']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].fillna('unknown')
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            print(f"  Encoded {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    # Drop unnecessary columns
    if 'transaction_id' in df.columns:
        df = df.drop('transaction_id', axis=1)
    
    # Separate features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    print(f"\nFeatures: {X.columns.tolist()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    
    # Calculate class weights for imbalance
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # Train XGBoost
    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"ROC-AUC: {auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fraud_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_pos_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"Fraud Recall: {fraud_recall:.4f} ({fraud_recall*100:.1f}%)")
    print(f"False Positive Rate: {false_pos_rate:.4f} ({false_pos_rate*100:.1f}%)")
    
    print("\nConfusion Matrix:")
    print(f"  True Negatives: {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives: {tp}")
    
    # Find optimal threshold
    thresholds = np.linspace(0.1, 0.9, 50)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred_t = (y_pred_proba >= threshold).astype(int)
        cm_t = confusion_matrix(y_test, y_pred_t)
        if cm_t.shape == (2, 2):
            tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
            precision = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
            recall = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
    print(f"\nOptimal threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
    
    # Save model, scaler, and encoders
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_fraud_model.pkl')
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    
    with open('models/optimal_threshold.txt', 'w') as f:
        f.write(str(best_threshold))
    
    print("\n" + "="*60)
    print("✅ MODEL SAVED SUCCESSFULLY!")
    print("="*60)
    print("Files saved:")
    print("  - models/xgboost_fraud_model.pkl")
    print("  - models/feature_scaler.pkl")
    print("  - models/label_encoders.pkl")
    print("  - models/optimal_threshold.txt")
    
    return model, scaler, label_encoders

if __name__ == "__main__":
    retrain_fixed_model()