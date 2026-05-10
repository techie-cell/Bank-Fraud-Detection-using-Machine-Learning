"""
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
    
    print(f"\nXGBoost Results:")
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
