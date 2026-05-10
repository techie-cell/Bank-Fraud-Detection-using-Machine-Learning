"""
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
        print(f"\n{name} Results:")
        print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    joblib.dump(lr_model, 'models/logistic_regression.pkl')
    joblib.dump(rf_model, 'models/random_forest.pkl')
    
    return lr_model, rf_model

if __name__ == "__main__":
    train_baseline_models()
