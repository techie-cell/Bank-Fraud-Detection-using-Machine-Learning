"""
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
        """Load and preprocess Credit Card dataset"""
        df = pd.read_csv(filepath)
        
        if 'Class' in df.columns:
            df['is_fraud'] = df['Class']
            df = df.drop('Class', axis=1)
        elif 'is_fraud' not in df.columns:
            raise ValueError("Dataset must contain 'Class' or 'is_fraud' column")
        
        # Remove Time column if exists
        if 'Time' in df.columns:
            df = df.drop('Time', axis=1)
            
        # Separate features and target
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud']
        
        # Handle Amount column
        if 'Amount' in X.columns:
            X['Amount'] = np.log1p(X['Amount'])
            
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def load_synthetic_data(self, filepath):
        """Load and preprocess Synthetic dataset"""
        df = pd.read_csv(filepath)
        
        # Drop transaction_id column (non-numeric, not useful for training)
        if 'transaction_id' in df.columns:
            df = df.drop('transaction_id', axis=1)
        
        # Convert timestamp to numerical features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['days_from_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days
            df = df.drop('timestamp', axis=1)
        
        # Encode categorical variables
        categorical_cols = ['transaction_type', 'merchant_category', 'country']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = df[col].fillna('unknown')
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Handle missing values - only for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Separate features and target
        if 'is_fraud' not in df.columns:
            raise ValueError("Dataset must contain 'is_fraud' column")
        
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud']
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def preprocess(self, data_path, test_size=0.2, use_smote=True):
        """Complete preprocessing pipeline"""
        
        if self.dataset_type == 'credit_card':
            X, y = self.load_credit_card_data(data_path)
        else:
            X, y = self.load_synthetic_data(data_path)
        
        print(f"Original data shape: {X.shape}")
        print(f"Fraud ratio: {y.mean():.4f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Apply SMOTE for handling imbalance
        if use_smote:
            try:
                smote = SMOTE(random_state=42, sampling_strategy=0.1)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"After SMOTE - Training shape: {X_train.shape}, Fraud ratio: {y_train.mean():.4f}")
            except Exception as e:
                print(f"SMOTE failed: {e}. Using class weights instead.")
        
        return X_train, X_test, y_train, y_test
    
    def save(self, path):
        """Save preprocessor state"""
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'dataset_type': self.dataset_type
        }, path)
        
    def load(self, path):
        """Load preprocessor state"""
        state = joblib.load(path)
        self.scaler = state['scaler']
        self.label_encoders = state['label_encoders']
        self.feature_columns = state['feature_columns']
        self.dataset_type = state['dataset_type']

# Utility function for feature engineering
def create_features(df):
    """Create additional features for fraud detection"""
    
    # Day/Time patterns
    if 'hour' in df.columns:
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
        
    # Amount patterns
    if 'Amount' in df.columns:
        df['amount_log'] = np.log1p(df['Amount'])
        df['amount_squared'] = df['Amount'] ** 2
        
    # Risk combination
    risk_cols = [col for col in df.columns if 'risk' in col.lower()]
    if len(risk_cols) >= 2:
        df['combined_risk'] = df[risk_cols].mean(axis=1)
        df['max_risk'] = df[risk_cols].max(axis=1)
        
    return df