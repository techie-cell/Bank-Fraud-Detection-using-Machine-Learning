"""
Synthetic Financial Fraud Dataset Generator
Creates realistic transaction data for fraud detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

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
    countries_list = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'SG', 'RU', 'NG', 'BR', 'IN', 'CN', 'UA', 'PK']
    
    # Probability distribution for normal transactions (must sum to 1)
    countries_probs_normal = [0.12, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06, 0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.05]
    # Ensure sum is 1
    countries_probs_normal = np.array(countries_probs_normal) / sum(countries_probs_normal)
    
    # Risk scores for each country
    countries_risk = {
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
        'transaction_type': [''] * n_transactions,
        'merchant_category': [''] * n_transactions,
        'country': [''] * n_transactions,
        'device_risk_score': np.zeros(n_transactions),
        'ip_risk_score': np.zeros(n_transactions),
        'is_fraud': np.zeros(n_transactions, dtype=int)
    }
    
    # Merchant categories
    merchant_categories = ['retail', 'restaurant', 'travel', 'electronics', 
                          'entertainment', 'grocery', 'gas', 'healthcare']
    
    # Generate normal transactions
    for i in range(n_normal):
        # Normal transaction patterns
        amount = np.random.exponential(100) + 20
        amount = min(amount, 5000)
        
        transaction_type = np.random.choice(transaction_types, p=[0.4, 0.2, 0.1, 0.2, 0.1])
        
        # Select country with proper probability
        country_idx = np.random.choice(len(countries_list), p=countries_probs_normal)
        country = countries_list[country_idx]
        
        # Risk scores based on country
        device_risk = np.random.normal(countries_risk[country] * 0.3, 0.1)
        device_risk = np.clip(device_risk, 0, 1)
        
        ip_risk = np.random.normal(countries_risk[country] * 0.4, 0.15)
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
    high_risk_countries = ['RU', 'NG', 'BR', 'PK', 'UA', 'CN']
    high_risk_probs = [0.3, 0.25, 0.15, 0.15, 0.1, 0.05]
    
    for i in range(n_normal, n_transactions):
        # Fraud patterns: unusual amounts, risky countries, high risk scores
        amount = np.random.choice([
            np.random.exponential(500) + 500,  # Large amounts
            np.random.exponential(50) + 5,      # Very small amounts
        ], p=[0.7, 0.3])
        amount = min(amount, 10000)
        
        # Fraud tends to use specific transaction types
        transaction_type = np.random.choice(transaction_types, p=[0.2, 0.4, 0.3, 0.05, 0.05])
        
        # Fraud often from high-risk countries
        country = np.random.choice(high_risk_countries, p=high_risk_probs)
        
        # Higher risk scores for fraud
        device_risk = np.random.normal(0.7, 0.2)
        device_risk = np.clip(device_risk, 0.3, 1)
        
        ip_risk = np.random.normal(0.8, 0.15)
        ip_risk = np.clip(ip_risk, 0.4, 1)
        
        # Fraud can target any merchant, including high-risk ones
        fraud_merchants = merchant_categories + ['gambling', 'adult']
        merchant_category = np.random.choice(fraud_merchants)
        
        data['amount'][i] = amount
        data['transaction_type'][i] = transaction_type
        data['merchant_category'][i] = merchant_category
        data['country'][i] = country
        data['device_risk_score'][i] = device_risk
        data['ip_risk_score'][i] = ip_risk
        data['is_fraud'][i] = 1
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def ensure_data_directory():
    """Ensure data directory exists"""
    os.makedirs('data/synthetic', exist_ok=True)

if __name__ == "__main__":
    print("Generating synthetic fraud dataset...")
    ensure_data_directory()
    
    # Generate different sizes
    datasets = [
        (50000, 'synthetic_fraud_small.csv'),
        (100000, 'synthetic_fraud_medium.csv'),
        (200000, 'synthetic_fraud_large.csv')
    ]
    
    for size, filename in datasets:
        print(f"Generating {filename} with {size:,} transactions...")
        df = generate_synthetic_fraud_dataset(size)
        filepath = f'data/synthetic/{filename}'
        df.to_csv(filepath, index=False)
        fraud_count = df['is_fraud'].sum()
        fraud_pct = (fraud_count / len(df)) * 100
        print(f"  - Generated {len(df):,} transactions")
        print(f"  - Fraud cases: {fraud_count:,} ({fraud_pct:.2f}%)")
        print(f"  - Saved to: {filepath}")
        print()
    
    print("="*50)
    print("Dataset generation complete!")
    print("="*50)
    
    # Display sample statistics
    df_sample = pd.read_csv('data/synthetic/synthetic_fraud_medium.csv')
    print(f"\nSample dataset statistics (medium):")
    print(f"  Total transactions: {len(df_sample):,}")
    print(f"  Fraud ratio: {df_sample['is_fraud'].mean()*100:.2f}%")
    print(f"  Features: {list(df_sample.columns)}")
    print(f"\nSample transaction:")
    print(df_sample.head(1).T)