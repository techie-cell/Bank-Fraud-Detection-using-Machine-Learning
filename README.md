# Bank-Fraud-Detection-using-Machine-Learning
AI-Powered Real-Time Bank Fraud Detection System using Machine Learning, Explainable AI, FastAPI, and Streamlit.
## Overview
This project is an AI-powered real-time fraud detection system developed for financial institutions to identify suspicious banking transactions with high accuracy and low latency.

The system uses Machine Learning, Explainable AI, FastAPI, and Streamlit to detect fraudulent activities while minimizing false positives.

---

# Business Problem

Traditional rule-based fraud detection systems fail to detect evolving fraud patterns and generate excessive false alerts.

This project solves the problem by:
- Detecting fraud in real time
- Handling highly imbalanced transaction data
- Providing explainable AI predictions
- Supporting scalable API deployment
- Monitoring fraud detection performance

---

# Features

- Machine Learning based detection
- XGBoost classification model
- SHAP explainability
- SMOTE for imbalance handling
- FastAPI backend
- Interactive Streamlit dashboard
- Fraud probability scoring
- Threshold optimization
- Drift monitoring support
- Docker-ready architecture

---

# Tech Stack

- Python
- Scikit-learn
- XGBoost
- SHAP
- FastAPI
- Streamlit
- Pandas
- NumPy
- Docker

---

# Datasets Used

## 1. Credit Card Fraud Detection Dataset
- Source: Kaggle
- Contains anonymized transaction data
- Highly imbalanced fraud classification dataset

## 2. Synthetic Financial Fraud Dataset
- Simulated banking fraud transactions
- Includes transaction risk features

---

# Project Structure

```bash
Bank-Fraud-Detection/
│
├── api/
├── dashboard/
├── data/
├── logs/
├── models/
├── notebooks/
├── src/
├── requirements.txt
# Workflow

## Data Collection
- Load transaction datasets
- Merge fraud-related features

## Data Preprocessing
- Missing value handling
- Encoding categorical variables
- Feature scaling

## Feature Engineering
- Transaction frequency
- Device risk score
- IP risk score
- Merchant category analysis
- Time-based features

## Handling Imbalanced Data
- SMOTE oversampling
- Cost-sensitive learning

## Model Training

Models implemented:
- Logistic Regression
- Random Forest
- XGBoost

## Explainability
- SHAP value analysis
- Feature importance visualization

## Deployment
- FastAPI REST API
- Streamlit Dashboard

---

# Model Performance

| Metric | Value |
|--------|--------|
| Recall | >80% |
| False Positive Rate | <5% |
| Inference Latency | <100ms |

---

# Dashboard Features

- Fraud prediction interface
- Transaction risk visualization
- SHAP explanation graphs
- Fraud analytics charts
Fraud analytics charts├── start_api.bat
├── start_dashboard.bat
└── README.md
