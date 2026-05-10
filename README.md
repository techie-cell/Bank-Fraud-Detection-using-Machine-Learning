# AI-Powered Bank Fraud Detection System

# Overview

This project is an AI-powered fraud detection system developed for financial institutions to identify suspicious banking transactions using Machine Learning and Explainable AI techniques.

The system analyzes historical transaction datasets to classify fraudulent and non-fraudulent transactions with high accuracy while minimizing false positives. The project uses Machine Learning models, SHAP explainability, FastAPI, and Streamlit for fraud prediction, analysis, and visualization.

---

# Business Problem

Traditional rule-based fraud detection systems struggle to identify evolving fraud patterns and often generate excessive false alerts.

This project addresses these challenges by:
- Detecting suspicious transaction patterns using Machine Learning
- Handling highly imbalanced fraud datasets
- Providing explainable AI predictions
- Supporting API-based fraud prediction
- Visualizing fraud analytics through dashboards

---

# Features

- Machine Learning based fraud detection
- Historical transaction analysis
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
├── start_api.bat
├── start_dashboard.bat
└── README.md
```

---

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

## Fraud Prediction System
- Batch fraud prediction
- API-based fraud scoring
- Fraud probability generation
- Dashboard-based transaction analysis

## Deployment
- FastAPI backend
- Streamlit Dashboard
- Local deployment environment

---

# Model Performance

| Metric | Value |
|--------|--------|
| Recall | >80% |
| False Positive Rate | <5% |
| Prediction Accuracy | High Accuracy |

---

# Dashboard Features

- Fraud prediction interface
- Transaction risk visualization
- SHAP explanation graphs
- Fraud analytics charts
- Prediction probability scores
- Interactive dashboard

---

# Current Limitations

- Uses historical datasets for training
- Not connected to live banking systems
- No streaming transaction pipeline
- Local deployment only

---

# API Endpoints

## Predict Fraud

```http
POST /predict
```

## Health Check

```http
GET /health
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/techie-cell/Bank-Fraud-Detection-using-Machine-Learning.git
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment (Windows)

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run API

```bash
uvicorn api.app:app --reload
```

---

# Run Dashboard

```bash
streamlit run dashboard/app_streamlit.py
```

---

# Future Enhancements

- Kafka streaming integration
- Graph Neural Networks for fraud rings
- Autoencoder anomaly detection
- Federated learning
- Online learning using River
- Cloud deployment using AWS

---

# Industry Applications

- Banking
- FinTech
- Insurance Fraud Detection
- Telecom Fraud Analytics
- Crypto Transaction Monitoring

--

Developed as part of AI/ML Internship Project at Glowlogics Solutions Pvt Ltd.
