"""
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
