"""
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
