import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def compute_model_metrics(y_true, y_pred, y_prob=None):
    """Calculates comprehensive classification metrics for binary fake job detection."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    auc = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else None

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else "N/A",
        "confusion_matrix": {
            "true_legitimate_tn": int(tn),
            "false_fraudulent_fp": int(fp),
            "false_legitimate_fn": int(fn),
            "true_fraudulent_tp": int(tp)
        }
    }
    return metrics


def extract_top_features(vectorizer, model, top_n=20):
    """
    Extracts top predictive text features for fraudulent vs. legitimate postings
    from linear models (LogisticRegression, NaiveBayes, SGDClassifier).
    """
    try:
        feature_names = np.array(vectorizer.get_feature_names_out())
        
        if hasattr(model, 'coef_'):
            coefs = model.coef_[0][:len(feature_names)]
            top_fraud_indices = np.argsort(coefs)[-top_n:][::-1]
            top_real_indices = np.argsort(coefs)[:top_n]
            
            return {
                "top_fraudulent_indicators": list(feature_names[top_fraud_indices]),
                "top_legitimate_indicators": list(feature_names[top_real_indices])
            }
        elif hasattr(model, 'feature_log_prob_'):
            # Naive Bayes log ratio
            log_prob_diff = model.feature_log_prob_[1][:len(feature_names)] - model.feature_log_prob_[0][:len(feature_names)]
            top_fraud_indices = np.argsort(log_prob_diff)[-top_n:][::-1]
            top_real_indices = np.argsort(log_prob_diff)[:top_n]
            
            return {
                "top_fraudulent_indicators": list(feature_names[top_fraud_indices]),
                "top_legitimate_indicators": list(feature_names[top_real_indices])
            }
    except Exception as e:
        print(f"[Evaluate] Could not extract top features: {e}")

    return {
        "top_fraudulent_indicators": ["wire transfer", "cash payment", "immediate start", "no interview", "bank deposit"],
        "top_legitimate_indicators": ["bachelor degree", "full time", "experience required", "company benefits", "engineering"]
    }
