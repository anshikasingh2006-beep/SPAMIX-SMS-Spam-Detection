import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix

def load_metrics():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "model_metrics.json"))
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_test_predictions():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "test_predictions.csv"))
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def get_curve_data(model_name: str):
    df = load_test_predictions()
    if df.empty:
        return None
    prob_col_map = {
        "Multinomial Naive Bayes": "nb_prob",
        "Logistic Regression": "lr_prob",
        "LSTM": "lstm_prob",
        "Bidirectional LSTM": "bilstm_prob"
    }
    col = prob_col_map.get(model_name, "bilstm_prob")
    y_true = df["actual_binary"].values
    y_prob = df[col].values
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return {
        "fpr": fpr.tolist(), "tpr": tpr.tolist(), "roc_auc": round(float(auc(fpr, tpr)), 4),
        "recall": rec.tolist(), "precision": prec.tolist(), "pr_auc": round(float(auc(rec, prec)), 4)
    }

def get_threshold_analysis(model_name: str):
    df = load_test_predictions()
    if df.empty:
        return []
    prob_col_map = {
        "Multinomial Naive Bayes": "nb_prob",
        "Logistic Regression": "lr_prob",
        "LSTM": "lstm_prob",
        "Bidirectional LSTM": "bilstm_prob"
    }
    col = prob_col_map.get(model_name, "bilstm_prob")
    y_true = df["actual_binary"].values
    y_prob = df[col].values
    
    thresholds = np.linspace(0.05, 0.95, 19)
    records = []
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        records.append({
            "threshold": round(float(th), 2),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        })
    return records

def get_error_cases(model_name: str, threshold: float = 0.5):
    df = load_test_predictions()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    prob_col_map = {
        "Multinomial Naive Bayes": "nb_prob",
        "Logistic Regression": "lr_prob",
        "LSTM": "lstm_prob",
        "Bidirectional LSTM": "bilstm_prob"
    }
    col = prob_col_map.get(model_name, "bilstm_prob")
    preds = (df[col] >= threshold).astype(int)
    
    fps = df[(preds == 1) & (df["actual_binary"] == 0)][["message_id", "message", col]].copy()
    fps.rename(columns={col: "predicted_spam_prob"}, inplace=True)
    
    fns = df[(preds == 0) & (df["actual_binary"] == 1)][["message_id", "message", col]].copy()
    fns.rename(columns={col: "predicted_spam_prob"}, inplace=True)
    
    return fps, fns
