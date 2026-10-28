import os
import joblib
import json
import torch
import numpy as np

def explain_prediction(text: str, model_name: str = "Logistic Regression"):
    """
    Explainable AI engine:
    Computes per-word importance weights and produces HTML badge highlights.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    tokens = text.split()
    if not tokens:
        return {"tokens": [], "html": "<p>Empty text</p>", "top_spam_words": [], "top_ham_words": []}
        
    weights = {}
    try:
        tfidf = joblib.load(os.path.join(base_dir, "tfidf_vectorizer.joblib"))
        lr = joblib.load(os.path.join(base_dir, "logistic_regression_model.joblib"))
        feature_names = tfidf.get_feature_names_out()
        coefs = lr.coef_[0]
        feat_to_idx = {feat: idx for idx, feat in enumerate(feature_names)}
        
        for w in tokens:
            w_clean = w.lower().strip(".,!?:;'\"()[]{}")
            if w_clean in feat_to_idx:
                idx = feat_to_idx[w_clean]
                weights[w] = float(coefs[idx])
            else:
                weights[w] = 0.0
    except Exception as e:
        # Fallback to lexical heuristic if model artifacts not yet loaded
        spam_lexicon = {"free": 3.2, "prize": 3.0, "winner": 3.5, "claim": 2.8, "cash": 2.5, "urgent": 2.4, "won": 2.2, "call": 1.5, "txt": 1.4}
        ham_lexicon = {"home": -1.5, "later": -1.4, "sorry": -1.8, "come": -1.2, "ok": -1.1, "hi": -0.8, "love": -1.3}
        for w in tokens:
            w_clean = w.lower().strip(".,!?:;'\"()[]{}")
            if w_clean in spam_lexicon:
                weights[w] = spam_lexicon[w_clean]
            elif w_clean in ham_lexicon:
                weights[w] = ham_lexicon[w_clean]
            else:
                weights[w] = 0.0

    # Build HTML spans
    html_spans = []
    token_attributions = []
    for w in tokens:
        score = weights.get(w, 0.0)
        token_attributions.append({"token": w, "weight": round(score, 3)})
        if score > 0.5:
            # Strong Spam
            alpha = min(0.85, 0.2 + abs(score) * 0.15)
            span = f'<span style="background-color: rgba(239, 68, 68, {alpha}); color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; margin: 1px;" title="Spam weight: +{score:.2f}">{w}</span>'
        elif score < -0.5:
            # Strong Ham
            alpha = min(0.85, 0.2 + abs(score) * 0.15)
            span = f'<span style="background-color: rgba(34, 197, 94, {alpha}); color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; margin: 1px;" title="Ham weight: {score:.2f}">{w}</span>'
        else:
            span = f'<span style="padding: 2px 4px; margin: 1px;">{w}</span>'
        html_spans.append(span)
        
    html_markup = f'<div style="line-height: 2.2; font-size: 1.1rem; padding: 12px; background: #1e293b; border-radius: 8px; border: 1px solid #334155;">{" ".join(html_spans)}</div>'
    
    # Sort top spam and ham terms
    sorted_tokens = sorted(token_attributions, key=lambda x: x["weight"], reverse=True)
    top_spam = [t for t in sorted_tokens if t["weight"] > 0][:5]
    top_ham = [t for t in sorted_tokens if t["weight"] < 0][-5:]
    
    return {
        "tokens": token_attributions,
        "html": html_markup,
        "top_spam_words": top_spam,
        "top_ham_words": top_ham
    }
