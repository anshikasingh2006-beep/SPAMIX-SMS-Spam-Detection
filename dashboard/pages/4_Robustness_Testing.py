import streamlit as st
st.set_page_config(page_title="SPAMIX | Robustness Testing", page_icon="🛡️", layout="wide")
st.sidebar.markdown('### 🛡️ SPAMIX AI\n*Deep Learning SMS Classifier*')
st.sidebar.markdown('---')
import os
import joblib
import json
import torch
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
from src.robustness import apply_all_perturbations, perturb_leetspeak, perturb_spacing, perturb_typos, perturb_obfuscation
from src.train_models import LSTMClassifier, BiLSTMClassifier



st.title("🛡️ SPAMIX — Adversarial Robustness & Stress Testing")
st.markdown("Test how well machine learning and deep learning models resist common spammer evasion strategies (leetspeak, deliberate character spacing, typos, and URL masking).")

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
models_dir = os.path.join(base_dir, "models")

@st.cache_resource
def load_models_for_robustness():
    tfidf = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    nb = joblib.load(os.path.join(models_dir, "naive_bayes_model.joblib"))
    lr = joblib.load(os.path.join(models_dir, "logistic_regression_model.joblib"))
    
    with open(os.path.join(models_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    lstm = LSTMClassifier(vocab_size=len(vocab), embed_dim=64, hidden_dim=64)
    lstm.load_state_dict(torch.load(os.path.join(models_dir, "lstm_model.pt"), map_location="cpu"))
    lstm.eval()
    
    bilstm = BiLSTMClassifier(vocab_size=len(vocab), embed_dim=64, hidden_dim=64)
    bilstm.load_state_dict(torch.load(os.path.join(models_dir, "bilstm_model.pt"), map_location="cpu"))
    bilstm.eval()
    
    return tfidf, nb, lr, lstm, bilstm, vocab

tfidf, nb, lr, lstm, bilstm, vocab = load_models_for_robustness()

def score_text(text: str):
    if not text.strip():
        return {"NB": 0.0, "LR": 0.0, "LSTM": 0.0, "BiLSTM": 0.0}
    
    feat = tfidf.transform([text])
    nb_p = float(nb.predict_proba(feat)[0, 1])
    lr_p = float(lr.predict_proba(feat)[0, 1])
    
    tokens = [t.lower().strip(".,!?:;'\"()[]{}") for t in text.split() if t.strip(".,!?:;'\"()[]{}")]
    seq = [vocab.get(t, vocab.get("<UNK>", 1)) for t in tokens][:100]
    if len(seq) < 100:
        seq += [vocab.get("<PAD>", 0)] * (100 - len(seq))
    inp = torch.tensor([seq], dtype=torch.long)
    
    with torch.no_grad():
        lstm_p = float(torch.sigmoid(lstm(inp).squeeze()).item())
        bilstm_p = float(torch.sigmoid(bilstm(inp).squeeze()).item())
        
    return {
        "Multinomial Naive Bayes": round(nb_p, 4),
        "Logistic Regression": round(lr_p, 4),
        "LSTM": round(lstm_p, 4),
        "Bidirectional LSTM": round(bilstm_p, 4)
    }

st.subheader("🧪 Live Adversarial Sandbox")
default_spam = "WINNER! You have won a £1,000 cash prize or a free cruise holiday! Claim immediately at http://prize-claim.com or call now."
test_sms = st.text_area("Input Base Spam Message to Perturb:", default_spam, height=90)

if st.button("⚡ Generate Perturbations & Stress-Test All Models", type="primary"):
    variants = apply_all_perturbations(test_sms)
    
    results = []
    for var_type, var_text in variants.items():
        scores = score_text(var_text)
        results.append({
            "Perturbation Type": var_type,
            "Perturbed Message": var_text,
            "Naive Bayes Spam Prob": f"{scores['Multinomial Naive Bayes']*100:.1f}%",
            "Logistic Reg Spam Prob": f"{scores['Logistic Regression']*100:.1f}%",
            "LSTM Spam Prob": f"{scores['LSTM']*100:.1f}%",
            "BiLSTM Spam Prob": f"{scores['Bidirectional LSTM']*100:.1f}%"
        })
        
    df_res = pd.DataFrame(results)
    st.dataframe(df_res, use_container_width=True)
    
    # Visual comparison bar chart
    st.subheader("📊 Robustness Comparison Chart (Spam Detection Probability)")
    chart_data = []
    for var_type, var_text in variants.items():
        scores = score_text(var_text)
        for m_name, score in scores.items():
            chart_data.append({
                "Perturbation": var_type,
                "Model": m_name,
                "Spam Probability (%)": score * 100
            })
            
    df_chart = pd.DataFrame(chart_data)
    fig = px.bar(
        df_chart,
        x="Perturbation",
        y="Spam Probability (%)",
        color="Model",
        barmode="group",
        title="Spam Probability Retention Under Adversarial Perturbations (Higher is More Robust)"
    )
    fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Classification Threshold (50%)")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### 🎓 Viva & Project Discussion Takeaways:
    1. **Dictionary & N-Gram Fragility:** Traditional Bag-of-Words and TF-IDF models suffer significantly when spaces (`c l a i m`) or leetspeak substitutions (`w1nn3r`) break tokens into out-of-vocabulary elements.
    2. **Deep Sequence Robustness:** Deep neural models with embeddings and pooling learn sub-word patterns and contextual structures, retaining higher detection confidence even when surface text is perturbed.
    3. **Real-World Recommendation:** A robust production spam filter must integrate both learned deep neural representations and heuristic rule-based pattern engines (e.g. regex normalization) to defend against adversarial evasion.
    """)
