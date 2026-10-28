import streamlit as st
st.set_page_config(page_title="SPAMIX | Live SMS Classifier", page_icon="⚡", layout="wide")
st.sidebar.markdown('### 🛡️ SPAMIX AI\n*Deep Learning SMS Classifier*')
st.sidebar.markdown('---')
import os
import joblib
import json
import torch
import numpy as np
import plotly.graph_objects as go
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
from src.feature_engineering import get_suspicious_patterns, extract_features
from src.xai_engine import explain_prediction
from src.database import log_prediction, update_feedback
from src.train_models import LSTMClassifier, BiLSTMClassifier



st.title("⚡ SPAMIX — Live SMS Classifier with Explainable AI")
st.markdown("Analyze any custom message, inspect model confidence, tune classification thresholds, and visualize word-level attribution.")

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
models_dir = os.path.join(base_dir, "models")

# Sidebar Controls
st.sidebar.header("⚙️ Model & Threshold Settings")
model_choice = st.sidebar.selectbox(
    "Select Classifier:",
    ["Bidirectional LSTM (PyTorch)", "LSTM (PyTorch)", "Multinomial Naive Bayes", "Logistic Regression"]
)

threshold = st.sidebar.slider(
    "Spam Decision Threshold:",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Messages with spam probability ≥ this threshold will be classified as SPAM. Lower values increase recall (fewer missed spams); higher values increase precision (fewer false alarms)."
)

# Realistic Presets
st.markdown("##### 💡 Quick Test Presets:")
col_p1, col_p2, col_p3, col_p4 = st.columns(4)

preset_text = ""
if "current_sms" not in st.session_state:
    st.session_state.current_sms = "WINNER! You have won a £1,000 cash prize or a free holiday! Call 09061701461 to claim now. T&C apply."

with col_p1:
    if st.button("🎁 Prize Spam"):
        st.session_state.current_sms = "WINNER! You have won a £1,000 cash prize or a free holiday! Call 09061701461 to claim now. T&C apply."
with col_p2:
    if st.button("🚨 Phishing / Bank Alert"):
        st.session_state.current_sms = "URGENT: Your account has been suspended due to suspicious activity. Verify immediately at http://bit.ly/bank-auth to restore access."
with col_p3:
    if st.button("💬 Friendly Ham"):
        st.session_state.current_sms = "Hey Anshika, are we still meeting in the campus library around 4 PM today? Let me know!"
with col_p4:
    if st.button("📦 Order Update Ham"):
        st.session_state.current_sms = "Your package has been delivered to your front door. Thank you for shopping with us!"

input_text = st.text_area(
    "Enter SMS Message to Analyze:",
    value=st.session_state.current_sms,
    height=110
)

# Helper function for live inference
@st.cache_resource
def load_all_models():
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

try:
    tfidf, nb, lr, lstm, bilstm, vocab = load_all_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure train_models.py has completed successfully.")
    st.stop()

def predict_single(text: str, model_name: str):
    if not text.strip():
        return 0.0
    if model_name == "Multinomial Naive Bayes":
        feat = tfidf.transform([text])
        return float(nb.predict_proba(feat)[0, 1])
    elif model_name == "Logistic Regression":
        feat = tfidf.transform([text])
        return float(lr.predict_proba(feat)[0, 1])
    elif model_name in ["LSTM (PyTorch)", "Bidirectional LSTM (PyTorch)"]:
        tokens = [t.lower().strip(".,!?:;'\"()[]{}") for t in text.split() if t.strip(".,!?:;'\"()[]{}")]
        seq = [vocab.get(t, vocab.get("<UNK>", 1)) for t in tokens][:100]
        if len(seq) < 100:
            seq += [vocab.get("<PAD>", 0)] * (100 - len(seq))
        inp = torch.tensor([seq], dtype=torch.long)
        with torch.no_grad():
            if "Bidirectional" in model_name:
                logit = bilstm(inp).squeeze()
            else:
                logit = lstm(inp).squeeze()
            prob = torch.sigmoid(logit).item()
        return float(prob)
    return 0.0

if st.button("🚀 Analyze Message", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter a valid message.")
    else:
        spam_prob = predict_single(input_text, model_choice)
        is_spam = spam_prob >= threshold
        pred_label = "SPAM" if is_spam else "HAM"
        confidence_pct = (spam_prob if is_spam else (1.0 - spam_prob)) * 100
        
        # Log to SQLite
        pred_id = log_prediction(
            message=input_text,
            model_used=model_choice,
            predicted_label=pred_label,
            confidence=confidence_pct,
            threshold=threshold
        )
        st.session_state.last_pred_id = pred_id
        
        st.markdown("---")
        
        # Results Layout
        r_col1, r_col2 = st.columns([3, 2])
        
        with r_col1:
            if is_spam:
                st.error(f"### 🚨 SPAM DETECTED\n**Classification:** Unsolicited / Fraudulent SMS\n**Assigned Model:** {model_choice} (Decision Threshold: {threshold:.2f})")
            else:
                st.success(f"### 🛡️ LEGITIMATE MESSAGE (HAM)\n**Classification:** Authentic Personal / Transactional SMS\n**Assigned Model:** {model_choice} (Decision Threshold: {threshold:.2f})")
                
            # Suspicious Pattern Detection
            st.subheader("🔍 Rule-Based Pattern Checklist")
            patterns = get_suspicious_patterns(input_text)
            if patterns:
                for p in patterns:
                    st.warning(f"⚠️ **[{p['severity']} Alert] {p['type']}:** {p['detail']}")
            else:
                st.info("✅ No obvious heuristic red flags (URLs, premium phone numbers, urgent hooks) detected.")

            # Text Features Breakdown
            st.subheader("📐 Extracted Text Statistics")
            feats = extract_features(input_text)
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            f_col1.metric("Characters", feats["char_count"])
            f_col2.metric("Word Count", feats["word_count"])
            f_col3.metric("Digits", feats["digit_count"])
            f_col4.metric("Caps Ratio", f"{feats['caps_ratio']*100:.1f}%")

        with r_col2:
            st.subheader("📊 Model Confidence Meter")
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=spam_prob * 100,
                title={'text': "Estimated Spam Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ef4444" if is_spam else "#22c55e"},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(34, 197, 94, 0.2)"},
                        {'range': [30, 70], 'color': "rgba(234, 179, 8, 0.2)"},
                        {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            gauge_fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)

        # Explainable AI Section
        st.markdown("---")
        st.subheader("🧠 Explainable AI (XAI): Word & Token Attribution")
        st.markdown("""
        Understand *why* the model made this decision. 
        - **Red highlighted words** push the prediction toward **SPAM**.
        - **Green highlighted words** indicate legitimate **HAM** patterns.
        """)
        
        xai_res = explain_prediction(input_text, model_name=model_choice)
        st.markdown(xai_res["html"], unsafe_allow_html=True)
        
        # Word Attribution Bar Chart
        if xai_res["top_spam_words"] or xai_res["top_ham_words"]:
            xai_terms = xai_res["top_spam_words"] + xai_res["top_ham_words"]
            t_names = [t["token"] for t in xai_terms]
            t_weights = [t["weight"] for t in xai_terms]
            colors = ["#ef4444" if w > 0 else "#22c55e" for w in t_weights]
            
            fig_bar = go.Figure(go.Bar(
                x=t_weights,
                y=t_names,
                orientation='h',
                marker_color=colors
            ))
            fig_bar.update_layout(
                title="Token Attribution Influence Values (Positive: Spam, Negative: Ham)",
                xaxis_title="Influence Weight",
                yaxis_title="Token",
                height=260
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Human-in-the-Loop Feedback
        st.markdown("---")
        st.subheader("📝 Provide Human Feedback for Continuous Improvement")
        st.caption("Was this prediction accurate? Submit feedback to log ground truth into the SQLite database for error tracking.")
        
        fb_col1, fb_col2, fb_col3 = st.columns(3)
        with fb_col1:
            if st.button("✅ Prediction was Correct"):
                if "last_pred_id" in st.session_state:
                    update_feedback(st.session_state.last_pred_id, "Correct")
                    st.success("Feedback logged: Marked as Correct!")
        with fb_col2:
            if st.button("❌ False Positive (Actually Ham)"):
                if "last_pred_id" in st.session_state:
                    update_feedback(st.session_state.last_pred_id, "False Positive")
                    st.warning("Feedback logged: Marked as False Positive!")
        with fb_col3:
            if st.button("⚠️ False Negative (Missed Spam)"):
                if "last_pred_id" in st.session_state:
                    update_feedback(st.session_state.last_pred_id, "False Negative")
                    st.error("Feedback logged: Marked as False Negative!")
