import streamlit as st
import os
import json
import pandas as pd
import sqlite3

st.set_page_config(
    page_title="SPAMIX | SMS Spam AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.markdown("### 🛡️ SPAMIX AI\n*Deep Learning SMS Classifier*")
st.sidebar.markdown("---")

# Custom Styling

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #3b82f6;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
    }
    .kpi-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .kpi-lbl {
        font-size: 0.9rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;"><span style="background: linear-gradient(135deg, #2563eb, #7c3aed); color: white; padding: 5px 16px; border-radius: 8px; font-weight: 900; font-size: 1.6rem; letter-spacing: 0.1em; box-shadow: 0 4px 14px rgba(37,99,235,0.45);">SPAMIX</span><span class="main-title" style="margin-bottom: 0;">🛡️ SMS Spam Detection & Analytics System</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A Deep Learning & Natural Language Processing System for SMS Classification | Final Year B.Tech Project</div>', unsafe_allow_html=True)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
metrics_path = os.path.join(base_dir, "models", "model_metrics.json")
proc_data_path = os.path.join(base_dir, "data", "processed", "sms_spam_clean.csv")
db_path = os.path.join(base_dir, "database", "predictions.db")

total_msgs = 5160
spam_pct = 12.4
best_model = "Bidirectional LSTM"
best_f1 = 0.91

if os.path.exists(proc_data_path):
    try:
        df_clean = pd.read_csv(proc_data_path)
        total_msgs = len(df_clean)
        spam_pct = round(((df_clean["label"] == "spam").sum() / total_msgs) * 100, 1)
    except Exception:
        pass

if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            best_m = max(metrics.items(), key=lambda x: x[1].get("f1_score", 0))
            best_model = best_m[0]
            best_f1 = best_m[1].get("f1_score", 0)
    except Exception:
        pass

pred_count = 0
if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        pred_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.close()
    except Exception:
        pass

# KPI Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-val">{total_msgs:,}</div><div class="kpi-lbl">Corpus Messages</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-val">{spam_pct}%</div><div class="kpi-lbl">Spam Ratio</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-val">{best_f1:.3f}</div><div class="kpi-lbl">Top Model F1 ({best_model})</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-val">{pred_count}</div><div class="kpi-lbl">Inference Queries Logged</div></div>', unsafe_allow_html=True)

st.markdown("---")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("🎯 Project Purpose & Research Scope")
    st.markdown("""
    This project demonstrates an enterprise-grade AI solution for identifying unsolicited, fraudulent, and phishing SMS communications (**Spam**) from legitimate personal/transactional messages (**Ham**).
    
    **Key Research Question:**
    > *Does a deep sequence-based architecture (Bidirectional LSTM) deliver statistically superior discrimination compared to traditional TF-IDF linear baselines under strict class imbalance and adversarial evasion?*
    
    #### 🚀 Highlights of the System:
    - **Dual-Phase Data Engineering:** Features both authentic UCI benchmark records and a documented *Simulated Messy Dataset* to demonstrate end-to-end cleaning, regex normalization, and deduplication.
    - **4-Model Comparison Suite:** Multinomial Naive Bayes, Logistic Regression, LSTM, and Bidirectional LSTM evaluated under identical 70/15/15 stratified splits.
    - **Explainable AI (XAI):** Real-time per-token influence highlighting to unpack model decisions and expose suspicious lexical anchors.
    - **Adversarial Robustness Testing:** Stress-tests classifiers against leetspeak, spacing evasions, typos, and obfuscated URLs.
    - **SQLite Feedback Logging:** Tracks production inference logs and enables human-in-the-loop review.
    """)

with col_right:
    st.subheader("🏛️ Architecture Overview")
    st.markdown("""
    ```
    [ Raw UCI SMS Corpus ]
              │
    [ Data Cleaning & Normalization ]
              │
    [ Stratified 70/15/15 Split ]
         ┌────┴────────────────────────┐
    [ TF-IDF Vectorizer ]     [ PyTorch Sequence Vocab ]
         │                                  │
    ┌────┴────────────┐             ┌───────┴──────────┐
    Naive Bayes   Logistic Reg     LSTM         BiLSTM
         │             │             │             │
    └────┬─────────────┴─────────────┴─────────────┘
         │
    [ Comprehensive Evaluation: PR-AUC, F1, Latency ]
         │
    [ Streamlit Interactive Multi-Page Dashboard ]
    ```
    """)
    st.info("💡 **Navigation:** Use the sidebar on the left to navigate across the pages: **Dataset Explorer**, **Live Prediction**, **Model Evaluation**, **Robustness Testing**, and **Prediction History**.")
