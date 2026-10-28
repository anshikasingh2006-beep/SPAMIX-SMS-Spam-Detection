import streamlit as st
st.set_page_config(page_title="SPAMIX | Prediction History", page_icon="📜", layout="wide")
st.sidebar.markdown('### 🛡️ SPAMIX AI\n*Deep Learning SMS Classifier*')
st.sidebar.markdown('---')
import os
import pandas as pd
import sqlite3
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
from src.database import get_prediction_history, clear_prediction_history



st.title("📜 SPAMIX — Prediction History & Audit Logging")
st.markdown("Inspect all real-time inference requests recorded in the SQLite database (`predictions.db`), filter review feedback, and export academic reports.")

df_hist = get_prediction_history(1000)

if df_hist.empty:
    st.info("ℹ️ No predictions logged yet. Head over to the **Live Prediction** page to classify messages, and they will automatically appear here.")
else:
    # Top KPI summary
    total_logged = len(df_hist)
    spam_logged = (df_hist["predicted_label"] == "SPAM").sum()
    ham_logged = (df_hist["predicted_label"] == "HAM").sum()
    reviewed_count = (df_hist["review_feedback"] != "Unreviewed").sum()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Queries Logged", total_logged)
    c2.metric("Spam Predictions", spam_logged)
    c3.metric("Ham Predictions", ham_logged)
    c4.metric("Human-Reviewed Records", reviewed_count)
    
    st.markdown("---")
    
    # Filter Controls
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_label = st.multiselect("Filter by Predicted Label:", ["SPAM", "HAM"], default=["SPAM", "HAM"])
    with col_f2:
        filter_model = st.multiselect("Filter by Model:", df_hist["model_used"].unique().tolist(), default=df_hist["model_used"].unique().tolist())
        
    filtered_df = df_hist[
        df_hist["predicted_label"].isin(filter_label) &
        df_hist["model_used"].isin(filter_model)
    ]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download CSV
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Inference Logs as CSV",
        data=csv_data,
        file_name="sms_prediction_history.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Report Export
    st.subheader("📑 Academic Project Summary Report")
    st.markdown("Generate a formatted evaluation report for your B.Tech thesis documentation or viva submission.")
    
    report_text = f"""# B.Tech Project Evaluation Report: SMS Spam Detection Using Deep Learning
**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Project:** SMS Spam Detection Using Deep Learning & Interactive Analytics Dashboard  
**Total Dataset Records:** 5,160  
**Inference Queries Logged in Session:** {total_logged}  
**Spam Flagged:** {spam_logged} | **Ham Flagged:** {ham_logged}  

## 1. Summary of Machine Learning & Deep Learning Performance
- **Multinomial Naive Bayes:** Accuracy 98.06% | F1-Score 0.9180 | ROC-AUC 0.9907
- **Logistic Regression:** Accuracy 96.51% | Precision 98.59% | F1-Score 0.8383 | ROC-AUC 0.9966
- **LSTM (PyTorch):** Accuracy 97.42% | Recall 93.75% | F1-Score 0.9000 | ROC-AUC 0.9697
- **Bidirectional LSTM (PyTorch):** Accuracy 97.42% | Precision 90.43% | Recall 88.54% | F1-Score 0.8947 | ROC-AUC 0.9873

## 2. Methodology & Academic Findings
1. Class imbalance (87.5% ham vs 12.5% spam) requires precision-recall optimization and weighted loss functions.
2. Sequence modeling via Bidirectional LSTM captures long-range dependencies and subtle grammatical fraud patterns.
3. Traditional TF-IDF linear baselines excel in throughput and near-zero false alarms, making them ideal first-tier filters.
4. Explainable AI (XAI) feature attribution provides transparent visibility into word influence, preventing algorithmic black-box concerns.
"""
    st.download_button(
        label="📄 Download Academic Summary Report (Markdown)",
        data=report_text.encode("utf-8"),
        file_name="SMS_Spam_Detection_Project_Report.md",
        mime="text/markdown"
    )

    with st.expander("⚠️ Database Maintenance"):
        if st.button("🗑️ Clear Prediction History Database"):
            clear_prediction_history()
            st.warning("Database cleared. Refresh the page to update.")
