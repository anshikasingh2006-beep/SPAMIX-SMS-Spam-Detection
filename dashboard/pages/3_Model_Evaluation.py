import streamlit as st
st.set_page_config(page_title="SPAMIX | Model Evaluation", page_icon="🔬", layout="wide")
st.sidebar.markdown('### 🛡️ SPAMIX AI\n*Deep Learning SMS Classifier*')
st.sidebar.markdown('---')
import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
from src.evaluate import load_metrics, load_test_predictions, get_curve_data, get_threshold_analysis, get_error_cases



st.title("🔬 SPAMIX — Model Evaluation & Performance Lab")
st.markdown("Evaluate all 4 architectures on the held-out 15% stratified test partition (774 messages). Explore ROC/PR curves, confusion matrices, error cases, and threshold tuning.")

metrics = load_metrics()
test_preds = load_test_predictions()

if not metrics or test_preds.empty:
    st.error("Model metrics or test predictions not found. Please verify that train_models.py has completed.")
    st.stop()

# 1. Performance Comparison Table
st.subheader("📊 Comparative Evaluation Leaderboard")

table_rows = []
for model_name, m in metrics.items():
    table_rows.append({
        "Model Architecture": model_name,
        "Accuracy (%)": f"{m['accuracy']*100:.2f}%",
        "Precision (%)": f"{m['precision']*100:.2f}%",
        "Recall (%)": f"{m['recall']*100:.2f}%",
        "F1-Score": f"{m['f1_score']:.4f}",
        "PR-AUC": f"{m['pr_auc']:.4f}",
        "ROC-AUC": f"{m['roc_auc']:.4f}",
        "Latency (ms/sample)": f"{m['latency_ms']:.3f} ms"
    })

df_table = pd.DataFrame(table_rows)
st.dataframe(df_table, use_container_width=True)

st.markdown("""
> 💡 **Academic Takeaway for Viva:** 
> - **Naive Bayes & Logistic Regression** serve as extremely fast, high-precision baselines with negligible latency. Logistic Regression achieves exceptionally low False Positives.
> - **LSTM & Bidirectional LSTM** effectively capture long-range semantic dependencies and contextual sequence patterns, delivering superior Recall on evasive spam phrases without requiring manual feature engineering.
""")

st.markdown("---")

# Tabbed Deep-Dive
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 ROC & PR Curves",
    "🔲 Confusion Matrices",
    "🎚️ Threshold Optimization Sweep",
    "🕵️ Error Analysis (False Positives & Negatives)"
])

# Tab 1: ROC & PR Curves
with tab1:
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.subheader("ROC Curves Comparison")
        fig_roc = go.Figure()
        colors = {"Multinomial Naive Bayes": "#3b82f6", "Logistic Regression": "#10b981", "LSTM": "#f59e0b", "Bidirectional LSTM": "#8b5cf6"}
        
        for m_name in metrics.keys():
            data = get_curve_data(m_name)
            if data:
                fig_roc.add_trace(go.Scatter(
                    x=data["fpr"],
                    y=data["tpr"],
                    mode='lines',
                    name=f"{m_name} (AUC = {data['roc_auc']:.4f})",
                    line=dict(color=colors.get(m_name, "#ffffff"), width=2)
                ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name="Chance"))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Recall)", height=450)
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_c2:
        st.subheader("Precision-Recall Curves Comparison")
        fig_pr = go.Figure()
        for m_name in metrics.keys():
            data = get_curve_data(m_name)
            if data:
                fig_pr.add_trace(go.Scatter(
                    x=data["recall"],
                    y=data["precision"],
                    mode='lines',
                    name=f"{m_name} (PR-AUC = {data['pr_auc']:.4f})",
                    line=dict(color=colors.get(m_name, "#ffffff"), width=2)
                ))
        fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=450)
        st.plotly_chart(fig_pr, use_container_width=True)

# Tab 2: Confusion Matrices
with tab2:
    st.subheader("Confusion Matrices (Test Set: N = 774)")
    cols = st.columns(4)
    for idx, (m_name, m) in enumerate(metrics.items()):
        cm = m["confusion_matrix"]
        matrix_vals = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        with cols[idx]:
            st.markdown(f"**{m_name}**")
            fig_cm = px.imshow(
                matrix_vals,
                text_auto=True,
                x=["Pred Ham", "Pred Spam"],
                y=["Actual Ham", "Actual Spam"],
                color_continuous_scale="Blues",
                aspect="auto"
            )
            fig_cm.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig_cm, use_container_width=True)
            st.caption(f"FN (Missed Spam): **{cm['fn']}** | FP (False Alarms): **{cm['fp']}**")

# Tab 3: Threshold Sweep
with tab3:
    st.subheader("Interactive Threshold Sweep Analyzer")
    selected_sweep_model = st.selectbox(
        "Select Model for Threshold Analysis:",
        list(metrics.keys()),
        index=3
    )
    sweep_data = get_threshold_analysis(selected_sweep_model)
    if sweep_data:
        df_sw = pd.DataFrame(sweep_data)
        
        fig_sw = go.Figure()
        fig_sw.add_trace(go.Scatter(x=df_sw["threshold"], y=df_sw["precision"], mode="lines+markers", name="Precision", line=dict(color="#10b981", width=2)))
        fig_sw.add_trace(go.Scatter(x=df_sw["threshold"], y=df_sw["recall"], mode="lines+markers", name="Recall", line=dict(color="#ef4444", width=2)))
        fig_sw.add_trace(go.Scatter(x=df_sw["threshold"], y=df_sw["f1_score"], mode="lines+markers", name="F1-Score", line=dict(color="#3b82f6", width=2)))
        
        fig_sw.update_layout(
            title=f"Precision, Recall & F1 Trade-off vs Decision Threshold ({selected_sweep_model})",
            xaxis_title="Classification Threshold",
            yaxis_title="Metric Value",
            height=420
        )
        st.plotly_chart(fig_sw, use_container_width=True)
        
        st.dataframe(df_sw, use_container_width=True)

# Tab 4: Error Cases
with tab4:
    st.subheader("Error Case Inspection (Where Models Failed)")
    selected_err_model = st.selectbox("Select Model for Error Case Inspection:", list(metrics.keys()), index=3)
    err_th = st.slider("Inspect at Threshold:", 0.1, 0.9, 0.5, 0.05)
    
    fps, fns = get_error_cases(selected_err_model, threshold=err_th)
    
    c_fp, c_fn = st.columns(2)
    with c_fp:
        st.markdown(f"#### 🚨 False Positives ({len(fps)} instances)")
        st.caption("Legitimate messages erroneously flagged as SPAM (Blocked user communication)")
        if not fps.empty:
            st.dataframe(fps, use_container_width=True)
        else:
            st.success("No False Positives at this threshold! Zero legitimate messages were blocked.")
            
    with c_fn:
        st.markdown(f"#### ⚠️ False Negatives ({len(fns)} instances)")
        st.caption("Spam messages that bypassed the filter (Missed spam)")
        if not fns.empty:
            st.dataframe(fns, use_container_width=True)
        else:
            st.success("No False Negatives at this threshold! All spam messages were caught.")
