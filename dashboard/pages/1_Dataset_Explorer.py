import streamlit as st
st.set_page_config(page_title="SPAMIX | Dataset Explorer", page_icon="📊", layout="wide")
st.sidebar.markdown('### 🛡️ SPAMIX AI\n*Deep Learning SMS Classifier*')
st.sidebar.markdown('---')
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
from src.feature_engineering import add_feature_columns



st.title("📊 SPAMIX — Dataset Explorer & EDA")
st.markdown("Inspect raw data quality, explore the simulated messy dataset, and analyze statistical text patterns.")

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
raw_path = os.path.join(base_dir, "data", "raw", "sms_spam_raw.csv")
messy_path = os.path.join(base_dir, "data", "simulated_raw", "sms_spam_messy_demo.csv")
clean_path = os.path.join(base_dir, "data", "processed", "sms_spam_clean.csv")

# Dataset Selection
data_option = st.sidebar.radio(
    "Select Dataset View:",
    ["Cleaned UCI Dataset (Processed)", "Simulated Messy Dataset (Raw Quality Demo)", "Upload Custom CSV"]
)

df = None
if data_option == "Cleaned UCI Dataset (Processed)":
    if os.path.exists(clean_path):
        df = pd.read_csv(clean_path)
        st.success(f"Loaded Cleaned Dataset: **{len(df):,} messages** (Deduplicated, Normalized, and Verified).")
elif data_option == "Simulated Messy Dataset (Raw Quality Demo)":
    if os.path.exists(messy_path):
        df = pd.read_csv(messy_path)
        st.warning(f"Loaded Simulated Messy Dataset: **{len(df):,} rows** containing intentional anomalies to demonstrate data cleaning pipelines.")
        with st.expander("🔍 Inspect Simulated Defects & Audit Report", expanded=True):
            st.markdown("""
            - **Duplicate Messages Added:** 60 records
            - **Missing Messages (NaN):** 15 rows
            - **Missing/Corrupted Labels:** 20 rows (`UNKNOWN`, `Sp@m`, `123`)
            - **Label Casing & Whitespace Inconsistencies:** 80 rows (`  SPAM `, `Ham`, etc.)
            - **HTML Entities & Formatting Noise:** 100 rows (`&lt;b&gt;URGENT&lt;/b&gt;`, `&amp; claim prize`)
            """)
            if "data_quality_flag" in df.columns:
                flag_counts = df["data_quality_flag"].value_counts().reset_index()
                flag_counts.columns = ["Data Quality Flag", "Row Count"]
                st.dataframe(flag_counts, use_container_width=True)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.info(f"Custom file loaded: {len(df):,} rows")

if df is not None and not df.empty:
    # Ensure engineered features exist
    if "char_count" not in df.columns and "message" in df.columns:
        df = add_feature_columns(df)
        
    # Quick Metrics
    c1, c2, c3, c4 = st.columns(4)
    total_r = len(df)
    ham_r = (df["label"].str.lower() == "ham").sum() if "label" in df.columns else 0
    spam_r = (df["label"].str.lower() == "spam").sum() if "label" in df.columns else 0
    avg_len = int(df["char_count"].mean()) if "char_count" in df.columns else 0
    
    c1.metric("Total Records", f"{total_r:,}")
    c2.metric("Ham Count", f"{ham_r:,}")
    c3.metric("Spam Count", f"{spam_r:,}")
    c4.metric("Avg Character Length", f"{avg_len} chars")
    
    st.markdown("---")
    
    # Visualizations Row 1
    v1, v2 = st.columns(2)
    with v1:
        st.subheader("1. Class Distribution (Class Imbalance)")
        if "label" in df.columns:
            counts = df["label"].str.lower().value_counts()
            fig = px.pie(
                values=counts.values,
                names=counts.index,
                color=counts.index,
                color_discrete_map={"ham": "#22c55e", "spam": "#ef4444"},
                hole=0.45,
                title="Proportion of Ham vs Spam Messages"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label+value')
            st.plotly_chart(fig, use_container_width=True)
            
    with v2:
        st.subheader("2. Message Length Distribution (Chars)")
        if "char_count" in df.columns and "label" in df.columns:
            clean_sub = df[df["label"].str.lower().isin(["ham", "spam"])].copy()
            fig = px.histogram(
                clean_sub,
                x="char_count",
                color="label",
                barmode="overlay",
                nbins=60,
                color_discrete_map={"ham": "#22c55e", "spam": "#ef4444"},
                title="Message Character Length Comparison",
                labels={"char_count": "Characters per Message"}
            )
            fig.update_layout(xaxis_range=[0, 300])
            st.plotly_chart(fig, use_container_width=True)

    # Visualizations Row 2
    v3, v4 = st.columns(2)
    with v3:
        st.subheader("3. Word Count by Class")
        if "word_count" in df.columns and "label" in df.columns:
            clean_sub = df[df["label"].str.lower().isin(["ham", "spam"])].copy()
            fig = px.box(
                clean_sub,
                x="label",
                y="word_count",
                color="label",
                color_discrete_map={"ham": "#22c55e", "spam": "#ef4444"},
                title="Word Count Spread across Classes",
                labels={"word_count": "Word Count", "label": "SMS Class"}
            )
            fig.update_layout(yaxis_range=[0, 70])
            st.plotly_chart(fig, use_container_width=True)
            
    with v4:
        st.subheader("4. Feature Correlation Matrix")
        feat_cols = ["char_count", "word_count", "digit_count", "uppercase_count", "url_count", "contains_currency", "contains_phone_number"]
        available_feats = [c for c in feat_cols if c in df.columns]
        if len(available_feats) > 1:
            corr = df[available_feats].corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="Viridis",
                title="Feature Correlation Heatmap",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Word Frequency Analysis
    st.subheader("5. Lexical Frequency: Top Spam vs Ham Terms")
    col_w1, col_w2 = st.columns(2)
    
    def get_top_words(texts, n=15):
        from collections import Counter
        stopwords = set(["i", "me", "my", "you", "your", "the", "a", "an", "to", "in", "for", "of", "and", "is", "it", "that", "on", "at", "with", "have", "are", "be", "this", "from", "or", "ur", "u", "so", "call", "now"])
        words = []
        for t in texts:
            for w in str(t).lower().split():
                w = w.strip(".,!?:;'\"()[]{}")
                if len(w) > 2 and w not in stopwords and not w.isdigit():
                    words.append(w)
        return pd.DataFrame(Counter(words).most_common(n), columns=["Word", "Frequency"])

    with col_w1:
        if "label" in df.columns and "message" in df.columns:
            spam_msgs = df[df["label"].str.lower() == "spam"]["message"]
            top_spam = get_top_words(spam_msgs, 15)
            fig = px.bar(
                top_spam,
                x="Frequency",
                y="Word",
                orientation="h",
                color="Frequency",
                color_continuous_scale="Reds",
                title="Top 15 Suspicious Keywords in Spam"
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
            
    with col_w2:
        if "label" in df.columns and "message" in df.columns:
            ham_msgs = df[df["label"].str.lower() == "ham"]["message"]
            top_ham = get_top_words(ham_msgs, 15)
            fig = px.bar(
                top_ham,
                x="Frequency",
                y="Word",
                orientation="h",
                color="Frequency",
                color_continuous_scale="Greens",
                title="Top 15 Legitimate Keywords in Ham"
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    # Interactive Table Viewer
    st.subheader("📋 Dataset Table Browser")
    search_q = st.text_input("🔍 Search message contents or ID:", "")
    view_df = df.copy()
    if search_q:
        view_df = view_df[view_df["message"].astype(str).str.contains(search_q, case=False, na=False)]
    st.dataframe(view_df.head(100), use_container_width=True)
    
    csv_bytes = view_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Filtered Dataset as CSV", csv_bytes, "sms_dataset_view.csv", "text/csv")
