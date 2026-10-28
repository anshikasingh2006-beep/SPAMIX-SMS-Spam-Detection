# 🛡️ SPAMIX: SMS Spam Detection Using Deep Learning — Interactive Analytics Dashboard

A complete, production-grade B.Tech Final Year / 7th Semester Data Science and Deep Learning Project.

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python Version](https://img.shields.io/badge/Python-3.10-blue)
![Framework](https://img.shields.io/badge/Deep%20Learning-PyTorch-orange)
![Dashboard](https://img.shields.io/badge/Web%20App-Streamlit-red)

---

## 🌟 Key Highlights
- **Authentic Benchmark Dataset:** 5,574 UCI SMS messages + documented simulated messy dataset for cleaning pipeline demonstration.
- **4 Benchmark Architectures:**
  1. Multinomial Naive Bayes (TF-IDF N-grams)
  2. Logistic Regression (High Precision Baseline)
  3. LSTM with Temporal Max-Pooling (PyTorch)
  4. Bidirectional LSTM with Temporal Max-Pooling (PyTorch)
- **High Performance:** Up to **98.06% Accuracy**, **98.59% Precision**, **93.75% Recall**, and **0.9966 ROC-AUC**.
- **Interactive Streamlit Web Dashboard:**
  - 📊 **Dataset Explorer & EDA:** Class distributions, message lengths, word frequencies, correlation heatmaps.
  - ⚡ **Live Prediction & XAI:** Real-time SMS classifier, confidence gauge, word-level attribution highlights, rule-based alerts.
  - 🔬 **Model Evaluation:** ROC/PR curves, confusion matrices, error analysis, threshold optimizer.
  - 🛡️ **Robustness Sandbox:** Stress-test models against leetspeak, spacing evasions, typos, and URL masking.
  - 📜 **Prediction History:** SQLite audit log, CSV export, and academic report downloader.
- **5 Complete Jupyter Notebooks** for classroom presentations and lab submissions.
- **Full Academic Project Report & 25 Viva Q&A Guide.**

---

## 📁 Project Directory Structure
```
SMS_Spam_Detection_Project/
├── data/
│   ├── raw/                 # Authentic UCI SMS Spam Collection (5,574 messages)
│   ├── simulated_raw/       # Controlled messy dataset (duplicates, missing labels, casing issues)
│   └── processed/           # Cleaned dataset & stratified 70/15/15 splits (train, val, test)
├── notebooks/               # 5 Full Jupyter Notebooks
│   ├── 01_data_understanding_and_cleaning.ipynb
│   ├── 02_eda_and_feature_engineering.ipynb
│   ├── 03_baseline_models_nb_lr.ipynb
│   ├── 04_deep_learning_lstm_bilstm.ipynb
│   └── 05_model_evaluation_and_xai.ipynb
├── src/                     # Modular Python pipeline
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── xai_engine.py
│   ├── robustness.py
│   └── database.py
├── models/                  # Serialized models and metrics
│   ├── tfidf_vectorizer.joblib
│   ├── naive_bayes_model.joblib
│   ├── logistic_regression_model.joblib
│   ├── lstm_model.pt
│   ├── bilstm_model.pt
│   ├── vocab.json
│   ├── model_metrics.json
│   └── test_predictions.csv
├── dashboard/               # Streamlit Multi-Page Web App
│   ├── app.py               # Main Landing Portal & Architecture Overview
│   └── pages/
│       ├── 1_📊_Dataset_Explorer.py
│       ├── 2_⚡_Live_Prediction.py
│       ├── 3_🔬_Model_Evaluation.py
│       ├── 4_🛡️_Robustness_Testing.py
│       └── 5_📜_Prediction_History.py
├── reports/
│   └── Project_Report.md    # Complete Academic B.Tech Report with 25 Viva Voce Q&As
├── database/                # SQLite DB (predictions.db)
├── run_dashboard.bat        # 1-Click Windows Dashboard Launcher
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

---

## 🚀 How to Run the Dashboard

### Option 1: 1-Click Launch (Recommended for Windows)
Simply double-click `run_dashboard.bat` inside `C:\Users\hp\Desktop\SMS_Spam_Detection_Project\`.

### Option 2: Terminal / Command Prompt
```bash
cd "C:\Users\hp\Desktop\SMS_Spam_Detection_Project"
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Summary of Experimental Results
| Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 98.06% | 96.55% | 87.50% | 0.9180 | 0.9907 | < 0.001 ms |
| **Logistic Regression** | 96.51% | **98.59%** | 72.92% | 0.8383 | **0.9966** | 0.001 ms |
| **LSTM (PyTorch)** | 97.42% | 86.54% | **93.75%** | 0.9000 | 0.9697 | 0.093 ms |
| **Bidirectional LSTM** | **97.42%** | 90.43% | 88.54% | **0.8947** | 0.9873 | 0.176 ms |
