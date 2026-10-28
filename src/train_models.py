import os
import json
import time
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

torch.manual_seed(42)
np.random.seed(42)

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.data = []
        for text in texts:
            tokens = [t.lower().strip(".,!?:;'\"()[]{}") for t in str(text).split()]
            tokens = [t for t in tokens if t]
            seq = [vocab.get(t, vocab.get("<UNK>", 1)) for t in tokens][:max_len]
            if len(seq) < max_len:
                seq += [vocab.get("<PAD>", 0)] * (max_len - len(seq))
            self.data.append(seq)
        self.labels = labels if labels is not None else [0] * len(texts)
        self.data = torch.tensor(self.data, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        pooled, _ = torch.max(out, dim=1)
        return self.fc(pooled)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.bilstm(emb)
        pooled, _ = torch.max(out, dim=1)
        return self.fc(pooled)

def build_vocab(texts, max_vocab=5000):
    word_counts = {}
    for text in texts:
        for word in str(text).lower().split():
            w = word.strip(".,!?:;'\"()[]{}")
            if w:
                word_counts[w] = word_counts.get(w, 0) + 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in sorted_words[:max_vocab - 2]:
        vocab[word] = len(vocab)
    return vocab

def train_neural_model(model, train_loader, val_loader, epochs=10, lr=0.003, pos_weight_val=2.5, device="cpu"):
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_b)
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = model(x_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * len(x_b)
                
        val_loss /= len(val_loader.dataset)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def compute_metrics(y_true, y_probs, threshold=0.5, latency_ms=0.0):
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(rec, prec)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "latency_ms": round(float(latency_ms), 3),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }
    }

def run_training_pipeline():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    proc_dir = os.path.join(project_dir, "data", "processed")
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    train_df = pd.read_csv(os.path.join(proc_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(proc_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(proc_dir, "test.csv"))
    
    y_train = (train_df["label"] == "spam").astype(int).values
    y_val = (val_df["label"] == "spam").astype(int).values
    y_test = (test_df["label"] == "spam").astype(int).values
    
    # 1. TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(train_df["message"])
    X_val_tfidf = tfidf.transform(val_df["message"])
    X_test_tfidf = tfidf.transform(test_df["message"])
    joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    
    # Model 1: Naive Bayes
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_tfidf, y_train)
    t_start = time.time()
    nb_test_probs = nb.predict_proba(X_test_tfidf)[:, 1]
    nb_lat = ((time.time() - t_start) / len(test_df)) * 1000
    joblib.dump(nb, os.path.join(models_dir, "naive_bayes_model.joblib"))
    nb_metrics = compute_metrics(y_test, nb_test_probs, latency_ms=nb_lat)
    print("Multinomial Naive Bayes trained:", nb_metrics)
    
    # Model 2: Logistic Regression
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=42)
    lr.fit(X_train_tfidf, y_train)
    t_start = time.time()
    lr_test_probs = lr.predict_proba(X_test_tfidf)[:, 1]
    lr_lat = ((time.time() - t_start) / len(test_df)) * 1000
    joblib.dump(lr, os.path.join(models_dir, "logistic_regression_model.joblib"))
    lr_metrics = compute_metrics(y_test, lr_test_probs, latency_ms=lr_lat)
    print("Logistic Regression trained:", lr_metrics)
    
    # 2. PyTorch Datasets
    vocab = build_vocab(train_df["message"], max_vocab=5000)
    with open(os.path.join(models_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
        
    train_dataset = TextDataset(train_df["message"].tolist(), y_train, vocab, max_len=100)
    val_dataset = TextDataset(val_df["message"].tolist(), y_val, vocab, max_len=100)
    test_dataset = TextDataset(test_df["message"].tolist(), y_test, vocab, max_len=100)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Model 3: LSTM (with temporal max-pooling)
    lstm_model = LSTMClassifier(vocab_size=len(vocab), embed_dim=64, hidden_dim=64)
    lstm_model = train_neural_model(lstm_model, train_loader, val_loader, epochs=10, lr=0.003, pos_weight_val=2.5)
    torch.save(lstm_model.state_dict(), os.path.join(models_dir, "lstm_model.pt"))
    
    lstm_model.eval()
    t_start = time.time()
    with torch.no_grad():
        lstm_test_logits = lstm_model(test_dataset.data).squeeze()
        lstm_test_probs = torch.sigmoid(lstm_test_logits).numpy()
    lstm_lat = ((time.time() - t_start) / len(test_df)) * 1000
    lstm_metrics = compute_metrics(y_test, lstm_test_probs, latency_ms=lstm_lat)
    print("LSTM trained:", lstm_metrics)
    
    # Model 4: Bidirectional LSTM
    bilstm_model = BiLSTMClassifier(vocab_size=len(vocab), embed_dim=64, hidden_dim=64)
    bilstm_model = train_neural_model(bilstm_model, train_loader, val_loader, epochs=10, lr=0.003, pos_weight_val=2.5)
    torch.save(bilstm_model.state_dict(), os.path.join(models_dir, "bilstm_model.pt"))
    
    bilstm_model.eval()
    t_start = time.time()
    with torch.no_grad():
        bilstm_test_logits = bilstm_model(test_dataset.data).squeeze()
        bilstm_test_probs = torch.sigmoid(bilstm_test_logits).numpy()
    bilstm_lat = ((time.time() - t_start) / len(test_df)) * 1000
    bilstm_metrics = compute_metrics(y_test, bilstm_test_probs, latency_ms=bilstm_lat)
    print("BiLSTM trained:", bilstm_metrics)
    
    # Save all metrics
    all_metrics = {
        "Multinomial Naive Bayes": nb_metrics,
        "Logistic Regression": lr_metrics,
        "LSTM": lstm_metrics,
        "Bidirectional LSTM": bilstm_metrics
    }
    with open(os.path.join(models_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
        
    # Save test predictions table
    test_results_df = test_df.copy()
    test_results_df["actual_label"] = test_df["label"]
    test_results_df["actual_binary"] = y_test
    test_results_df["nb_prob"] = np.round(nb_test_probs, 4)
    test_results_df["lr_prob"] = np.round(lr_test_probs, 4)
    test_results_df["lstm_prob"] = np.round(lstm_test_probs, 4)
    test_results_df["bilstm_prob"] = np.round(bilstm_test_probs, 4)
    test_results_df.to_csv(os.path.join(models_dir, "test_predictions.csv"), index=False)
    print("All models successfully trained and serialized!")

if __name__ == "__main__":
    run_training_pipeline()
