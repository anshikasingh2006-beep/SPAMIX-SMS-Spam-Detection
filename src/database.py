import sqlite3
import os
from datetime import datetime
import pandas as pd

def get_db_path():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    db_dir = os.path.join(base_dir, "database")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "predictions.db")

def init_db():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            message TEXT NOT NULL,
            model_used TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            threshold REAL NOT NULL,
            review_feedback TEXT DEFAULT 'Unreviewed',
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(message: str, model_used: str, predicted_label: str, confidence: float, threshold: float = 0.5, review_feedback: str = "Unreviewed", notes: str = ""):
    init_db()
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO predictions (timestamp, message, model_used, predicted_label, confidence, threshold, review_feedback, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (ts, message, model_used, predicted_label, round(float(confidence), 4), round(float(threshold), 2), review_feedback, notes))
    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last_id

def update_feedback(pred_id: int, feedback: str):
    init_db()
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE predictions SET review_feedback = ? WHERE id = ?", (feedback, pred_id))
    conn.commit()
    conn.close()

def get_prediction_history(limit: int = 500) -> pd.DataFrame:
    init_db()
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM predictions ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return df

def clear_prediction_history():
    init_db()
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
