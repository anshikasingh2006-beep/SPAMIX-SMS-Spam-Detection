import re
import html
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def clean_text(text: str) -> str:
    """
    Cleans raw SMS message text:
    - Decodes HTML entities (&amp;, &lt;, etc.)
    - Removes HTML tags
    - Normalizes extra whitespace
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_sms_dataset(df: pd.DataFrame, drop_duplicates: bool = True):
    """
    Cleans and standardizes the SMS dataset.
    Returns: (cleaned_df, audit_report_dict)
    """
    initial_rows = len(df)
    report = {
        "initial_rows": initial_rows,
        "missing_messages": 0,
        "missing_labels": 0,
        "invalid_labels_removed": 0,
        "duplicates_removed": 0,
        "empty_messages_removed": 0,
        "final_rows": 0
    }
    
    clean_df = df.copy()
    
    # 1. Missing message check
    missing_msgs = clean_df["message"].isna().sum()
    report["missing_messages"] = int(missing_msgs)
    clean_df = clean_df.dropna(subset=["message"])
    
    # 2. Missing & invalid label check
    clean_df["label"] = clean_df["label"].astype(str).str.strip().str.lower()
    invalid_labels = ~clean_df["label"].isin(["ham", "spam"])
    report["invalid_labels_removed"] = int(invalid_labels.sum())
    clean_df = clean_df[~invalid_labels]
    
    # 3. Clean message text
    clean_df["message"] = clean_df["message"].apply(clean_text)
    
    # 4. Remove empty messages
    empty_msgs = clean_df["message"].str.len() == 0
    report["empty_messages_removed"] = int(empty_msgs.sum())
    clean_df = clean_df[~empty_msgs]
    
    # 5. Deduplicate
    if drop_duplicates:
        dups = clean_df.duplicated(subset=["message"]).sum()
        report["duplicates_removed"] = int(dups)
        clean_df = clean_df.drop_duplicates(subset=["message"]).reset_index(drop=True)
    else:
        clean_df = clean_df.reset_index(drop=True)
        
    report["final_rows"] = len(clean_df)
    report["ham_count"] = int((clean_df["label"] == "ham").sum())
    report["spam_count"] = int((clean_df["label"] == "spam").sum())
    report["spam_percentage"] = round((report["spam_count"] / report["final_rows"]) * 100, 2)
    
    return clean_df, report

def process_and_save_data(raw_csv_path: str, output_dir: str):
    """
    Full pipeline to clean raw data, perform 70/15/15 stratified split,
    and save processed artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_raw = pd.read_csv(raw_csv_path)
    clean_df, report = clean_sms_dataset(df_raw, drop_duplicates=True)
    
    # Save full cleaned dataset
    clean_path = os.path.join(output_dir, "sms_spam_clean.csv")
    clean_df.to_csv(clean_path, index=False)
    
    # Stratified Train (70%), Val (15%), Test (15%) split
    # Step 1: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        clean_df,
        test_size=0.30,
        random_state=42,
        stratify=clean_df["label"]
    )
    
    # Step 2: split 30% temp into 50% val (15% total) and 50% test (15% total)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    report["train_rows"] = len(train_df)
    report["val_rows"] = len(val_df)
    report["test_rows"] = len(test_df)
    
    return report
