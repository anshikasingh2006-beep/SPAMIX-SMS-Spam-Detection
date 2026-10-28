import re
import pandas as pd
import numpy as np

URL_REGEX = re.compile(r"https?://\S+|www\.\S+|bit\.ly/\S+", re.IGNORECASE)
PHONE_REGEX = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{5,11}\b")
CURRENCY_REGEX = re.compile(r"[\$£€¥₹]|(?:\b(?:usd|gbp|eur|inr|rs|bucks)\b)", re.IGNORECASE)
URGENCY_REGEX = re.compile(
    r"\b(urgent|claim|winner|won|prize|free|action|call now|reply|exclusive|selected|congratulations|congrats|guaranteed|cash|bonus|award|limited|expire|expires|final)\b",
    re.IGNORECASE
)

def extract_features(text: str) -> dict:
    """
    Extracts informative structural and lexical features from an SMS message.
    """
    if pd.isna(text) or not isinstance(text, str):
        text = ""
    words = text.split()
    urls = URL_REGEX.findall(text)
    phone_numbers = PHONE_REGEX.findall(text)
    urgency_matches = URGENCY_REGEX.findall(text)
    
    char_count = len(text)
    word_count = len(words)
    avg_word_length = (sum(len(w) for w in words) / word_count) if word_count > 0 else 0.0
    digit_count = sum(c.isdigit() for c in text)
    uppercase_count = sum(c.isupper() for c in text)
    special_char_count = sum(not c.isalnum() and not c.isspace() for c in text)
    exclamation_count = text.count("!")
    question_count = text.count("?")
    
    contains_currency = 1 if bool(CURRENCY_REGEX.search(text)) else 0
    contains_phone_number = 1 if len(phone_numbers) > 0 else 0
    url_count = len(urls)
    urgency_word_count = len(urgency_matches)
    
    caps_ratio = uppercase_count / (char_count + 1e-5)
    digit_ratio = digit_count / (char_count + 1e-5)
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "avg_word_length": round(avg_word_length, 2),
        "digit_count": digit_count,
        "uppercase_count": uppercase_count,
        "special_char_count": special_char_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "url_count": url_count,
        "contains_currency": contains_currency,
        "contains_phone_number": contains_phone_number,
        "urgency_word_count": urgency_word_count,
        "caps_ratio": round(caps_ratio, 3),
        "digit_ratio": round(digit_ratio, 3)
    }

def add_feature_columns(df: pd.DataFrame, text_col: str = "message") -> pd.DataFrame:
    """
    Applies feature extraction across a DataFrame.
    """
    features_list = df[text_col].apply(extract_features).tolist()
    feat_df = pd.DataFrame(features_list, index=df.index)
    return pd.concat([df, feat_df], axis=1)

def get_suspicious_patterns(text: str) -> list:
    """
    Rule-based analyzer for live suspicious pattern detection.
    """
    flags = []
    if pd.isna(text) or not text.strip():
        return flags
    
    if URL_REGEX.search(text):
        flags.append({"type": "URL Detected", "detail": "Contains direct web link or shortener", "severity": "High"})
    if CURRENCY_REGEX.search(text):
        flags.append({"type": "Monetary Hook", "detail": "Contains currency symbol or monetary prize reference", "severity": "Medium"})
    if PHONE_REGEX.search(text):
        flags.append({"type": "Phone/Shortcode", "detail": "Prompts calling/texting premium rate number", "severity": "High"})
    urgency_matches = URGENCY_REGEX.findall(text)
    if urgency_matches:
        unique_matches = list(set(w.lower() for w in urgency_matches))
        flags.append({"type": "Urgency Trigger", "detail": f"Keywords found: {', '.join(unique_matches)}", "severity": "Medium"})
    if sum(c.isupper() for c in text) > 10 and (sum(c.isupper() for c in text) / (len(text) + 1e-5)) > 0.3:
        flags.append({"type": "Excessive Capitals", "detail": "High proportion of uppercase text (shouting)", "severity": "Low"})
    if text.count("!") >= 2:
        flags.append({"type": "Sensational Punctuation", "detail": f"Multiple exclamation marks ({text.count('!')})", "severity": "Low"})
        
    return flags
