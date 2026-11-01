import os
import json
import re
import pandas as pd
import numpy as np
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Ensure NLTK stopwords are available
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLEANED_PARQUET_PATH = PROCESSED_DIR / "amazon_foods_cleaned.parquet"

def clean_text(text):
    """Strips HTML, removes punctuation/numbers, lowercases, and removes stopwords."""
    if not isinstance(text, str) or not text:
        return ""
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove non-alphabetical characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()
    # Remove stopwords and excess whitespace
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)

def run_phase2_amazon():
    print("--- Phase 2: Preprocessing, Text Cleaning & Dual-Split Protocol (Amazon Fine Foods) ---")
    
    if not CLEANED_PARQUET_PATH.exists():
        print(f"❌ ERROR: Cleaned file not found at '{CLEANED_PARQUET_PATH}'. Run Phase 1 first.")
        return

    # 1. Load Cleaned Dataset
    df = pd.read_parquet(CLEANED_PARQUET_PATH)
    print(f"Loaded {len(df):,} cleaned interactions.")
    
    # 2. Text Preprocessing (Concatenate summary + body, then clean)
    print("Executing NLP text cleaning on reviews...")
    raw_text_series = df["text_summary"].fillna("") + " " + df["text_body"].fillna("")
    df["clean_text"] = raw_text_series.apply(clean_text)
    
    # 3. Mappings (Contiguous Integer Encoding)
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())
    
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
    
    df["user_idx"] = df["user_id"].map(user_to_idx).astype(np.int32)
    df["item_idx"] = df["item_id"].map(item_to_idx).astype(np.int32)
    
    # Save Mappings JSON
    mappings = {
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "num_users": len(unique_users),
        "num_items": len(unique_items)
    }
    mapping_path = PROCESSED_DIR / "amazon_foods_mappings.json"
    with open(mapping_path, "w") as f:
        json.dump(mappings, f)
        
    # 4. Protocol A: 80/20 Stratified/Random Train/Test Split (Rating Regression)
    print("Generating Protocol A (80/20 Train/Test Split)...")
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, shuffle=True)
    
    train_path = PROCESSED_DIR / "amazon_foods_train.parquet"
    test_path = PROCESSED_DIR / "amazon_foods_test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    # 5. Protocol B: Leave-One-Out (LOO) + 99 Negative Sampling (Top-N Ranking)
    print("Generating Protocol B (Leave-One-Out + 99 Negative Sampling)...")
    df_sorted = df.sort_values(["user_idx", "timestamp"])
    
    # Hold out the latest interaction per user
    loo_test_positives = df_sorted.groupby("user_idx").tail(1).copy()
    loo_train = df_sorted.drop(loo_test_positives.index).copy()
    
    # Negative Sampling: Sample 99 unobserved items per user
    all_item_set = set(range(len(unique_items)))
    user_observed_map = df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    
    np.random.seed(42)
    negative_samples = {}
    for u_idx in unique_users:
        u_int = user_to_idx[u_idx]
        observed = user_observed_map.get(u_int, set())
        unobserved = list(all_item_set - observed)
        
        if len(unobserved) >= 99:
            negs = np.random.choice(unobserved, size=99, replace=False).tolist()
        else:
            negs = np.random.choice(unobserved, size=99, replace=True).tolist()
        negative_samples[str(u_int)] = negs
        
    loo_test_path = PROCESSED_DIR / "amazon_foods_loo_test.parquet"
    loo_train_path = PROCESSED_DIR / "amazon_foods_loo_train.parquet"
    neg_samples_path = PROCESSED_DIR / "amazon_foods_negative_samples.json"
    
    loo_test_positives.to_parquet(loo_test_path, index=False)
    loo_train.to_parquet(loo_train_path, index=False)
    
    with open(neg_samples_path, "w") as f:
        json.dump(negative_samples, f)
        
    print("\n✅ Phase 2 Execution Complete!")
    print(f"• User Count: {len(unique_users):,} | Item Count: {len(unique_items):,}")
    print(f"• Train Set (80%): {len(train_df):,} rows saved to {train_path.name}")
    print(f"• Test Set (20%): {len(test_df):,} rows saved to {test_path.name}")
    print(f"• LOO Test Set: {len(loo_test_positives):,} rows saved to {loo_test_path.name}")
    print(f"• Mappings & 99-Negative Samples Saved.")

if __name__ == "__main__":
    run_phase2_amazon()