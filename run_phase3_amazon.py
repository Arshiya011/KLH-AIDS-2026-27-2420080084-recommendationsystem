import json
import numpy as np
import pandas as pd
import nltk
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon quiet
nltk.download('vader_lexicon', quiet=True)

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "amazon_foods_train.parquet"
TEST_PATH = PROCESSED_DIR / "amazon_foods_test.parquet"

def build_amazon_features():
    print("--- Phase 3: Feature Engineering Store (Amazon Fine Foods) ---")
    
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        print("❌ ERROR: Train/Test split files missing. Please execute Phase 2 first.")
        return

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    print(f"Loaded Train ({len(train_df):,} rows) and Test ({len(test_df):,} rows).")
    
    # Global Mean Rating (calculated strictly on train set)
    global_mean = float(train_df["rating"].mean())
    
    # ---------------------------------------------------------
    # 1. POPULARITY & BIASES
    # ---------------------------------------------------------
    print("1/4 Computing Popularity & Bias Signals...")
    
    # User Statistics
    u_stats = train_df.groupby("user_idx")["rating"].agg(
        user_count="count",
        user_mean_rating="mean"
    ).reset_index()
    u_stats["user_bias"] = u_stats["user_mean_rating"] - global_mean
    
    # Item Statistics & Empirical Bayes Shrunk Mean
    i_stats = train_df.groupby("item_idx")["rating"].agg(
        item_count="count",
        item_mean_rating="mean"
    ).reset_index()
    
    m_weight = 10.0 # Shrinkage parameter
    i_stats["item_shrunk_mean"] = (
        (i_stats["item_count"] * i_stats["item_mean_rating"] + m_weight * global_mean) / 
        (i_stats["item_count"] + m_weight)
    )
    i_stats["item_bias"] = i_stats["item_mean_rating"] - global_mean

    # Merge Biases
    def merge_biases(df):
        df = df.merge(u_stats, on="user_idx", how="left")
        df = df.merge(i_stats, on="item_idx", how="left")
        
        df["user_count"] = df["user_count"].fillna(0).astype(np.int32)
        df["user_mean_rating"] = df["user_mean_rating"].fillna(global_mean).astype(np.float32)
        df["user_bias"] = df["user_bias"].fillna(0.0).astype(np.float32)
        
        df["item_count"] = df["item_count"].fillna(0).astype(np.int32)
        df["item_mean_rating"] = df["item_mean_rating"].fillna(global_mean).astype(np.float32)
        df["item_shrunk_mean"] = df["item_shrunk_mean"].fillna(global_mean).astype(np.float32)
        df["item_bias"] = df["item_bias"].fillna(0.0).astype(np.float32)
        return df

    train_df = merge_biases(train_df)
    test_df = merge_biases(test_df)

    # ---------------------------------------------------------
    # 2. SVD COLLABORATIVE LATENT FACTORS
    # ---------------------------------------------------------
    print("2/4 Fitting Collaborative SVD Factors (k=16)...")
    
    n_users = max(train_df["user_idx"].max(), test_df["user_idx"].max()) + 1
    n_items = max(train_df["item_idx"].max(), test_df["item_idx"].max()) + 1
    
    # Construct interaction matrix from training split
    from scipy.sparse import csr_matrix
    R_train = csr_matrix(
        (train_df["rating"].values, (train_df["user_idx"].values, train_df["item_idx"].values)),
        shape=(n_users, n_items),
        dtype=np.float32
    )
    
    svd_collab = TruncatedSVD(n_components=16, random_state=42)
    user_factors = svd_collab.fit_transform(R_train) # (n_users, 16)
    item_factors = svd_collab.components_.T           # (n_items, 16)
    
    def attach_svd_factors(df):
        u_idx = df["user_idx"].values
        i_idx = df["item_idx"].values
        
        u_f = user_factors[u_idx] # (N, 16)
        i_f = item_factors[i_idx] # (N, 16)
        
        dot_prod = np.sum(u_f * i_f, axis=1, keepdims=True)
        
        for k in range(16):
            df[f"user_svd_{k}"] = u_f[:, k].astype(np.float32)
            df[f"item_svd_{k}"] = i_f[:, k].astype(np.float32)
            
        df["svd_dot_product"] = dot_prod.squeeze().astype(np.float32)
        return df

    train_df = attach_svd_factors(train_df)
    test_df = attach_svd_factors(test_df)

    # ---------------------------------------------------------
    # 3. TEMPORAL SIGNALS
    # ---------------------------------------------------------
    print("3/4 Engineering Temporal Signals...")
    
    def add_temporal(df):
        dt = pd.to_datetime(df["timestamp"], unit='s')
        df["year"] = dt.dt.year.astype(np.int16)
        df["month"] = dt.dt.month.astype(np.int8)
        df["day_of_week"] = dt.dt.dayofweek.astype(np.int8)
        df["hour"] = dt.dt.hour.astype(np.int8)
        
        min_user_time = train_df.groupby("user_idx")["timestamp"].min().to_dict()
        u_min = df["user_idx"].map(min_user_time).fillna(df["timestamp"])
        df["user_tenure_days"] = ((df["timestamp"] - u_min) / 86400.0).astype(np.float32)
        return df

    train_df = add_temporal(train_df)
    test_df = add_temporal(test_df)

    # ---------------------------------------------------------
    # 4. AMAZON CONTENT & TEXT FEATURES
    # ---------------------------------------------------------
    print("4/4 Extracting Text TF-IDF + SVD, VADER Sentiment & Helpfulness Signals...")
    
    # 4a. Character and Word Counts
    for df in [train_df, test_df]:
        df["review_char_len"] = df["text_body"].fillna("").str.len().astype(np.int32)
        df["review_word_count"] = df["text_body"].fillna("").str.split().str.len().fillna(0).astype(np.int32)
        df["summary_char_len"] = df["text_summary"].fillna("").str.len().astype(np.int32)
        df["summary_word_count"] = df["text_summary"].fillna("").str.split().str.len().fillna(0).astype(np.int32)
        
        # Helpfulness Ratio with Laplace +1 Smoothing
        df["help_ratio"] = (df["help_num"] / (df["help_den"] + 1.0)).astype(np.float32)

    # 4b. TF-IDF + TruncatedSVD (Text Factors)
    tfidf = TfidfVectorizer(max_features=10000, min_df=5, ngram_range=(1, 2))
    train_tfidf = tfidf.fit_transform(train_df["clean_text"].fillna(""))
    test_tfidf = tfidf.transform(test_df["clean_text"].fillna(""))
    
    svd_text = TruncatedSVD(n_components=20, random_state=42)
    train_text_svd = svd_text.fit_transform(train_tfidf)
    test_text_svd = svd_text.transform(test_tfidf)
    
    for k in range(20):
        train_df[f"text_svd_{k}"] = train_text_svd[:, k].astype(np.float32)
        test_df[f"text_svd_{k}"] = test_text_svd[:, k].astype(np.float32)

    # 4c. VADER Sentiment Polarity
    sia = SentimentIntensityAnalyzer()
    
    def extract_vader(series):
        results = []
        for text in series.fillna(""):
            if not text:
                results.append((0.0, 0.0, 0.0, 0.0))
            else:
                scores = sia.polarity_scores(text)
                results.append((scores['pos'], scores['neg'], scores['neu'], scores['compound']))
        return np.array(results, dtype=np.float32)

    print("Computing VADER sentiment scores for train set...")
    train_vader = extract_vader(train_df["clean_text"])
    print("Computing VADER sentiment scores for test set...")
    test_vader = extract_vader(test_df["clean_text"])
    
    for idx, col in enumerate(["vader_pos", "vader_neg", "vader_neu", "vader_compound"]):
        train_df[col] = train_vader[:, idx]
        test_df[col] = test_vader[:, idx]

    # ---------------------------------------------------------
    # SAVE FEATURE MATRIX PARQUETS
    # ---------------------------------------------------------
    feat_train_path = PROCESSED_DIR / "amazon_foods_features_train.parquet"
    feat_test_path = PROCESSED_DIR / "amazon_foods_features_test.parquet"
    
    train_df.to_parquet(feat_train_path, index=False)
    test_df.to_parquet(feat_test_path, index=False)
    
    print("\n✅ Phase 3 Execution Complete!")
    print(f"• Feature Train Shape: {train_df.shape} -> saved to {feat_train_path.name}")
    print(f"• Feature Test Shape:  {test_df.shape} -> saved to {feat_test_path.name}")

if __name__ == "__main__":
    build_amazon_features()