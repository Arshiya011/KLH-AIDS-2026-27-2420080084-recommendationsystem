import json
import pandas as pd
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data_loader import load_movielens_100k
from src.feature_engineering import (
    compute_bias_and_popularity_features,
    extract_svd_latent_features,
    extract_movielens_metadata,
    build_feature_matrix
)

def main():
    print("======================================================================")
    print("             PHASE 3: FEATURE ENGINEERING (MOVIELENS 100K)            ")
    print("======================================================================\n")
    
    # 1. Load Preprocessed Data & Mappings
    train_df = pd.read_parquet(PROCESSED_DATA_DIR / "movielens_100k_rating_train.parquet")
    test_df = pd.read_parquet(PROCESSED_DATA_DIR / "movielens_100k_rating_test.parquet")
    
    with open(PROCESSED_DATA_DIR / "movielens_100k_mappings.json", "r") as f:
        mappings = json.load(f)
        
    _, movies_meta, _ = load_movielens_100k(RAW_DATA_DIR)
    
    # 2. Fit Features ON TRAIN SET ONLY (No Data Leakage)
    print("[INFO] Fitting User/Item stats, Bayesian shrunk means, and SVD factors on Train set...")
    user_stats, item_stats, global_mean = compute_bias_and_popularity_features(train_df, shrink_factor=10.0)
    user_svd, item_svd = extract_svd_latent_features(train_df, n_components=16)
    movie_meta_processed = extract_movielens_metadata(movies_meta, mappings)
    
    # 3. Build Feature Matrices for Train and Test
    print("[INFO] Constructing final tabular feature matrices for Train and Test splits...")
    train_features = build_feature_matrix(
        train_df, user_stats, item_stats, user_svd, item_svd, movie_meta_processed, global_mean, svd_k=16
    )
    test_features = build_feature_matrix(
        test_df, user_stats, item_stats, user_svd, item_svd, movie_meta_processed, global_mean, svd_k=16
    )
    
    # 4. Save Feature Matrices
    train_features.to_parquet(PROCESSED_DATA_DIR / "movielens_100k_features_train.parquet", index=False)
    test_features.to_parquet(PROCESSED_DATA_DIR / "movielens_100k_features_test.parquet", index=False)
    
    print("\n---------------- FEATURE MATRIX SUMMARY ----------------")
    print(f"Total Feature Columns Generated: {train_features.shape[1]}")
    print(f"Train Matrix Shape: {train_features.shape}")
    print(f"Test Matrix Shape:  {test_features.shape}")
    print("--------------------------------------------------------\n")
    print("[SUCCESS] Feature engineering complete! Parquet feature files saved.")

if __name__ == "__main__":
    main()