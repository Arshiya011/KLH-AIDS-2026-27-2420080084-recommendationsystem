import json
import pandas as pd
from src.config import PROCESSED_DATA_DIR, SEED, NUM_NEGATIVES
from src.preprocessing import create_id_mappings, normalize_ratings, random_train_test_split, leave_one_out_split

def main():
    print("======================================================================")
    print("             PHASE 2: PREPROCESSING & SPLITTING (MOVIELENS 100K)      ")
    print("======================================================================\n")
    
    # 1. Load Phase 1 Cleaned Data
    cleaned_path = PROCESSED_DATA_DIR / "movielens_100k_cleaned.parquet"
    if not cleaned_path.exists():
        raise FileNotFoundError("Run 'python run_phase1.py' first to generate cleaned parquet data.")
        
    df = pd.read_parquet(cleaned_path)
    
    # 2. Contiguous Integer Indexing
    df_mapped, mappings = create_id_mappings(df)
    
    # 3. Min-Max Rating Normalization [0, 1]
    df_preprocessed = normalize_ratings(df_mapped, min_rating=1.0, max_rating=5.0)
    
    # 4. Protocol A: Random 80/20 Split (Rating Prediction)
    train_rating_df, test_rating_df = random_train_test_split(df_preprocessed, test_size=0.20, seed=SEED)
    
    # 5. Protocol B: Temporal Leave-One-Out + 99 Negatives (Ranking Evaluation)
    train_ranking_df, test_positives_df, negative_samples = leave_one_out_split(
        df_preprocessed, num_negatives=NUM_NEGATIVES, seed=SEED
    )
    
    # 6. Save Artifacts
    df_preprocessed.to_parquet(PROCESSED_DATA_DIR / "movielens_100k_preprocessed.parquet", index=False)
    train_rating_df.to_parquet(PROCESSED_DATA_DIR / "movielens_100k_rating_train.parquet", index=False)
    test_rating_df.to_parquet(PROCESSED_DATA_DIR / "movielens_100k_rating_test.parquet", index=False)
    
    train_ranking_df.to_parquet(PROCESSED_DATA_DIR / "movielens_100k_ranking_train.parquet", index=False)
    test_positives_df.to_parquet(PROCESSED_DATA_DIR / "movielens_100k_ranking_test_positives.parquet", index=False)
    
    with open(PROCESSED_DATA_DIR / "movielens_100k_negative_samples.json", "w") as f:
        json.dump(negative_samples, f)
        
    with open(PROCESSED_DATA_DIR / "movielens_100k_mappings.json", "w") as f:
        json.dump({
            "user_to_idx": mappings["user_to_idx"],
            "item_to_idx": mappings["item_to_idx"]
        }, f)
        
    print(f"[SUMMARY] Total Contiguous Users: {len(mappings['user_to_idx'])}")
    print(f"[SUMMARY] Total Contiguous Items: {len(mappings['item_to_idx'])}\n")
    
    print("---------------- PROTOCOL A: RANDOM 80/20 SPLIT ----------------")
    print(f"Train Set Size: {len(train_rating_df)} rows ({len(train_rating_df)/len(df_preprocessed):.1%})")
    print(f"Test Set Size:  {len(test_rating_df)} rows ({len(test_rating_df)/len(df_preprocessed):.1%})")
    print("----------------------------------------------------------------\n")
    
    print("--------- PROTOCOL B: LEAVE-ONE-OUT + 99 NEGATIVES --------------")
    print(f"Train Interactions:  {len(train_ranking_df)} rows")
    print(f"Test Holdout Users:  {len(test_positives_df)} users (1 pos item each)")
    print(f"Sampled Negatives:   {len(negative_samples)} users × {NUM_NEGATIVES} items")
    print("----------------------------------------------------------------\n")
    print("[SUCCESS] Preprocessing artifacts saved successfully!")

if __name__ == "__main__":
    main()