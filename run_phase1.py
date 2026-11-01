from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, K_CORE
from src.data_loader import load_movielens_100k
from src.cleaning import clean_movielens_100k

def main():
    print("======================================================================")
    print("               PHASE 1: DATA CLEANING (MOVIELENS 100K)                ")
    print("======================================================================\n")
    
    # 1. Load Data
    raw_ratings, movies_meta, users_meta = load_movielens_100k(RAW_DATA_DIR)
    
    # 2. Clean Data & Apply 5-Core Filtering
    cleaned_ratings, summary_table = clean_movielens_100k(raw_ratings, k_core=K_CORE)
    
    # 3. Save Processed Artifacts
    cleaned_ratings.to_parquet(PROCESSED_DATA_DIR / "movielens_100k_cleaned.parquet", index=False)
    summary_table.to_csv(RESULTS_DIR / "tables" / "movielens_100k_phase1_summary.csv")
    
    # 4. Print Summary Table
    print("\n---------------- DATA QUALITY & SPARSITY SUMMARY ----------------")
    print(summary_table.to_string())
    print("------------------------------------------------------------------\n")
    print(f"[SUCCESS] Cleaned data saved to: {PROCESSED_DATA_DIR / 'movielens_100k_cleaned.parquet'}")
    print(f"[SUCCESS] Quality table saved to: {RESULTS_DIR / 'tables' / 'movielens_100k_phase1_summary.csv'}\n")

if __name__ == "__main__":
    main()