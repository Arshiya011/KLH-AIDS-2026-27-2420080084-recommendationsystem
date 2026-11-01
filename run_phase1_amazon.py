import os
import pandas as pd
import numpy as np
from pathlib import Path

# Paths matched to your project structure
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "amazon" / "Reviews.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TABLES_DIR = BASE_DIR / "results" / "tables"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

def run_phase1_amazon():
    print("--- Phase 1: Cleaning & Core Pruning (Amazon Fine Foods) ---")
    
    if not RAW_DATA_PATH.exists():
        print(f"❌ ERROR: Could not find dataset at '{RAW_DATA_PATH}'. Please place 'Reviews.csv' inside 'data/raw/amazon/'.")
        return

    # 1. Load Raw Data
    df_raw = pd.read_csv(RAW_DATA_PATH)
    
    # 2. Schema Mapping
    df = df_raw.rename(columns={
        "UserId": "user_id",
        "ProductId": "item_id",
        "Score": "rating",
        "Time": "timestamp",
        "Summary": "text_summary",
        "Text": "text_body",
        "HelpfulnessNumerator": "help_num",
        "HelpfulnessDenominator": "help_den"
    })
    
    df["rating"] = df["rating"].astype(np.float32)
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df["text_summary"] = df["text_summary"].fillna("").astype(str)
    df["text_body"] = df["text_body"].fillna("").astype(str)
    
    # 3. Drop Helpfulness Anomalies (Numerator > Denominator)
    df_valid_help = df[df["help_num"] <= df["help_den"]].copy()
    
    # 4. Deduplicate (user_id, item_id) -> Keep Last by Timestamp
    df_sorted = df_valid_help.sort_values("timestamp")
    df_dedup = df_sorted.drop_duplicates(subset=["user_id", "item_id"], keep="last").copy()
    
    # Core Filtering Function
    def apply_core_filtering(data_df, min_core=5):
        df_core = data_df.copy()
        while True:
            u_counts = df_core["user_id"].value_counts()
            i_counts = df_core["item_id"].value_counts()
            
            valid_users = u_counts[u_counts >= min_core].index
            valid_items = i_counts[i_counts >= min_core].index
            
            new_df = df_core[df_core["user_id"].isin(valid_users) & df_core["item_id"].isin(valid_items)]
            
            if len(new_df) == len(df_core):
                break
            df_core = new_df
        return df_core

    df_5core = apply_core_filtering(df_dedup, min_core=5)
    df_3core = apply_core_filtering(df_dedup, min_core=3)
    
    # 5. Metrics Calculation
    def calc_metrics(data_df):
        r = len(data_df)
        u = data_df["user_id"].nunique()
        i = data_df["item_id"].nunique()
        density = (r / (u * i)) * 100 if (u * i) > 0 else 0
        return r, u, i, round(density, 4)

    r_raw, u_raw, i_raw, d_raw = calc_metrics(df_raw.rename(columns={"UserId": "user_id", "ProductId": "item_id"}))
    r_dedup, u_dedup, i_dedup, d_dedup = calc_metrics(df_dedup)
    r_3c, u_3c, i_3c, d_3c = calc_metrics(df_3core)
    r_5c, u_5c, i_5c, d_5c = calc_metrics(df_5core)
    
    summary_data = {
        "Stage": [
            "1. Raw Dataset",
            "2. Deduplicated & Invalid Dropped",
            "3. Cleaned (3-Core Alternative)",
            "4. Cleaned (5-Core Primary)"
        ],
        "Rows": [r_raw, r_dedup, r_3c, r_5c],
        "Unique Users": [u_raw, u_dedup, u_3c, u_5c],
        "Unique Items": [i_raw, i_dedup, i_3c, i_5c],
        "Density %": [d_raw, d_dedup, d_3c, d_5c],
        "Volume Retained %": [
            100.0,
            round((r_dedup / r_raw) * 100, 2),
            round((r_3c / r_raw) * 100, 2),
            round((r_5c / r_raw) * 100, 2)
        ]
    }
    
    summary_df = pd.DataFrame(summary_data).set_index("Stage")
    
    summary_path = TABLES_DIR / "amazon_foods_phase1_summary.csv"
    summary_df.to_csv(summary_path)
    
    cleaned_parquet_path = PROCESSED_DIR / "amazon_foods_cleaned.parquet"
    df_5core.to_parquet(cleaned_parquet_path, index=False)
    
    print("\nPhase 1 Summary Table:")
    print(summary_df.to_string())
    print(f"\n✅ Cleaned dataset saved to: {cleaned_parquet_path}")
    print(f"✅ Summary table saved to: {summary_path}")

if __name__ == "__main__":
    run_phase1_amazon()