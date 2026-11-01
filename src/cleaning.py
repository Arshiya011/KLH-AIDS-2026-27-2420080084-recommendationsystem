import pandas as pd

def compute_quality_metrics(df: pd.DataFrame) -> dict:
    """Calculates data quality & sparsity metrics for a given ratings DataFrame."""
    n_rows = len(df)
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()
    possible_interactions = n_users * n_items
    
    sparsity = (1.0 - (n_rows / possible_interactions)) * 100.0 if possible_interactions > 0 else 0.0
    density = (n_rows / possible_interactions) * 100.0 if possible_interactions > 0 else 0.0
    
    return {
        "Rows": n_rows,
        "Unique Users": n_users,
        "Unique Items": n_items,
        "Sparsity %": round(sparsity, 4),
        "Density %": round(density, 4),
        "Min Rating": df['rating'].min() if n_rows > 0 else 0,
        "Max Rating": df['rating'].max() if n_rows > 0 else 0
    }

def clean_movielens_100k(
    ratings_df: pd.DataFrame, 
    k_core: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans raw ratings data:
    1. Removes missing ratings/IDs.
    2. Deduplicates (user, item) pairs (keeps latest timestamp).
    3. Filters out-of-range ratings (1.0 to 5.0).
    4. Applies iterative k-core filtering until convergence.
    """
    metrics_before = compute_quality_metrics(ratings_df)
    metrics_before["Stage"] = "1. Raw Data"
    
    df = ratings_df.copy()
    
    # Drop missing essential values
    df = df.dropna(subset=['user_id', 'item_id', 'rating'])
    
    # Sort by timestamp and keep latest user-item rating interaction
    df = df.sort_values(by=['user_id', 'item_id', 'timestamp'])
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
    
    # Enforce standard rating bounds
    df = df[(df['rating'] >= 1.0) & (df['rating'] <= 5.0)]
    
    # Iterative k-core filtering
    while True:
        initial_len = len(df)
        
        # Filter items with fewer than k_core ratings
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= k_core].index
        df = df[df['item_id'].isin(valid_items)]
        
        # Filter users with fewer than k_core ratings
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= k_core].index
        df = df[df['user_id'].isin(valid_users)]
        
        # Stop when no records are dropped in a full iteration
        if len(df) == initial_len:
            break
            
    metrics_after = compute_quality_metrics(df)
    metrics_after["Stage"] = f"2. Cleaned ({k_core}-Core)"
    
    summary_df = pd.DataFrame([metrics_before, metrics_after]).set_index("Stage")
    return df, summary_df