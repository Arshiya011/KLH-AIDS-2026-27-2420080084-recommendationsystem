import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import SEED, NUM_NEGATIVES, PROCESSED_DATA_DIR

def create_id_mappings(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encodes raw user_id and item_id strings to contiguous integer indices [0, N-1].
    Saves and returns forward/reverse index mapping dictionaries.
    """
    unique_users = sorted(df['user_id'].unique())
    unique_items = sorted(df['item_id'].unique())
    
    user_to_idx = {user: int(idx) for idx, user in enumerate(unique_users)}
    item_to_idx = {item: int(idx) for idx, item in enumerate(unique_items)}
    
    idx_to_user = {int(idx): user for user, idx in user_to_idx.items()}
    idx_to_item = {int(idx): item for item, idx in item_to_idx.items()}
    
    df_mapped = df.copy()
    df_mapped['user_idx'] = df_mapped['user_id'].map(user_to_idx)
    df_mapped['item_idx'] = df_mapped['item_id'].map(item_to_idx)
    
    mappings = {
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_user": idx_to_user,
        "idx_to_item": idx_to_item
    }
    
    return df_mapped, mappings

def normalize_ratings(df: pd.DataFrame, min_rating: float = 1.0, max_rating: float = 5.0) -> pd.DataFrame:
    """Applies Min-Max scaling to compress ratings into [0, 1]."""
    df_scaled = df.copy()
    df_scaled['rating_norm'] = (df_scaled['rating'] - min_rating) / (max_rating - min_rating)
    return df_scaled

def random_train_test_split(df: pd.DataFrame, test_size: float = 0.20, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Protocol A: Uniform 80/20 Train/Test split for Rating Prediction (RMSE/MAE).
    """
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * test_size)
    
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    train_df = df.iloc[train_indices].copy().reset_index(drop=True)
    test_df = df.iloc[test_indices].copy().reset_index(drop=True)
    
    return train_df, test_df

def leave_one_out_split(df: pd.DataFrame, num_negatives: int = NUM_NEGATIVES, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Protocol B: Temporal Leave-One-Out split + Negative Sampling for Ranking Metrics (NDCG, MAP, HR).
    - Holds out the latest interaction per user for testing.
    - Samples 99 unobserved negative items per user for test evaluation.
    """
    np.random.seed(seed)
    df_sorted = df.sort_values(by=['user_idx', 'timestamp'])
    
    # Hold out latest interaction per user for testing
    test_train_ranks = df_sorted.groupby('user_idx').cumcount(ascending=False)
    
    train_df = df_sorted[test_train_ranks > 0].copy().reset_index(drop=True)
    test_positives = df_sorted[test_train_ranks == 0].copy().reset_index(drop=True)
    
    all_items = set(df['item_idx'].unique())
    user_interacted_items = df.groupby('user_idx')['item_idx'].apply(set).to_dict()
    
    negative_samples = {}
    for user_idx, pos_item in zip(test_positives['user_idx'], test_positives['item_idx']):
        unobserved_items = list(all_items - user_interacted_items[user_idx])
        sampled_negatives = np.random.choice(unobserved_items, size=num_negatives, replace=False).tolist()
        negative_samples[str(user_idx)] = {
            "positive_item": int(pos_item),
            "negative_items": [int(x) for x in sampled_negatives]
        }
        
    return train_df, test_positives, negative_samples