import numpy as np
import pandas as pd

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def compute_user_ranking_metrics(ranked_item_indices: list[int], positive_item_idx: int, k_list: list[int] = [5, 10, 20]) -> dict:
    """
    Computes Hit Ratio (HR), Precision, Recall, MAP, and NDCG for a single user at K={5, 10, 20}.
    """
    metrics = {}
    
    # Locate rank position of positive item (1-indexed)
    if positive_item_idx in ranked_item_indices:
        rank_pos = ranked_item_indices.index(positive_item_idx) + 1
    else:
        rank_pos = None

    for k in k_list:
        if rank_pos is not None and rank_pos <= k:
            hit = 1.0
            precision = 1.0 / k
            recall = 1.0
            map_score = 1.0 / rank_pos
            ndcg = 1.0 / np.log2(rank_pos + 1)
        else:
            hit = 0.0
            precision = 0.0
            recall = 0.0
            map_score = 0.0
            ndcg = 0.0

        metrics[f"HR@{k}"] = hit
        metrics[f"Precision@{k}"] = precision
        metrics[f"Recall@{k}"] = recall
        metrics[f"MAP@{k}"] = map_score
        metrics[f"NDCG@{k}"] = ndcg

    return metrics

def evaluate_rating_predictions(test_preds_df: pd.DataFrame, model_cols: dict) -> pd.DataFrame:
    """Computes RMSE and MAE across all models."""
    y_true = test_preds_df['rating'].values
    results = []
    
    for model_name, col_name in model_cols.items():
        y_pred = test_preds_df[col_name].values
        y_pred_clipped = np.clip(y_pred, 1.0, 5.0)
        
        results.append({
            "Model": model_name,
            "RMSE": round(compute_rmse(y_true, y_pred_clipped), 4),
            "MAE": round(compute_mae(y_true, y_pred_clipped), 4)
        })
        
    return pd.DataFrame(results).set_index("Model")