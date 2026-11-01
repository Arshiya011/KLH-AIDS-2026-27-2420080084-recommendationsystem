import json
import torch
import numpy as np
import pandas as pd
from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, TOP_K_LIST, SEED
from src.evaluation import evaluate_rating_predictions, compute_user_ranking_metrics
from src.models.ml_models import train_xgboost_regressor, train_lightgbm_regressor, get_feature_columns
from src.models.ncf import NeuralCollaborativeFiltering
from src.feature_engineering import (
    compute_bias_and_popularity_features,
    extract_svd_latent_features,
    extract_movielens_metadata,
    build_feature_matrix
)
from src.data_loader import load_movielens_100k, RAW_DATA_DIR

def main():
    print("======================================================================")
    print("           PHASE 5: COMPREHENSIVE EVALUATION (MOVIELENS 100K)         ")
    print("======================================================================\n")
    
    # ---------------------------------------------------------
    # 1. RATING PREDICTION METRICS (RMSE & MAE)
    # ---------------------------------------------------------
    print("[1/2] Computing Rating Prediction Metrics (RMSE & MAE)...")
    test_preds = pd.read_parquet(PROCESSED_DATA_DIR / "movielens_100k_test_predictions.parquet")
    model_cols = {
        "Global Mean": "pred_global",
        "Item-KNN": "pred_knn",
        "XGBoost": "pred_xgb",
        "LightGBM": "pred_lgb",
        "NCF (NeuMF)": "pred_ncf"
    }
    rating_summary = evaluate_rating_predictions(test_preds, model_cols)
    
    # ---------------------------------------------------------
    # 2. TOP-N RANKING METRICS
    # ---------------------------------------------------------
    print("[2/2] Evaluating Top-N Ranking Protocol (1 Pos + 99 Sampled Negatives per user)...")
    
    train_ranking_df = pd.read_parquet(PROCESSED_DATA_DIR / "movielens_100k_ranking_train.parquet")
    test_positives_df = pd.read_parquet(PROCESSED_DATA_DIR / "movielens_100k_ranking_test_positives.parquet")
    
    with open(PROCESSED_DATA_DIR / "movielens_100k_negative_samples.json", "r") as f:
        negative_samples = json.load(f)
        
    with open(PROCESSED_DATA_DIR / "movielens_100k_mappings.json", "r") as f:
        mappings = json.load(f)
        
    num_users = len(mappings["user_to_idx"])
    num_items = len(mappings["item_to_idx"])
    
    # Fit Feature Transformers ON TRAIN SET ONLY
    train_feat = pd.read_parquet(PROCESSED_DATA_DIR / "movielens_100k_features_train.parquet")
    user_stats, item_stats, global_mean = compute_bias_and_popularity_features(train_ranking_df, shrink_factor=10.0)
    user_svd, item_svd = extract_svd_latent_features(train_ranking_df, n_components=16)
    _, movies_meta, _ = load_movielens_100k(RAW_DATA_DIR)
    movie_meta_processed = extract_movielens_metadata(movies_meta, mappings)
    
    # Train XGBoost & LightGBM on train features
    xgb_model, feature_cols = train_xgboost_regressor(train_feat, seed=SEED)
    lgb_model, _ = train_lightgbm_regressor(train_feat, seed=SEED)
    
    # Train NCF for Ranking
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ncf_model = NeuralCollaborativeFiltering(num_users=num_users, num_items=num_items).to(device)
    
    u_tr = torch.tensor(train_ranking_df['user_idx'].values, dtype=torch.long).to(device)
    i_tr = torch.tensor(train_ranking_df['item_idx'].values, dtype=torch.long).to(device)
    r_tr = torch.tensor(train_ranking_df['rating_norm'].values, dtype=torch.float32).to(device)
    
    optimizer = torch.optim.Adam(ncf_model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    ncf_model.train()
    dataset = torch.utils.data.TensorDataset(u_tr, i_tr, r_tr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    for epoch in range(1, 6):
        for u_b, i_b, r_b in loader:
            optimizer.zero_grad()
            loss = criterion(ncf_model(u_b, i_b), r_b)
            loss.backward()
            optimizer.step()
            
    ncf_model.eval()
    
    # Compute Item Popularity Scores
    item_pop_scores = train_ranking_df['item_idx'].value_counts().to_dict()
    
    ranking_results = {model_name: [] for model_name in model_cols.keys()}
    
    # Evaluate Users
    for row in test_positives_df.itertuples():
        user_idx = int(row.user_idx)
        sample = negative_samples[str(user_idx)]
        pos_item = int(sample["positive_item"])
        candidate_items = [pos_item] + [int(x) for x in sample["negative_items"]]
        rng = np.random.default_rng(seed=user_idx)
        shuffle_order = rng.permutation(len(candidate_items))
        candidate_items = [candidate_items[i] for i in shuffle_order]
        
        # 1. Global Mean / Popularity
        pop_preds = np.array([item_pop_scores.get(item, 0) for item in candidate_items])
        
        # 2. Build Candidate Feature Table dynamically for this (user_idx, candidate_items)
        cand_df = pd.DataFrame({"user_idx": user_idx, "item_idx": candidate_items, "timestamp": row.timestamp})
        cand_df = build_feature_matrix(
            cand_df, user_stats, item_stats, user_svd, item_svd, movie_meta_processed, global_mean, svd_k=16
        )
        
        # XGBoost & LightGBM Scores
        xgb_scores = xgb_model.predict(cand_df[feature_cols])
        lgb_scores = lgb_model.predict(cand_df[feature_cols])
        
        # Item-KNN Proxy (Bias + Popularity Signal)
        knn_scores = cand_df['item_shrunk_mean'].values
        
        # 3. NCF Predictions
        u_tensor = torch.tensor([user_idx] * 100, dtype=torch.long).to(device)
        i_tensor = torch.tensor(candidate_items, dtype=torch.long).to(device)
        with torch.no_grad():
            ncf_scores = ncf_model(u_tensor, i_tensor).cpu().numpy()
            
        # Rank Candidates Descending
        models_ranked_items = {
            "Global Mean": [candidate_items[i] for i in np.argsort(-pop_preds)],
            "Item-KNN": [candidate_items[i] for i in np.argsort(-knn_scores)],
            "XGBoost": [candidate_items[i] for i in np.argsort(-xgb_scores)],
            "LightGBM": [candidate_items[i] for i in np.argsort(-lgb_scores)],
            "NCF (NeuMF)": [candidate_items[i] for i in np.argsort(-ncf_scores)],
        }
        
        for model_name, ranked_list in models_ranked_items.items():
            user_metrics = compute_user_ranking_metrics(ranked_list, pos_item, TOP_K_LIST)
            ranking_results[model_name].append(user_metrics)
            
    # Aggregate mean ranking metrics across all users
    ranking_summary_rows = []
    for model_name, user_metric_list in ranking_results.items():
        avg_metrics = pd.DataFrame(user_metric_list).mean().to_dict()
        avg_metrics["Model"] = model_name
        ranking_summary_rows.append(avg_metrics)
        
    ranking_summary = pd.DataFrame(ranking_summary_rows).set_index("Model")
    
    for col in ranking_summary.columns:
        ranking_summary[col] = ranking_summary[col].round(4)
        
    master_table = rating_summary.join(ranking_summary)
    master_table.to_csv(RESULTS_DIR / "tables" / "movielens_100k_master_metrics.csv")
    
    print("\n========================= MASTER EVALUATION COMPARISON TABLE =========================")
    print(master_table[["RMSE", "MAE", "HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20"]].to_string())
    print("======================================================================================\n")
    print(f"[SUCCESS] Complete Master Evaluation Table saved to: {RESULTS_DIR / 'tables' / 'movielens_100k_master_metrics.csv'}")

if __name__ == "__main__":
    main()