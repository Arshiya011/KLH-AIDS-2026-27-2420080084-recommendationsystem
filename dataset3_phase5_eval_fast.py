import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

# Import NeuMF class definition
from dataset3_phase4_models import NeuMF

os.makedirs("results/tables", exist_ok=True)

print("Starting Fast Phase 5 Evaluation for Dataset 3 (Book-Crossing)...")

# ---------------------------------------------------------
# Step 1: Load Test Splits & Artifacts
# ---------------------------------------------------------
df_train = pd.read_csv("data/processed/dataset3_phase3_train_features.csv")
df_test_rmse = pd.read_csv("data/processed/dataset3_phase3_test_features.csv")
df_full = pd.read_csv("data/processed/dataset3_phase2_full.csv")

with open("data/processed/dataset3_phase2_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

user_encoder = artifacts['user_encoder']
item_encoder = artifacts['item_encoder']
negative_samples = artifacts['negative_samples']

n_users = len(user_encoder.classes_)
n_items = len(item_encoder.classes_)

ignore_cols = [
    'user_id', 'item_id', 'rating', 'rating_raw', 'rating_scaled',
    'user_idx', 'item_idx', 'title', 'author', 'publisher', 'location',
    'author_clean', 'publisher_clean', 'age_bucket', 'country'
]
features = [c for c in df_test_rmse.columns if c not in ignore_cols]

# ---------------------------------------------------------
# Step 2: Load Models
# ---------------------------------------------------------
with open("models/global_mean_bookcrossing.pkl", "rb") as f:
    global_mean_val = pickle.load(f)["global_mean"]

with open("models/item_knn_bookcrossing.pkl", "rb") as f:
    knn_artifact = pickle.load(f)
    item_sim = knn_artifact["item_sim"]
    R_train = knn_artifact["R_train"]

with open("models/xgb_bookcrossing.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("models/lgb_bookcrossing.pkl", "rb") as f:
    lgb_model = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ncf_model = NeuMF(num_users=n_users, num_items=n_items).to(device)
ncf_model.load_state_dict(torch.load("models/ncf_bookcrossing_best.pt", map_location=device))
ncf_model.eval()

# ---------------------------------------------------------
# Step 3: Fast Vectorized RMSE
# ---------------------------------------------------------
print("Evaluating RMSE across all models...")
y_true = df_test_rmse['rating_scaled'].values

# Global Mean
preds_gm = np.full(len(df_test_rmse), global_mean_val)
rmse_gm = np.sqrt(mean_squared_error(y_true, preds_gm))

# Item-KNN
preds_knn = []
for _, row in df_test_rmse.iterrows():
    u, i = int(row['user_idx']), int(row['item_idx'])
    user_ratings = R_train[u, :]
    sims = item_sim[i, :]
    rated_indices = user_ratings > 0
    if np.sum(sims[rated_indices]) > 0:
        pred = np.sum(sims[rated_indices] * user_ratings[rated_indices]) / np.sum(sims[rated_indices])
    else:
        pred = global_mean_val
    preds_knn.append(pred)
rmse_knn = np.sqrt(mean_squared_error(y_true, preds_knn))

# XGBoost & LightGBM
preds_xgb = xgb_model.predict(df_test_rmse[features])
rmse_xgb = np.sqrt(mean_squared_error(y_true, preds_xgb))

preds_lgb = lgb_model.predict(df_test_rmse[features])
rmse_lgb = np.sqrt(mean_squared_error(y_true, preds_lgb))

# NCF
u_test_tensor = torch.tensor(df_test_rmse['user_idx'].values, dtype=torch.long).to(device)
i_test_tensor = torch.tensor(df_test_rmse['item_idx'].values, dtype=torch.long).to(device)
with torch.no_grad():
    preds_ncf = ncf_model(u_test_tensor, i_test_tensor).cpu().numpy()
rmse_ncf = np.sqrt(mean_squared_error(y_true, preds_ncf))

# ---------------------------------------------------------
# Step 4: Vectorized Batch Ranking Evaluation (HR@10 & NDCG@10)
# ---------------------------------------------------------
print("Preparing vectorized candidate matrix for ranking (589,100 samples)...")

# Pre-build lookup dictionaries for speed
user_feat_map = df_test_rmse.groupby('user_idx')[features].first().to_dict('index')
global_feature_defaults = df_test_rmse[features].mean().to_dict()

u_indices_list = []
i_indices_list = []

for u_idx, neg_items in negative_samples.items():
    user_rows = df_full[df_full['user_idx'] == u_idx]
    if len(user_rows) == 0: continue
    
    pos_item_idx = user_rows.iloc[0]['item_idx']
    candidates = np.append([pos_item_idx], neg_items)
    
    u_indices_list.extend([u_idx] * 100)
    i_indices_list.extend(candidates)

u_array = np.array(u_indices_list)
i_array = np.array(i_indices_list)
num_eval_users = len(u_array) // 100

print("Calculating ranking scores for NCF...")
u_tensor_rank = torch.tensor(u_array, dtype=torch.long).to(device)
i_tensor_rank = torch.tensor(i_array, dtype=torch.long).to(device)

with torch.no_grad():
    scores_ncf = ncf_model(u_tensor_rank, i_tensor_rank).cpu().numpy()

print("Calculating ranking scores for Tree models...")
# Construct vectorized DataFrame for GBDTs once
ranking_df = pd.DataFrame({'user_idx': u_array, 'item_idx': i_array})
user_feat_df = df_test_rmse[['user_idx'] + features].drop_duplicates('user_idx').set_index('user_idx')

ranking_df = ranking_df.join(user_feat_df, on='user_idx', how='left')
ranking_df[features] = ranking_df[features].fillna(global_feature_defaults)

scores_xgb = xgb_model.predict(ranking_df[features])
scores_lgb = lgb_model.predict(ranking_df[features])

def compute_metrics_from_scores(scores_flat):
    hits, ndcgs = [], []
    scores_matrix = scores_flat.reshape(num_eval_users, 100)
    
    for row in scores_matrix:
        # Candidate at index 0 is always the true positive item
        rank_indices = np.argsort(row)[::-1]
        pos_rank = np.where(rank_indices == 0)[0][0]
        
        if pos_rank < 10:
            hits.append(1.0)
            ndcgs.append(1.0 / np.log2(pos_rank + 2))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)
            
    return np.mean(hits), np.mean(ndcgs)

# Item-KNN Fast Ranking Scores
print("Calculating ranking scores for Item-KNN...")
scores_knn = []
for u_idx, neg_items in negative_samples.items():
    user_rows = df_full[df_full['user_idx'] == u_idx]
    if len(user_rows) == 0: continue
    
    pos_item_idx = user_rows.iloc[0]['item_idx']
    candidates = np.append([pos_item_idx], neg_items)
    
    user_ratings = R_train[u_idx, :]
    for item_cand in candidates:
        sims = item_sim[item_cand, :]
        rated = user_ratings > 0
        if np.sum(sims[rated]) > 0:
            scores_knn.append(np.sum(sims[rated] * user_ratings[rated]) / np.sum(sims[rated]))
        else:
            scores_knn.append(0.0)

scores_knn = np.array(scores_knn)
scores_gm = np.random.rand(len(u_array))

print("Computing final metrics...")
hr_gm, ndcg_gm = compute_metrics_from_scores(scores_gm)
hr_knn, ndcg_knn = compute_metrics_from_scores(scores_knn)
hr_xgb, ndcg_xgb = compute_metrics_from_scores(scores_xgb)
hr_lgb, ndcg_lgb = compute_metrics_from_scores(scores_lgb)
hr_ncf, ndcg_ncf = compute_metrics_from_scores(scores_ncf)

# ---------------------------------------------------------
# Step 5: Save Results
# ---------------------------------------------------------
master_metrics = pd.DataFrame([
    {"Dataset": "Book-Crossing", "Model": "Global Mean", "RMSE": round(rmse_gm, 4), "HR@10": round(hr_gm, 4), "NDCG@10": round(ndcg_gm, 4)},
    {"Dataset": "Book-Crossing", "Model": "Item-KNN", "RMSE": round(rmse_knn, 4), "HR@10": round(hr_knn, 4), "NDCG@10": round(ndcg_knn, 4)},
    {"Dataset": "Book-Crossing", "Model": "XGBoost", "RMSE": round(rmse_xgb, 4), "HR@10": round(hr_xgb, 4), "NDCG@10": round(ndcg_xgb, 4)},
    {"Dataset": "Book-Crossing", "Model": "LightGBM", "RMSE": round(rmse_lgb, 4), "HR@10": round(hr_lgb, 4), "NDCG@10": round(ndcg_lgb, 4)},
    {"Dataset": "Book-Crossing", "Model": "NCF (NeuMF)", "RMSE": round(rmse_ncf, 4), "HR@10": round(hr_ncf, 4), "NDCG@10": round(ndcg_ncf, 4)}
])

master_metrics.to_csv("results/tables/book_crossing_master_metrics.csv", index=False)

print("\n=== Phase 5 Evaluation Complete ===")
print(master_metrics.to_string(index=False))