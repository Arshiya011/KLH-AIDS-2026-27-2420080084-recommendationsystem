import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

# Import NeuMF class definition for loading weights
from dataset3_phase4_models import NeuMF

os.makedirs("results/tables", exist_ok=True)

print("Starting Phase 5 Evaluation for Dataset 3 (Book-Crossing)...")

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

# Feature column list for GBDT models
ignore_cols = [
    'user_id', 'item_id', 'rating', 'rating_raw', 'rating_scaled',
    'user_idx', 'item_idx', 'title', 'author', 'publisher', 'location',
    'author_clean', 'publisher_clean', 'age_bucket', 'country'
]
features = [c for c in df_test_rmse.columns if c not in ignore_cols]

# ---------------------------------------------------------
# Step 2: Load Trained Models
# ---------------------------------------------------------
# 1. Global Mean
with open("models/global_mean_bookcrossing.pkl", "rb") as f:
    global_mean_val = pickle.load(f)["global_mean"]

# 2. Item-KNN
with open("models/item_knn_bookcrossing.pkl", "rb") as f:
    knn_artifact = pickle.load(f)
    item_sim = knn_artifact["item_sim"]
    R_train = knn_artifact["R_train"]

# 3. XGBoost
with open("models/xgb_bookcrossing.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# 4. LightGBM
with open("models/lgb_bookcrossing.pkl", "rb") as f:
    lgb_model = pickle.load(f)

# 5. NCF (PyTorch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ncf_model = NeuMF(num_users=n_users, num_items=n_items).to(device)
ncf_model.load_state_dict(torch.load("models/ncf_bookcrossing_best.pt", map_location=device))
ncf_model.eval()

# ---------------------------------------------------------
# Step 3: Compute RMSE on Test Split
# ---------------------------------------------------------
y_true = df_test_rmse['rating_scaled'].values

# 1. Global Mean Prediction
preds_gm = np.full(len(df_test_rmse), global_mean_val)
rmse_gm = np.sqrt(mean_squared_error(y_true, preds_gm))

# 2. Item-KNN Prediction
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

# 3. XGBoost Prediction
preds_xgb = xgb_model.predict(df_test_rmse[features])
rmse_xgb = np.sqrt(mean_squared_error(y_true, preds_xgb))

# 4. LightGBM Prediction
preds_lgb = lgb_model.predict(df_test_rmse[features])
rmse_lgb = np.sqrt(mean_squared_error(y_true, preds_lgb))

# 5. NCF Prediction
u_test_tensor = torch.tensor(df_test_rmse['user_idx'].values, dtype=torch.long).to(device)
i_test_tensor = torch.tensor(df_test_rmse['item_idx'].values, dtype=torch.long).to(device)

with torch.no_grad():
    preds_ncf = ncf_model(u_test_tensor, i_test_tensor).cpu().numpy()
rmse_ncf = np.sqrt(mean_squared_error(y_true, preds_ncf))

# ---------------------------------------------------------
# Step 4: Compute Ranking Metrics (HR@10 & NDCG@10 via LOO + 99 Negatives)
# ---------------------------------------------------------
def evaluate_ranking(model_type, model_obj=None):
    hits, ndcgs = [], []
    
    # LOO test target per user
    for u_idx, neg_items in negative_samples.items():
        user_rows = df_full[df_full['user_idx'] == u_idx]
        if len(user_rows) == 0: continue
        
        pos_item_idx = user_rows.iloc[0]['item_idx']
        candidate_items = np.append([pos_item_idx], neg_items)
        
        if model_type == 'global_mean':
            scores = np.random.rand(100)
            
        elif model_type == 'knn':
            user_ratings = R_train[u_idx, :]
            scores = []
            for item_cand in candidate_items:
                sims = item_sim[item_cand, :]
                rated = user_ratings > 0
                if np.sum(sims[rated]) > 0:
                    scores.append(np.sum(sims[rated] * user_ratings[rated]) / np.sum(sims[rated]))
                else:
                    scores.append(0.0)
            scores = np.array(scores)
            
        elif model_type in ['xgb', 'lgb']:
            cand_df = pd.DataFrame({'user_idx': u_idx, 'item_idx': candidate_items})
            
            # Temporary feature merge for candidates
            for col in features:
                if col in df_test_rmse.columns:
                    val = df_test_rmse[df_test_rmse['user_idx'] == u_idx][col].values
                    cand_df[col] = val[0] if len(val) > 0 else 0
            
            scores = model_obj.predict(cand_df[features])
            
        elif model_type == 'ncf':
            u_tens = torch.tensor([u_idx] * 100, dtype=torch.long).to(device)
            i_tens = torch.tensor(candidate_items, dtype=torch.long).to(device)
            with torch.no_grad():
                scores = model_obj(u_tens, i_tens).cpu().numpy()

        # Rank candidates (descending)
        rank_indices = np.argsort(scores)[::-1]
        pos_rank = np.where(rank_indices == 0)[0][0]
        
        # Hit Ratio @ 10
        if pos_rank < 10:
            hits.append(1.0)
            ndcgs.append(1.0 / np.log2(pos_rank + 2))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)
            
    return np.mean(hits), np.mean(ndcgs)

print("Evaluating ranking metrics (HR@10 & NDCG@10)...")
hr_gm, ndcg_gm = evaluate_ranking('global_mean')
hr_knn, ndcg_knn = evaluate_ranking('knn')
hr_xgb, ndcg_xgb = evaluate_ranking('xgb', xgb_model)
hr_lgb, ndcg_lgb = evaluate_ranking('lgb', lgb_model)
hr_ncf, ndcg_ncf = evaluate_ranking('ncf', ncf_model)

# ---------------------------------------------------------
# Step 5: Save Final Master Metrics Table
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