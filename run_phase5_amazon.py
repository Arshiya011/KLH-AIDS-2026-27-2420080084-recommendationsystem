import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TABLES_DIR = BASE_DIR / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# PyTorch NeuMF Architecture Definition
# ---------------------------------------------------------
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim_mf=32, latent_dim_mlp=32, mlp_hidden_dims=[64, 32, 16]):
        super(NeuMF, self).__init__()
        
        self.user_embed_mf = nn.Embedding(num_users, latent_dim_mf)
        self.item_embed_mf = nn.Embedding(num_items, latent_dim_mf)
        
        self.user_embed_mlp = nn.Embedding(num_users, latent_dim_mlp)
        self.item_embed_mlp = nn.Embedding(num_items, latent_dim_mlp)
        
        mlp_layers = []
        input_dim = latent_dim_mlp * 2
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=0.2))
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        
        final_input_dim = latent_dim_mf + mlp_hidden_dims[-1]
        self.prediction_layer = nn.Linear(final_input_dim, 1)
        
    def forward(self, user_indices, item_indices):
        user_mf = self.user_embed_mf(user_indices)
        item_mf = self.item_embed_mf(item_indices)
        gmf_vector = user_mf * item_mf
        
        user_mlp = self.user_embed_mlp(user_indices)
        item_mlp = self.item_embed_mlp(item_indices)
        mlp_vector = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_vector = self.mlp(mlp_vector)
        
        fusion = torch.cat([gmf_vector, mlp_vector], dim=-1)
        logits = self.prediction_layer(fusion)
        return logits.squeeze(-1)

def calculate_ranking_metrics(model_preds_dict, loo_test_df, neg_samples, top_k=[5, 10]):
    """Calculates Hit Ratio (HR), NDCG, and MRR on Leave-One-Out + 99 Negative Samples."""
    results = {}
    
    user_indices = loo_test_df["user_idx"].values
    item_indices = loo_test_df["item_idx"].values
    
    for model_name, score_fn in model_preds_dict.items():
        print(f"  Evaluating ranking for: {model_name}...")
        hrs = {k: [] for k in top_k}
        ndcgs = {k: [] for k in top_k}
        mrrs = []
        
        for idx in range(len(loo_test_df)):
            u_int = int(user_indices[idx])
            pos_i = int(item_indices[idx])
            
            negs = neg_samples.get(str(u_int), [])
            if len(negs) == 0:
                continue
                
            candidate_items = [pos_i] + negs
    
    # --- FIX: shuffle candidates so tied scores don't favor index 0 ---
            rng = np.random.default_rng(seed=idx)  # seeded per-row for reproducibility
            perm = rng.permutation(len(candidate_items))
            candidate_items = [candidate_items[j] for j in perm]
            true_pos_in_shuffled = int(np.where(perm == 0)[0][0])  # where pos_i ended up
    
            scores = score_fn(u_int, candidate_items)
    
            ranked_indices = np.argsort(-scores)
            pos_rank = np.where(ranked_indices == true_pos_in_shuffled)[0][0] + 1
    
            mrrs.append(1.0 / pos_rank)
            for k in top_k:
                if pos_rank <= k:
                    hrs[k].append(1.0)
                    ndcgs[k].append(1.0 / np.log2(pos_rank + 1))
                else:
                    hrs[k].append(0.0)
                    ndcgs[k].append(0.0)
            
                                
        res = {
            f"HR@{k}": float(np.mean(hrs[k])) for k in top_k
        }
        for k in top_k:
            res[f"NDCG@{k}"] = float(np.mean(ndcgs[k]))
        res["MRR"] = float(np.mean(mrrs))
        results[model_name] = res
        
    return results

def run_phase5_amazon():
    print("--- Phase 5: Master Metrics Evaluation (Amazon Fine Foods) ---")
    
    pred_path = PROCESSED_DIR / "amazon_foods_test_predictions.parquet"
    loo_test_path = PROCESSED_DIR / "amazon_foods_loo_test.parquet"
    neg_samples_path = PROCESSED_DIR / "amazon_foods_negative_samples.json"
    feat_train_path = PROCESSED_DIR / "amazon_foods_features_train.parquet"
    feat_test_path = PROCESSED_DIR / "amazon_foods_features_test.parquet"
    mapping_path = PROCESSED_DIR / "amazon_foods_mappings.json"
    
    if not pred_path.exists():
        print("❌ ERROR: Test predictions missing. Run Phase 4 first.")
        return

    # Load Data
    preds_df = pd.read_parquet(pred_path)
    loo_test_df = pd.read_parquet(loo_test_path)
    train_feat = pd.read_parquet(feat_train_path)
    test_feat = pd.read_parquet(feat_test_path)
    
    with open(neg_samples_path, "r") as f:
        neg_samples = json.load(f)
        
    with open(mapping_path, "r") as f:
        mappings = json.load(f)

    y_true = preds_df["rating"].values
    
    models_map = {
        "Global Mean": "pred_global_mean",
        "Item-KNN": "pred_knn",
        "XGBoost": "pred_xgb",
        "LightGBM": "pred_lgb",
        "NCF (NeuMF)": "pred_ncf"
    }

    # 1. RATING REGRESSION METRICS
    print("1/2 Calculating Regression Metrics (RMSE, MAE)...")
    reg_metrics = {}
    for model_label, col in models_map.items():
        y_pred = preds_df[col].values
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        reg_metrics[model_label] = {"RMSE": round(rmse, 4), "MAE": round(mae, 4)}

    # 2. TOP-N RANKING METRICS
    print("2/2 Calculating Ranking Metrics (HR@5, HR@10, NDCG@5, NDCG@10, MRR)...")
    
    global_mean_val = float(np.mean(train_feat["rating"].values))

    # Pre-load trained models and extract exact feature names
    xgb_path = BASE_DIR / "models" / "xgboost_amazon.json"
    model_xgb = None
    xgb_feature_names = []
    if xgb_path.exists():
        model_xgb = xgb.XGBRegressor()
        model_xgb.load_model(str(xgb_path))
        xgb_feature_names = model_xgb.get_booster().feature_names

    lgb_path = BASE_DIR / "models" / "lightgbm_amazon.txt"
    model_lgb = None
    if lgb_path.exists():
        model_lgb = lgb.Booster(model_file=str(lgb_path))

    ncf_path = BASE_DIR / "models" / "ncf_amazon_best.pt"
    model_ncf = None
    if ncf_path.exists():
        try:
            model_ncf = torch.load(ncf_path, map_location=torch.device('cpu'), weights_only=False)
            model_ncf.eval()
        except Exception:
            model_ncf = None

    # Pre-build feature lookups
    item_feat_dict = train_feat.groupby("item_idx").first().to_dict(orient="index")
    user_feat_dict = train_feat.groupby("user_idx").first().to_dict(orient="index")

    # Scoring functions
    def score_global_mean(u_int, items):
        return np.full(len(items), global_mean_val)

    def score_knn(u_int, items):
        return np.array([item_feat_dict.get(i, {}).get("item_mean_rating", global_mean_val) for i in items])

    def score_xgb(u_int, items):
        if model_xgb is None or not xgb_feature_names:
            return np.full(len(items), global_mean_val)
        u_f = user_feat_dict.get(u_int, {})
        rows = []
        for i in items:
            i_f = item_feat_dict.get(i, {})
            merged = {**u_f, **i_f}
            rows.append([merged.get(c, 0.0) for c in xgb_feature_names])
        X_cand = pd.DataFrame(rows, columns=xgb_feature_names).apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
        return model_xgb.predict(X_cand)

    def score_lgb(u_int, items):
        if model_lgb is None or not xgb_feature_names:
            return np.full(len(items), global_mean_val)
        u_f = user_feat_dict.get(u_int, {})
        rows = []
        for i in items:
            i_f = item_feat_dict.get(i, {})
            merged = {**u_f, **i_f}
            rows.append([merged.get(c, 0.0) for c in xgb_feature_names])
        X_cand = pd.DataFrame(rows, columns=xgb_feature_names).apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
        return model_lgb.predict(X_cand)

    def score_ncf(u_int, items):
        if model_ncf is None:
            return np.full(len(items), global_mean_val)
        u_t = torch.tensor([u_int] * len(items), dtype=torch.long)
        i_t = torch.tensor(items, dtype=torch.long)
        with torch.no_grad():
            return model_ncf(u_t, i_t).numpy()

    scoring_fns = {
        "Global Mean": score_global_mean,
        "Item-KNN": score_knn,
        "XGBoost": score_xgb,
        "LightGBM": score_lgb,
        "NCF (NeuMF)": score_ncf
    }
    
    ranking_metrics = calculate_ranking_metrics(
        scoring_fns, 
        loo_test_df.sample(min(1000, len(loo_test_df)), random_state=42), 
        neg_samples
    )

    # 3. CONSOLIDATE MASTER METRICS TABLE
    master_rows = []
    for model_label in models_map.keys():
        r_m = reg_metrics.get(model_label, {})
        rk_m = ranking_metrics.get(model_label, {"HR@5": 0.0, "HR@10": 0.0, "NDCG@5": 0.0, "NDCG@10": 0.0, "MRR": 0.0})
        
        row = {
            "Model": model_label,
            "RMSE": r_m.get("RMSE", 0.0),
            "MAE": r_m.get("MAE", 0.0),
            "HR@5": round(rk_m.get("HR@5", 0.0), 4),
            "HR@10": round(rk_m.get("HR@10", 0.0), 4),
            "NDCG@5": round(rk_m.get("NDCG@5", 0.0), 4),
            "NDCG@10": round(rk_m.get("NDCG@10", 0.0), 4),
            "MRR": round(rk_m.get("MRR", 0.0), 4)
        }
        master_rows.append(row)
        
    master_df = pd.DataFrame(master_rows).set_index("Model")
    
    out_path = TABLES_DIR / "amazon_foods_master_metrics.csv"
    master_df.to_csv(out_path)
    
    print("\n=========================================================")
    print("  AMAZON FINE FOODS: MASTER EVALUATION METRICS TABLE  ")
    print("=========================================================")
    print(master_df.to_string())
    print(f"\n✅ Results matrix successfully saved to '{out_path}'")

if __name__ == "__main__":
    run_phase5_amazon()