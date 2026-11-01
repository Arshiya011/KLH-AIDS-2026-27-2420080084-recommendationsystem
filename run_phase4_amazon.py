import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# PyTorch NeuMF Architecture Definition
# ---------------------------------------------------------
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim_mf=32, latent_dim_mlp=32, mlp_hidden_dims=[64, 32, 16]):
        super(NeuMF, self).__init__()
        
        # GMF Embeddings
        self.user_embed_mf = nn.Embedding(num_users, latent_dim_mf)
        self.item_embed_mf = nn.Embedding(num_items, latent_dim_mf)
        
        # MLP Embeddings
        self.user_embed_mlp = nn.Embedding(num_users, latent_dim_mlp)
        self.item_embed_mlp = nn.Embedding(num_items, latent_dim_mlp)
        
        # MLP Layers
        mlp_layers = []
        input_dim = latent_dim_mlp * 2
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=0.2))
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        
        # NeuMF Final Prediction Layer
        final_input_dim = latent_dim_mf + mlp_hidden_dims[-1]
        self.prediction_layer = nn.Linear(final_input_dim, 1)
        
    def forward(self, user_indices, item_indices):
        # GMF Branch
        user_mf = self.user_embed_mf(user_indices)
        item_mf = self.item_embed_mf(item_indices)
        gmf_vector = user_mf * item_mf
        
        # MLP Branch
        user_mlp = self.user_embed_mlp(user_indices)
        item_mlp = self.item_embed_mlp(item_indices)
        mlp_vector = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_vector = self.mlp(mlp_vector)
        
        # Fusion
        fusion = torch.cat([gmf_vector, mlp_vector], dim=-1)
        logits = self.prediction_layer(fusion)
        return logits.squeeze(-1)

class RecSysDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        
    def __len__(self):
        return len(self.users)
        
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

def run_phase4_amazon():
    print("--- Phase 4: Model Training Pipeline (Amazon Fine Foods) ---")
    
    train_feat_path = PROCESSED_DIR / "amazon_foods_features_train.parquet"
    test_feat_path = PROCESSED_DIR / "amazon_foods_features_test.parquet"
    mapping_path = PROCESSED_DIR / "amazon_foods_mappings.json"
    
    if not train_feat_path.exists() or not test_feat_path.exists():
        print("❌ ERROR: Feature parquets missing. Run Phase 3 first.")
        return

    train_df = pd.read_parquet(train_feat_path)
    test_df = pd.read_parquet(test_feat_path)
    
    with open(mapping_path, "r") as f:
        mappings = json.load(f)
        
    num_users = mappings["num_users"]
    num_items = mappings["num_items"]
    
    ignore_cols = {
        'user_id', 'item_id', 'user_idx', 'item_idx', 'rating', 'timestamp', 
        'text_summary', 'text_body', 'clean_text'
    }
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]
    
    X_train = train_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
    y_train = train_df["rating"].values.astype(np.float32)
    
    X_test = test_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
    y_test = test_df["rating"].values.astype(np.float32)
    
    predictions_df = test_df[["user_idx", "item_idx", "rating"]].copy()

    # ---------------------------------------------------------
    # 1. MODEL 1: GLOBAL MEAN
    # ---------------------------------------------------------
    print("\n1/5 Training Model 1: Global Mean Baseline...")
    global_mean_val = float(np.mean(y_train))
    predictions_df["pred_global_mean"] = global_mean_val
    
    with open(MODELS_DIR / "global_mean_amazon.json", "w") as f:
        json.dump({"global_mean": global_mean_val}, f)

    # ---------------------------------------------------------
    # 2. MODEL 2: ITEM-KNN COLLABORATIVE FILTERING
    # ---------------------------------------------------------
    print("2/5 Computing Model 2: Item-KNN Cosine Similarity...")
    
    R_sparse = csr_matrix(
        (y_train, (train_df["user_idx"].values, train_df["item_idx"].values)),
        shape=(num_users, num_items),
        dtype=np.float32
    )
    
    item_sim = cosine_similarity(R_sparse.T, dense_output=False)
    
    def predict_item_knn(test_df_in, k=20):
        preds = []
        for _, row in test_df_in.iterrows():
            u = int(row["user_idx"])
            i = int(row["item_idx"])
            
            user_ratings = R_sparse[u].toarray().squeeze()
            rated_items = np.where(user_ratings > 0)[0]
            
            if len(rated_items) == 0:
                preds.append(global_mean_val)
                continue
                
            sims = np.atleast_1d(item_sim[i, rated_items].toarray().squeeze())
            
            top_k_idx = np.argsort(-sims)[:k]
            top_sims = sims[top_k_idx]
            top_ratings = user_ratings[rated_items[top_k_idx]]
            
            sim_sum = np.sum(np.abs(top_sims))
            if sim_sum > 0:
                pred = np.sum(top_sims * top_ratings) / sim_sum
            else:
                pred = global_mean_val
            preds.append(float(pred))
        return np.array(preds, dtype=np.float32)

    predictions_df["pred_knn"] = predict_item_knn(test_df)

    # ---------------------------------------------------------
    # 3. MODEL 3: XGBOOST REGRESSOR
    # ---------------------------------------------------------
    print("3/5 Training Model 3: XGBoost Regressor...")
    model_xgb = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model_xgb.fit(X_train, y_train)
    predictions_df["pred_xgb"] = model_xgb.predict(X_test)
    model_xgb.save_model(str(MODELS_DIR / "xgboost_amazon.json"))

    # ---------------------------------------------------------
    # 4. MODEL 4: LIGHTGBM REGRESSOR
    # ---------------------------------------------------------
    print("4/5 Training Model 4: LightGBM Regressor...")
    model_lgb = lgb.LGBMRegressor(
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    model_lgb.fit(X_train, y_train)
    predictions_df["pred_lgb"] = model_lgb.predict(X_test)
    model_lgb.booster_.save_model(str(MODELS_DIR / "lightgbm_amazon.txt"))

    # ---------------------------------------------------------
    # 5. MODEL 5: NCF (NEUMF) PYTORCH DEEP LEARNING
    # ---------------------------------------------------------
    print("5/5 Training Model 5: NCF NeuMF (PyTorch Neural Network)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    ncf_model = NeuMF(num_users=num_users, num_items=num_items).to(device)
    optimizer = torch.optim.Adam(ncf_model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    train_dataset = RecSysDataset(train_df["user_idx"].values, train_df["item_idx"].values, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    ncf_model.train()
    for epoch in range(1, 6):
        epoch_loss = 0.0
        for batch_u, batch_i, batch_r in train_loader:
            batch_u, batch_i, batch_r = batch_u.to(device), batch_i.to(device), batch_r.to(device)
            
            optimizer.zero_grad()
            outputs = ncf_model(batch_u, batch_i)
            loss = criterion(outputs, batch_r)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_u)
            
        print(f"  • Epoch {epoch}/5 - Loss: {epoch_loss / len(train_dataset):.4f}")
        
    ncf_model.eval()
    test_u_tensor = torch.tensor(test_df["user_idx"].values, dtype=torch.long).to(device)
    test_i_tensor = torch.tensor(test_df["item_idx"].values, dtype=torch.long).to(device)
    
    with torch.no_grad():
        ncf_preds = ncf_model(test_u_tensor, test_i_tensor).cpu().numpy()
        
    predictions_df["pred_ncf"] = ncf_preds
    
    ncf_save_path = MODELS_DIR / "ncf_amazon_best.pt"
    torch.save(ncf_model, ncf_save_path)

    # ---------------------------------------------------------
    # SAVE TEST PREDICTIONS PARQUET
    # ---------------------------------------------------------
    pred_path = PROCESSED_DIR / "amazon_foods_test_predictions.parquet"
    predictions_df.to_parquet(pred_path, index=False)
    
    print("\n✅ Phase 4 Execution Complete!")
    print(f"• All 5 Model Artifacts saved to '{MODELS_DIR}'")
    print(f"• Test Predictions matrix ({len(predictions_df):,} rows) saved to '{pred_path.name}'")

if __name__ == "__main__":
    run_phase4_amazon()