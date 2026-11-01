import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, SEED
from src.models.baselines import GlobalMeanRecommender, ItemKNNRecommender
from src.models.ml_models import train_xgboost_regressor, train_lightgbm_regressor
from src.models.ncf import NeuralCollaborativeFiltering

def main():
    print("======================================================================")
    print("                 PHASE 4: MODEL TRAINING (MOVIELENS 100K)             ")
    print("======================================================================\n")
    
    # 1. Load Data
    train_feat = pd.read_parquet(PROCESSED_DATA_DIR / "movielens_100k_features_train.parquet")
    test_feat = pd.read_parquet(PROCESSED_DATA_DIR / "movielens_100k_features_test.parquet")
    
    with open(PROCESSED_DATA_DIR / "movielens_100k_mappings.json", "r") as f:
        mappings = json.load(f)
        
    num_users = len(mappings["user_to_idx"])
    num_items = len(mappings["item_to_idx"])
    
    # ---------------------------------------------------------
    # 1. Non-ML Baselines
    # ---------------------------------------------------------
    print("[1/5] Training Non-ML Baselines (Global Mean & Item-KNN)...")
    global_model = GlobalMeanRecommender()
    global_model.fit(train_feat)
    
    item_knn = ItemKNNRecommender(k_neighbors=20)
    item_knn.fit(train_feat, n_users=num_users, n_items=num_items)
    
    # ---------------------------------------------------------
    # 2. Machine Learning Models (XGBoost & LightGBM)
    # ---------------------------------------------------------
    print("[2/5] Training XGBoost Regressor...")
    xgb_model, feature_cols = train_xgboost_regressor(train_feat, seed=SEED)
    
    print("[3/5] Training LightGBM Regressor...")
    lgb_model, _ = train_lightgbm_regressor(train_feat, seed=SEED)
    
    # ---------------------------------------------------------
    # 3. Deep Learning: Neural Collaborative Filtering (NCF)
    # ---------------------------------------------------------
    print("[4/5] Training Deep Learning: Neural Collaborative Filtering (NCF)...")
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ncf_model = NeuralCollaborativeFiltering(
        num_users=num_users, 
        num_items=num_items, 
        latent_dim_gmf=16, 
        latent_dim_mlp=32
    ).to(device)
    
    # DataLoader setup
    u_train = torch.tensor(train_feat['user_idx'].values, dtype=torch.long)
    i_train = torch.tensor(train_feat['item_idx'].values, dtype=torch.long)
    r_train = torch.tensor(train_feat['rating_norm'].values, dtype=torch.float32)
    
    train_dataset = TensorDataset(u_train, i_train, r_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(ncf_model.parameters(), lr=0.001)
    
    ncf_model.train()
    for epoch in range(1, 6):  # 5 fast epochs for execution verification
        epoch_loss = 0.0
        for u_batch, i_batch, r_batch in train_loader:
            u_batch, i_batch, r_batch = u_batch.to(device), i_batch.to(device), r_batch.to(device)
            optimizer.zero_grad()
            preds = ncf_model(u_batch, i_batch)
            loss = criterion(preds, r_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(r_batch)
        print(f"   Epoch {epoch}/5 - NCF Loss: {epoch_loss/len(train_feat):.4f}")
        
    # ---------------------------------------------------------
    # 4. Save Models & Generate Sample Predictions
    # ---------------------------------------------------------
    print("\n[5/5] Generating evaluation predictions for Test set...")
    
    test_preds_df = test_feat[['user_idx', 'item_idx', 'rating']].copy()
    test_preds_df['pred_global'] = global_model.predict(test_feat['user_idx'].values, test_feat['item_idx'].values)
    test_preds_df['pred_knn'] = item_knn.predict(test_feat['user_idx'].values, test_feat['item_idx'].values)
    test_preds_df['pred_xgb'] = xgb_model.predict(test_feat[feature_cols])
    test_preds_df['pred_lgb'] = lgb_model.predict(test_feat[feature_cols])
    
    # NCF Test Inference
    ncf_model.eval()
    u_test = torch.tensor(test_feat['user_idx'].values, dtype=torch.long).to(device)
    i_test = torch.tensor(test_feat['item_idx'].values, dtype=torch.long).to(device)
    with torch.no_grad():
        ncf_norm_preds = ncf_model(u_test, i_test).cpu().numpy()
        # Rescale normalized predictions back to 1.0 - 5.0 rating scale
        test_preds_df['pred_ncf'] = 1.0 + ncf_norm_preds * 4.0
        
    test_preds_df.to_parquet(PROCESSED_DATA_DIR / "movielens_100k_test_predictions.parquet", index=False)
    print("\n[SUCCESS] Phase 4 Model Training Complete! Saved test predictions.")

if __name__ == "__main__":
    main()