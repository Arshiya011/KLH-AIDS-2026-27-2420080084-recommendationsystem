import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

os.makedirs("models", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

print("Starting Phase 4 Model Training for Dataset 3 (Book-Crossing)...")

# ---------------------------------------------------------
# Step 1: Load Phase 2/3 Feature Stores & Artifacts
# ---------------------------------------------------------
df_train = pd.read_csv("data/processed/dataset3_phase3_train_features.csv")
df_test = pd.read_csv("data/processed/dataset3_phase3_test_features.csv")

with open("data/processed/dataset3_phase2_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

n_users = len(artifacts['user_encoder'].classes_)
n_items = len(artifacts['item_encoder'].classes_)

# ---------------------------------------------------------
# Model 1: Global Mean
# ---------------------------------------------------------
print("\n--- Training Model 1: Global Mean ---")
global_mean_val = df_train['rating_scaled'].mean()
print(f"Global Mean Rating: {global_mean_val:.4f}")

with open("models/global_mean_bookcrossing.pkl", "wb") as f:
    pickle.dump({"global_mean": global_mean_val}, f)

# ---------------------------------------------------------
# Model 2: Item-KNN
# ---------------------------------------------------------
print("\n--- Training Model 2: Item-KNN ---")
R_train = np.zeros((n_users, n_items))
for _, row in df_train.iterrows():
    R_train[int(row['user_idx']), int(row['item_idx'])] = row['rating_scaled']

# Item Similarity Matrix (Cosine)
from sklearn.metrics.pairwise import cosine_similarity
item_sim = cosine_similarity(R_train.T)
np.fill_diagonal(item_sim, 0)

with open("models/item_knn_bookcrossing.pkl", "wb") as f:
    pickle.dump({"item_sim": item_sim, "R_train": R_train}, f)

# ---------------------------------------------------------
# Model 3 & 4: XGBoost & LightGBM
# ---------------------------------------------------------
print("\n--- Training Models 3 & 4: XGBoost and LightGBM ---")

# Define feature subset for Tree models
ignore_cols = [
    'user_id', 'item_id', 'rating', 'rating_raw', 'rating_scaled',
    'user_idx', 'item_idx', 'title', 'author', 'publisher', 'location',
    'author_clean', 'publisher_clean', 'age_bucket', 'country'
]

features = [c for c in df_train.columns if c not in ignore_cols]

X_train, y_train = df_train[features], df_train['rating_scaled']
X_test, y_test = df_test[features], df_test['rating_scaled']

# Model 3: XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

with open("models/xgb_bookcrossing.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

# Model 4: LightGBM
lgb_model = lgb.LGBMRegressor(
    n_estimators=150,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train)

with open("models/lgb_bookcrossing.pkl", "wb") as f:
    pickle.dump(lgb_model, f)

# ---------------------------------------------------------
# Model 5: NCF / NeuMF (PyTorch)
# ---------------------------------------------------------
print("\n--- Training Model 5: NCF (NeuMF PyTorch) ---")

class NCFDataset(Dataset):
    def __init__(self, u, i, r):
        self.u = torch.tensor(u, dtype=torch.long)
        self.i = torch.tensor(i, dtype=torch.long)
        self.r = torch.tensor(r, dtype=torch.float32)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx]

train_dataset = NCFDataset(df_train['user_idx'].values, df_train['item_idx'].values, df_train['rating_scaled'].values)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim_gmf=16, layers_mlp=[32, 16, 8]):
        super(NeuMF, self).__init__()
        
        # GMF Embeddings
        self.embed_user_gmf = nn.Embedding(num_users, latent_dim_gmf)
        self.embed_item_gmf = nn.Embedding(num_items, latent_dim_gmf)
        
        # MLP Embeddings
        self.embed_user_mlp = nn.Embedding(num_users, layers_mlp[0] // 2)
        self.embed_item_mlp = nn.Embedding(num_items, layers_mlp[0] // 2)
        
        # MLP Layers
        mlp_modules = []
        for i in range(len(layers_mlp) - 1):
            mlp_modules.append(nn.Linear(layers_mlp[i], layers_mlp[i+1]))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=0.2))
        self.mlp = nn.Sequential(*mlp_modules)
        
        # Final Output Layer
        self.predict_layer = nn.Linear(latent_dim_gmf + layers_mlp[-1], 1)

    def forward(self, user_indices, item_indices):
        # GMF Part
        user_gmf = self.embed_user_gmf(user_indices)
        item_gmf = self.embed_item_gmf(item_indices)
        gmf_vector = user_gmf * item_gmf
        
        # MLP Part
        user_mlp = self.embed_user_mlp(user_indices)
        item_mlp = self.embed_item_mlp(item_indices)
        mlp_vector = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_vector = self.mlp(mlp_vector)
        
        # Concatenate GMF + MLP
        vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        prediction = self.predict_layer(vector)
        return prediction.squeeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ncf_model = NeuMF(num_users=n_users, num_items=n_items).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ncf_model.parameters(), lr=0.001, weight_decay=1e-5)

epochs = 10
ncf_model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for u, i, r in train_loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        optimizer.zero_grad()
        preds = ncf_model(u, i)
        loss = criterion(preds, r)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(r)
    print(f"Epoch {epoch+1}/{epochs} | Training Loss (MSE): {total_loss / len(train_dataset):.4f}")

torch.save(ncf_model.state_dict(), "models/ncf_bookcrossing_best.pt")

# Save Summary Table
summary_p4 = pd.DataFrame([
    {"Model": "Global Mean", "Type": "Baseline", "Saved Artifact": "models/global_mean_bookcrossing.pkl"},
    {"Model": "Item-KNN", "Type": "Collaborative Filtering", "Saved Artifact": "models/item_knn_bookcrossing.pkl"},
    {"Model": "XGBoost", "Type": "Gradient Boosted Trees", "Saved Artifact": "models/xgb_bookcrossing.pkl"},
    {"Model": "LightGBM", "Type": "Gradient Boosted Trees", "Saved Artifact": "models/lgb_bookcrossing.pkl"},
    {"Model": "NCF (NeuMF)", "Type": "Deep Learning", "Saved Artifact": "models/ncf_bookcrossing_best.pt"}
])

summary_p4.to_csv("results/tables/book_crossing_phase4_summary.csv", index=False)
print("\nPhase 4 complete! Model artifacts saved in models/ and summary in results/tables/book_crossing_phase4_summary.csv")