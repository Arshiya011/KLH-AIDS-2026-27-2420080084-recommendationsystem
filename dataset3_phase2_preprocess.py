import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

os.makedirs("results/tables", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Load cleaned 5-core data from Phase 1
df = pd.read_csv("data/processed/dataset3_cleaned_5core.csv")

# Rescale ratings from [1, 10] to [1, 5]
df['rating_raw'] = df['rating']
df['rating_scaled'] = 1.0 + 4.0 * ((df['rating_raw'] - 1.0) / 9.0)

# Label encoding zero-indexed IDs
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df['user_idx'] = user_encoder.fit_transform(df['user_id'])
df['item_idx'] = item_encoder.fit_transform(df['item_id'])

# 80/20 Stratified Split for RMSE
np.random.seed(42)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_list, test_list = [], []
for u_idx, group in df.groupby('user_idx'):
    n_test = max(1, int(len(group) * 0.2))
    test_list.append(group.iloc[:n_test])
    train_list.append(group.iloc[n_test:])

df_train_rmse = pd.concat(train_list).reset_index(drop=True)
df_test_rmse = pd.concat(test_list).reset_index(drop=True)

# Leave-One-Out (LOO) + 99 Negative Sampling
loo_test_list, loo_train_list = [], []
all_item_indices = set(range(len(item_encoder.classes_)))
negative_samples = {}

for u_idx, group in df.groupby('user_idx'):
    loo_test_item = group.iloc[0]
    loo_test_list.append(loo_test_item)
    loo_train_list.append(group.iloc[1:])
    
    user_observed = set(group['item_idx'])
    unobserved = list(all_item_indices - user_observed)
    neg_sampled = np.random.choice(unobserved, size=99, replace=False)
    negative_samples[u_idx] = neg_sampled

df_train_loo = pd.concat(loo_train_list).reset_index(drop=True)
df_test_loo = pd.DataFrame(loo_test_list).reset_index(drop=True)

# Save Processed Datasets
df.to_csv("data/processed/dataset3_phase2_full.csv", index=False)
df_train_rmse.to_csv("data/processed/dataset3_phase2_train_rmse.csv", index=False)
df_test_rmse.to_csv("data/processed/dataset3_phase2_test_rmse.csv", index=False)

artifacts = {
    'user_encoder': user_encoder,
    'item_encoder': item_encoder,
    'negative_samples': negative_samples
}

with open("data/processed/dataset3_phase2_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

# Summary Table
summary_p2 = pd.DataFrame([
    {
        "Dataset": "Book-Crossing",
        "Total Cleaned Interactions": len(df),
        "RMSE Train Set (80%)": len(df_train_rmse),
        "RMSE Test Set (20%)": len(df_test_rmse),
        "LOO Train Set": len(df_train_loo),
        "LOO Test Targets": len(df_test_loo),
        "Negative Samples / User": 99,
        "Rating Min (Scaled)": df['rating_scaled'].min(),
        "Rating Max (Scaled)": df['rating_scaled'].max(),
        "Rating Mean (Scaled)": round(df['rating_scaled'].mean(), 2)
    }
])

summary_p2.to_csv("results/tables/book_crossing_phase2_summary.csv", index=False)
print("Phase 2 complete! Saved results/tables/book_crossing_phase2_summary.csv")