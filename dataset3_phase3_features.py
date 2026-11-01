import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

os.makedirs("results/tables", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("Starting Phase 3 Feature Engineering for Dataset 3 (Book-Crossing)...")

# ---------------------------------------------------------
# Step 1: Load Phase 2 Data & Cleaned Metadata
# ---------------------------------------------------------
df_train = pd.read_csv("data/processed/dataset3_phase2_train_rmse.csv")
df_test = pd.read_csv("data/processed/dataset3_phase2_test_rmse.csv")
df_full = pd.read_csv("data/processed/dataset3_phase2_full.csv")

users_df = pd.read_csv("data/processed/dataset3_users_cleaned.csv")
books_df = pd.read_csv("data/processed/dataset3_books_cleaned.csv")

with open("data/processed/dataset3_phase2_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

user_encoder = artifacts['user_encoder']
item_encoder = artifacts['item_encoder']

# ---------------------------------------------------------
# Step 2: Demographics & Location Parsing
# ---------------------------------------------------------
# Parse Country from Location string ("city, state, country")
users_df['country'] = users_df['location'].astype(str).apply(
    lambda x: x.split(',')[-1].strip().lower() if ',' in x else 'unknown'
)
# Clean empty/invalid country strings
users_df['country'] = users_df['country'].replace(['', 'n/a', 'usa', 'united states'], ['unknown', 'unknown', 'usa', 'usa'])

# Map string user_ids to integer user_idx using encoder
users_df = users_df[users_df['user_id'].astype(str).isin(set(user_encoder.classes_.astype(str)))].copy()
users_df['user_idx'] = user_encoder.transform(users_df['user_id'].astype(str))

# Age Buckets
def get_age_bucket(age):
    if pd.isna(age): return 'unknown'
    if age < 18: return '<18'
    elif age <= 25: return '18-25'
    elif age <= 35: return '26-35'
    elif age <= 50: return '36-50'
    else: return '50+'

users_df['age_bucket'] = users_df['age'].apply(get_age_bucket)

# Fill missing Age with training median
train_merged = df_train.merge(users_df[['user_idx', 'age']], on='user_idx', how='left')
age_median = train_merged['age'].median()
users_df['age_clean'] = users_df['age'].fillna(age_median)

# ---------------------------------------------------------
# Step 3: Metadata & Book Age
# ---------------------------------------------------------
books_df = books_df[books_df['item_id'].astype(str).isin(set(item_encoder.classes_.astype(str)))].copy()
books_df['item_idx'] = item_encoder.transform(books_df['item_id'].astype(str))

# Compute Book Age (Reference Year: 2026)
year_median = books_df['year'].median()
books_df['year_clean'] = books_df['year'].fillna(year_median)
books_df['book_age'] = 2026 - books_df['year_clean']

# Clean Author / Publisher strings
books_df['author_clean'] = books_df['author'].fillna('unknown').str.lower().str.strip()
books_df['publisher_clean'] = books_df['publisher'].fillna('unknown').str.lower().str.strip()

# ---------------------------------------------------------
# Step 4: Fit Popularity, Biases, Target Encodings & User-Author Affinity (Train Only)
# ---------------------------------------------------------
global_mean = df_train['rating_scaled'].mean()

# User Biases & Counts
u_stats = df_train.groupby('user_idx')['rating_scaled'].agg(['count', 'mean']).reset_index()
u_stats.columns = ['user_idx', 'user_count', 'user_mean_rating']
u_stats['user_bias'] = u_stats['user_mean_rating'] - global_mean

# Item Biases & Counts
i_stats = df_train.groupby('item_idx')['rating_scaled'].agg(['count', 'mean']).reset_index()
i_stats.columns = ['item_idx', 'item_count', 'item_mean_rating']
i_stats['item_bias'] = i_stats['item_mean_rating'] - global_mean

# Merge author/publisher into train to calculate target encodings
train_with_meta = df_train.merge(books_df[['item_idx', 'author_clean', 'publisher_clean']], on='item_idx', how='left')

# Author / Publisher Frequency & Mean Rating Encodings
author_stats = train_with_meta.groupby('author_clean')['rating_scaled'].agg(['count', 'mean']).reset_index()
author_stats.columns = ['author_clean', 'author_count', 'author_target_enc']

pub_stats = train_with_meta.groupby('publisher_clean')['rating_scaled'].agg(['count', 'mean']).reset_index()
pub_stats.columns = ['publisher_clean', 'publisher_count', 'publisher_target_enc']

# User-Author Affinity (User's average rating for a specific author in train)
u_author_affinity = train_with_meta.groupby(['user_idx', 'author_clean'])['rating_scaled'].mean().reset_index()
u_author_affinity.columns = ['user_idx', 'author_clean', 'user_author_affinity']

# ---------------------------------------------------------
# Step 5: SVD Latent Factors (32 Dimensions on Train Matrix)
# ---------------------------------------------------------
n_users = len(user_encoder.classes_)
n_items = len(item_encoder.classes_)

# Create User-Item Interaction Matrix from Train Split
R = np.zeros((n_users, n_items))
for _, row in df_train.iterrows():
    R[int(row['user_idx']), int(row['item_idx'])] = row['rating_scaled']

# Decompose via SVD (k=16 for 16 user + 16 item factors = 32 features)
k_factors = 16
u_matrix, s_vals, vt_matrix = svds(R, k=k_factors)

u_svd_cols = [f'user_svd_{i}' for i in range(k_factors)]
i_svd_cols = [f'item_svd_{i}' for i in range(k_factors)]

user_svd_df = pd.DataFrame(u_matrix, columns=u_svd_cols)
user_svd_df['user_idx'] = range(n_users)

item_svd_df = pd.DataFrame(vt_matrix.T, columns=i_svd_cols)
item_svd_df['item_idx'] = range(n_items)

# ---------------------------------------------------------
# Step 6: Assemble Complete Feature Pipeline Function
# ---------------------------------------------------------
def attach_features(data_df):
    # 1. Merge User Demographics & Biases
    df_out = data_df.merge(users_df[['user_idx', 'age_clean', 'age_bucket', 'country']], on='user_idx', how='left')
    df_out = df_out.merge(u_stats, on='user_idx', how='left')
    
    # 2. Merge Book Metadata, Age & Biases
    df_out = df_out.merge(books_df[['item_idx', 'book_age', 'author_clean', 'publisher_clean']], on='item_idx', how='left')
    df_out = df_out.merge(i_stats, on='item_idx', how='left')
    
    # 3. Merge Target Encodings
    df_out = df_out.merge(author_stats, on='author_clean', how='left')
    df_out = df_out.merge(pub_stats, on='publisher_clean', how='left')
    
    # 4. Merge User-Author Affinity
    df_out = df_out.merge(u_author_affinity, on=['user_idx', 'author_clean'], how='left')
    
    # 5. Merge SVD Latent Factors
    df_out = df_out.merge(user_svd_df, on='user_idx', how='left')
    df_out = df_out.merge(item_svd_df, on='item_idx', how='left')
    
    # 6. Compute SVD Dot Product
    u_svd_vals = df_out[u_svd_cols].values
    i_svd_vals = df_out[i_svd_cols].values
    df_out['svd_dot_product'] = np.sum(u_svd_vals * i_svd_vals, axis=1)
    
    # Impute missing values resulting from test cold-starts / unseen keys
    df_out['user_count'] = df_out['user_count'].fillna(0)
    df_out['user_bias'] = df_out['user_bias'].fillna(0)
    df_out['user_mean_rating'] = df_out['user_mean_rating'].fillna(global_mean)
    
    df_out['item_count'] = df_out['item_count'].fillna(0)
    df_out['item_bias'] = df_out['item_bias'].fillna(0)
    df_out['item_mean_rating'] = df_out['item_mean_rating'].fillna(global_mean)
    
    df_out['author_count'] = df_out['author_count'].fillna(0)
    df_out['author_target_enc'] = df_out['author_target_enc'].fillna(global_mean)
    
    df_out['publisher_count'] = df_out['publisher_count'].fillna(0)
    df_out['publisher_target_enc'] = df_out['publisher_target_enc'].fillna(global_mean)
    
    df_out['user_author_affinity'] = df_out['user_author_affinity'].fillna(global_mean)
    
    return df_out

print("Transforming training split...")
df_train_feat = attach_features(df_train)

print("Transforming test split...")
df_test_feat = attach_features(df_test)

# ---------------------------------------------------------
# Step 7: Save Feature Stores & Summary Table
# ---------------------------------------------------------
df_train_feat.to_csv("data/processed/dataset3_phase3_train_features.csv", index=False)
df_test_feat.to_csv("data/processed/dataset3_phase3_test_features.csv", index=False)

# Summary Output
feature_list = [c for c in df_train_feat.columns if c not in ['user_id', 'item_id', 'rating', 'rating_raw']]

summary_p3 = pd.DataFrame([
    {
        "Dataset": "Book-Crossing",
        "Total Feature Count": len(feature_list),
        "Popularity & Bias Features": 6,
        "Demographic Features": 4, # age_clean, age_bucket, country, user_author_affinity
        "Metadata & Target Encodings": 6, # book_age, author_count, author_target_enc, pub_count, pub_target_enc
        "SVD Latent Factor Features": 33, # 16 user + 16 item + 1 dot product
        "Temporal Features": "Omitted (Missing Timestamps)",
        "Genre Features": "Substituted by Author/Publisher Metadata"
    }
])

summary_p3.to_csv("results/tables/book_crossing_phase3_summary.csv", index=False)
print("Phase 3 complete! Saved results/tables/book_crossing_phase3_summary.csv")