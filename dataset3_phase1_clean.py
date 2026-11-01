import os
import pandas as pd
import numpy as np

os.makedirs("results/tables", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ---------------------------------------------------------
# Step 1: Detect Data Directory and Filenames
# ---------------------------------------------------------
possible_dirs = [os.path.join("data", "raw"), "data", "."]
DATA_DIR = None

for d in possible_dirs:
    if os.path.exists(os.path.join(d, "BX-Books.csv")) or os.path.exists(os.path.join(d, "Books.csv")):
        DATA_DIR = d
        break

if DATA_DIR is None:
    raise FileNotFoundError("Could not find Book-Crossing CSV files in 'data/raw', 'data', or project root.")

books_file = "BX-Books.csv" if os.path.exists(os.path.join(DATA_DIR, "BX-Books.csv")) else "Books.csv"
users_file = "BX-Users.csv" if os.path.exists(os.path.join(DATA_DIR, "BX-Users.csv")) else "Users.csv"
ratings_file = "BX-Book-Ratings.csv" if os.path.exists(os.path.join(DATA_DIR, "BX-Book-Ratings.csv")) else "Ratings.csv"

print(f"Loading files from: {DATA_DIR}")

# ---------------------------------------------------------
# Step 2: Parsing CSVs
# ---------------------------------------------------------
books_df = pd.read_csv(
    os.path.join(DATA_DIR, books_file), 
    sep=';', 
    encoding='latin-1', 
    on_bad_lines='skip', 
    dtype=str
)

users_df = pd.read_csv(
    os.path.join(DATA_DIR, users_file), 
    sep=';', 
    encoding='latin-1', 
    on_bad_lines='skip', 
    dtype=str
)

ratings_df = pd.read_csv(
    os.path.join(DATA_DIR, ratings_file), 
    sep=';', 
    encoding='latin-1', 
    on_bad_lines='skip', 
    dtype=str
)

# Rename to unified schema standards
books_df = books_df.rename(columns={
    'ISBN': 'item_id', 
    'Book-Title': 'title', 
    'Book-Author': 'author', 
    'Year-Of-Publication': 'year', 
    'Publisher': 'publisher'
})
users_df = users_df.rename(columns={'User-ID': 'user_id', 'Location': 'location', 'Age': 'age'})
ratings_df = ratings_df.rename(columns={'User-ID': 'user_id', 'ISBN': 'item_id', 'Book-Rating': 'rating'})

ratings_df['rating'] = pd.to_numeric(ratings_df['rating'], errors='coerce')

# Record Raw Metrics
raw_user_cnt = ratings_df['user_id'].nunique()
raw_item_cnt = ratings_df['item_id'].nunique()
raw_interaction_cnt = len(ratings_df)
raw_sparsity = 100 * (1 - raw_interaction_cnt / (raw_user_cnt * raw_item_cnt))
raw_min_rating = ratings_df['rating'].min()
raw_max_rating = ratings_df['rating'].max()
raw_mean_rating = ratings_df['rating'].mean()

# ---------------------------------------------------------
# Step 3: Cleaning Implicit Zeros & Outliers
# ---------------------------------------------------------
# Drop implicit zero ratings
df_explicit = ratings_df[ratings_df['rating'] > 0].copy()

# Clean Users Age
users_df['age'] = pd.to_numeric(users_df['age'], errors='coerce')
users_df.loc[(users_df['age'] < 5) | (users_df['age'] > 100), 'age'] = np.nan

# Clean Book Publication Year
books_df['year'] = pd.to_numeric(books_df['year'], errors='coerce')
books_df.loc[(books_df['year'] < 1800) | (books_df['year'] > 2026), 'year'] = np.nan

# Keep interactions matching validated metadata keys
valid_users = users_df['user_id'].unique()
valid_books = books_df['item_id'].unique()
df_explicit = df_explicit[
    df_explicit['user_id'].isin(valid_users) & 
    df_explicit['item_id'].isin(valid_books)
]

# ---------------------------------------------------------
# Step 4: Iterative Core Filtering (5-Core)
# ---------------------------------------------------------
def apply_k_core(data, k=5):
    filtered = data.copy()
    while True:
        u_counts = filtered['user_id'].value_counts()
        i_counts = filtered['item_id'].value_counts()
        
        valid_u = u_counts[u_counts >= k].index
        valid_i = i_counts[i_counts >= k].index
        
        new_filtered = filtered[filtered['user_id'].isin(valid_u) & filtered['item_id'].isin(valid_i)]
        if len(new_filtered) == len(filtered):
            break
        filtered = new_filtered
    return filtered

df_5core = apply_k_core(df_explicit, k=5)

# Record Cleaned 5-Core Metrics
clean_user_cnt = df_5core['user_id'].nunique()
clean_item_cnt = df_5core['item_id'].nunique()
clean_interaction_cnt = len(df_5core)
clean_sparsity = 100 * (1 - clean_interaction_cnt / (clean_user_cnt * clean_item_cnt))
clean_min_rating = df_5core['rating'].min()
clean_max_rating = df_5core['rating'].max()
clean_mean_rating = df_5core['rating'].mean()

# Save cleaned dataset for Phase 2
df_5core.to_csv("data/processed/dataset3_cleaned_5core.csv", index=False)

# Save cleaned metadata artifacts for Phase 3
users_df.to_csv("data/processed/dataset3_users_cleaned.csv", index=False)
books_df.to_csv("data/processed/dataset3_books_cleaned.csv", index=False)

# ---------------------------------------------------------
# Step 5: Data Quality Summary Table
# ---------------------------------------------------------
summary_df = pd.DataFrame([
    {
        "Stage": "1. Raw Data",
        "Interactions": raw_interaction_cnt,
        "Rows": raw_interaction_cnt,
        "Unique Users": raw_user_cnt,
        "Unique Items": raw_item_cnt,
        "Sparsity %": f"{raw_sparsity:.2f}%",
        "Density %": f"{100 - raw_sparsity:.2f}%",
        "Min Rating": raw_min_rating,
        "Max Rating": raw_max_rating,
        "Mean Rating": f"{raw_mean_rating:.2f}"
    },
    {
        "Stage": "2. Cleaned (5-Core)",
        "Interactions": clean_interaction_cnt,
        "Rows": clean_interaction_cnt,
        "Unique Users": clean_user_cnt,
        "Unique Items": clean_item_cnt,
        "Sparsity %": f"{clean_sparsity:.2f}%",
        "Density %": f"{100 - clean_sparsity:.2f}%",
        "Min Rating": clean_min_rating,
        "Max Rating": clean_max_rating,
        "Mean Rating": f"{clean_mean_rating:.2f}"
    }
])

summary_df.to_csv("results/tables/book_crossing_phase1_summary.csv", index=False)
print("Phase 1 complete! Saved results/tables/book_crossing_phase1_summary.csv")