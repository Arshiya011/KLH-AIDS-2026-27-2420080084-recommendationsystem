import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD

def compute_bias_and_popularity_features(train_df: pd.DataFrame, shrink_factor: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Computes global, user, and item bias/popularity signals on the training split strictly.
    Uses Bayesian Shrunk Mean Rating to prevent items with 1-2 reviews from skewing means.
    """
    global_mean = train_df['rating'].mean()
    
    # 1. Item-level Features
    item_stats = train_df.groupby('item_idx').agg(
        item_count=('rating', 'count'),
        item_mean_rating=('rating', 'mean'),
        item_std_rating=('rating', 'std')
    ).reset_index()
    
    item_stats['item_std_rating'] = item_stats['item_std_rating'].fillna(0.0)
    # Bayesian / Shrunk Mean Rating: (C * global_mean + sum_ratings) / (C + n)
    item_stats['item_shrunk_mean'] = (
        (shrink_factor * global_mean) + (item_stats['item_count'] * item_stats['item_mean_rating'])
    ) / (shrink_factor + item_stats['item_count'])
    
    item_stats['item_bias'] = item_stats['item_mean_rating'] - global_mean
    
    # 2. User-level Features
    user_stats = train_df.groupby('user_idx').agg(
        user_count=('rating', 'count'),
        user_mean_rating=('rating', 'mean'),
        user_var_rating=('rating', 'var')
    ).reset_index()
    
    user_stats['user_var_rating'] = user_stats['user_var_rating'].fillna(0.0)
    user_stats['user_bias'] = user_stats['user_mean_rating'] - global_mean
    
    return user_stats, item_stats, global_mean

def extract_svd_latent_features(train_df: pd.DataFrame, n_components: int = 16) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs user-item interaction matrix from training data and extracts K-dimensional SVD latent vectors.
    """
    n_users = train_df['user_idx'].max() + 1
    n_items = train_df['item_idx'].max() + 1
    
    # Create User-Item Sparse Interaction Matrix
    interaction_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    for row in train_df.itertuples():
        interaction_matrix[row.user_idx, row.item_idx] = row.rating_norm
        
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(interaction_matrix)
    item_factors = svd.components_.T
    
    user_svd_cols = [f"user_svd_{i}" for i in range(n_components)]
    item_svd_cols = [f"item_svd_{i}" for i in range(n_components)]
    
    user_svd_df = pd.DataFrame(user_factors, columns=user_svd_cols)
    user_svd_df['user_idx'] = np.arange(n_users)
    
    item_svd_df = pd.DataFrame(item_factors, columns=item_svd_cols)
    item_svd_df['item_idx'] = np.arange(n_items)
    
    return user_svd_df, item_svd_df

def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts date/time signals from interaction timestamps."""
    df_temp = df.copy()
    dt_series = pd.to_datetime(df_temp['timestamp'], unit='s')
    
    df_temp['day_of_week'] = dt_series.dt.dayofweek
    df_temp['hour'] = dt_series.dt.hour
    df_temp['year'] = dt_series.dt.year
    
    # Calculate user tenure (days since first observed interaction)
    user_first_seen = df_temp.groupby('user_idx')['timestamp'].transform('min')
    df_temp['user_tenure_days'] = (df_temp['timestamp'] - user_first_seen) / (24 * 3600)
    
    return df_temp

def extract_movielens_metadata(movies_df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """Processes genre multi-hot vectors and movie release year."""
    movies = movies_df.copy()
    
    # Map raw item_id to contiguous item_idx
    item_map = {str(k): int(v) for k, v in mappings['item_to_idx'].items()}
    movies['item_idx'] = movies['item_id'].map(item_map)
    movies = movies.dropna(subset=['item_idx'])
    movies['item_idx'] = movies['item_idx'].astype(int)
    
    # Extract Release Year
    movies['release_year'] = pd.to_datetime(movies['release_date'], format='%d-%b-%Y', errors='coerce').dt.year
    movies['release_year'] = movies['release_year'].fillna(movies['release_year'].median())
    movies['movie_age_years'] = 2026 - movies['release_year']
    
    genre_cols = [
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    selected_cols = ['item_idx', 'movie_age_years'] + genre_cols
    return movies[selected_cols]

def build_feature_matrix(
    df: pd.DataFrame, 
    user_stats: pd.DataFrame, 
    item_stats: pd.DataFrame, 
    user_svd: pd.DataFrame, 
    item_svd: pd.DataFrame, 
    movie_meta: pd.DataFrame,
    global_mean: float,
    svd_k: int = 16
) -> pd.DataFrame:
    """Merges all feature groups into a unified tabular feature matrix."""
    df_feat = extract_temporal_features(df)
    
    # Join Stats & SVD Factors
    df_feat = df_feat.merge(user_stats, on='user_idx', how='left')
    df_feat = df_feat.merge(item_stats, on='item_idx', how='left')
    df_feat = df_feat.merge(user_svd, on='user_idx', how='left')
    df_feat = df_feat.merge(item_svd, on='item_idx', how='left')
    df_feat = df_feat.merge(movie_meta, on='item_idx', how='left')
    
    # Compute SVD Factor Dot Product (Explicit Latent Similarity Signal)
    user_svd_matrix = df_feat[[f"user_svd_{i}" for i in range(svd_k)]].values
    item_svd_matrix = df_feat[[f"item_svd_{i}" for i in range(svd_k)]].values
    df_feat['svd_dot_product'] = np.sum(user_svd_matrix * item_svd_matrix, axis=1)
    
    # Fill any missing values with training medians/defaults
    df_feat['user_mean_rating'] = df_feat['user_mean_rating'].fillna(global_mean)
    df_feat['item_mean_rating'] = df_feat['item_mean_rating'].fillna(global_mean)
    df_feat['user_bias'] = df_feat['user_bias'].fillna(0.0)
    df_feat['item_bias'] = df_feat['item_bias'].fillna(0.0)
    df_feat = df_feat.fillna(0.0)
    
    return df_feat