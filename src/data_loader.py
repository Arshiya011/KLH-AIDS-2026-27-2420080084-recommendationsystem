import os
import ssl
import zipfile
import urllib.request
import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_DIR

def download_movielens_100k(target_dir: Path = RAW_DATA_DIR) -> Path:
    """Downloads and extracts MovieLens 100K if not already present."""
    # Ensure raw data directory exists on disk
    target_dir.mkdir(parents=True, exist_ok=True)
    
    ml_dir = target_dir / "ml-100k"
    if not ml_dir.exists():
        print("[INFO] Downloading MovieLens 100K dataset...")
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        zip_path = target_dir / "ml-100k.zip"
        
        # Bypass SSL certificate verification on Windows
        ssl_context = ssl._create_unverified_context()
        
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        
        with urllib.request.urlopen(req, context=ssl_context) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        if zip_path.exists():
            os.remove(zip_path)
            
        print("[INFO] MovieLens 100K downloaded and unzipped successfully.")
    else:
        print("[INFO] MovieLens 100K dataset already present locally.")
    return ml_dir

def load_movielens_100k(data_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw MovieLens 100K and enforces unified baseline schema:
    user_id (str), item_id (str), rating (float), timestamp (int).
    """
    ml_dir = download_movielens_100k(data_dir)
    
    # 1. Primary Ratings (u.data)
    ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(
        ml_dir / "u.data", 
        sep="\t", 
        names=ratings_cols, 
        engine="python"
    )
    
    ratings_df['user_id'] = ratings_df['user_id'].astype(str)
    ratings_df['item_id'] = ratings_df['item_id'].astype(str)
    ratings_df['rating'] = ratings_df['rating'].astype(float)
    ratings_df['timestamp'] = ratings_df['timestamp'].astype(int)
    
    # 2. Movie Metadata (u.item)
    movie_cols = [
        'item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    movies_df = pd.read_csv(
        ml_dir / "u.item", 
        sep="|", 
        names=movie_cols, 
        encoding="latin-1", 
        engine="python"
    )
    movies_df['item_id'] = movies_df['item_id'].astype(str)
    
    # 3. User Metadata (u.user)
    user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    users_df = pd.read_csv(
        ml_dir / "u.user", 
        sep="|", 
        names=user_cols, 
        encoding="latin-1", 
        engine="python"
    )
    users_df['user_id'] = users_df['user_id'].astype(str)
    
    return ratings_df, movies_df, users_df