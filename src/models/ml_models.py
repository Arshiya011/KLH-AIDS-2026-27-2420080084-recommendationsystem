import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
from src.config import SEED

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Excludes ID, timestamp, and target columns from tabular inputs."""
    ignore_cols = {
        'user_id', 'item_id', 'user_idx', 'item_idx', 
        'rating', 'rating_norm', 'timestamp', 'release_date'
    }
    return [col for col in df.columns if col not in ignore_cols]

def train_xgboost_regressor(train_df: pd.DataFrame, seed: int = SEED) -> tuple[xgb.XGBRegressor, list[str]]:
    feature_cols = get_feature_columns(train_df)
    X_train = train_df[feature_cols]
    y_train = train_df['rating']
    
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, feature_cols

def train_lightgbm_regressor(train_df: pd.DataFrame, seed: int = SEED) -> tuple[lgb.LGBMRegressor, list[str]]:
    feature_cols = get_feature_columns(train_df)
    X_train = train_df[feature_cols]
    y_train = train_df['rating']
    
    model = lgb.LGBMRegressor(
        n_estimators=150,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, feature_cols