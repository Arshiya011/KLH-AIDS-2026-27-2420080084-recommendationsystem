import shap
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_shap_explainability(model: xgb.XGBRegressor, train_features: pd.DataFrame, feature_cols: list[str], save_dir: Path):
    """
    Computes SHAP TreeExplainer feature importance and beeswarm summary plot.
    """
    X_sample = train_features[feature_cols].sample(n=min(1000, len(train_features)), random_state=42)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    # 1. SHAP Feature Importance Bar Plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.title("SHAP Global Feature Importance (XGBoost)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "shap_feature_importance.png", dpi=300)
    plt.close()
    
    # 2. SHAP Summary Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title("SHAP Beeswarm Feature Impact", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "shap_beeswarm.png", dpi=300)
    plt.close()