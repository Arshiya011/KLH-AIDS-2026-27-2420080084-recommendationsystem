import pandas as pd
from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, SEED
from src.visualization import plot_rating_metrics_comparison, plot_ranking_metrics_comparison
from src.explainability import generate_shap_explainability
from src.models.ml_models import train_xgboost_regressor

def main():
    print("======================================================================")
    print("         PHASE 6 & 7: VISUALIZATIONS & SHAP EXPLAINABILITY            ")
    print("======================================================================\n")
    
    # 1. Load Master Metrics
    master_df = pd.read_csv(RESULTS_DIR / "tables" / "movielens_100k_master_metrics.csv", index_col="Model")
    
    # 2. Generate Plots
    print("[INFO] Generating master comparative visual plots...")
    plot_rating_metrics_comparison(master_df, RESULTS_DIR / "figures" / "rating_prediction_comparison.png")
    plot_ranking_metrics_comparison(master_df, RESULTS_DIR / "figures" / "ranking_metrics_comparison.png")
    
    # 3. Fit XGBoost & Run SHAP
    print("[INFO] Running SHAP TreeExplainer feature importance analysis...")
    train_feat = pd.read_parquet(PROCESSED_DATA_DIR / "movielens_100k_features_train.parquet")
    xgb_model, feature_cols = train_xgboost_regressor(train_feat, seed=SEED)
    
    generate_shap_explainability(xgb_model, train_feat, feature_cols, RESULTS_DIR / "figures")
    
    print("\n[SUCCESS] Visualizations and SHAP feature importance plots saved to results/figures/")

if __name__ == "__main__":
    main()