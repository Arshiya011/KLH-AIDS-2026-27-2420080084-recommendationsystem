import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

sns.set_theme(style="whitegrid")

def plot_rating_metrics_comparison(master_df: pd.DataFrame, save_path: Path):
    """Bar chart comparing RMSE and MAE across models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot = master_df[['RMSE', 'MAE']].reset_index().melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    sns.barplot(data=df_plot, x='Model', y='Score', hue='Metric', palette='Set2', ax=ax)
    ax.set_title("Rating Prediction Performance (Lower is Better)", fontsize=14, fontweight='bold')
    ax.set_ylim(0.5, 1.1)
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_ranking_metrics_comparison(master_df: pd.DataFrame, save_path: Path):
    """Line plot comparing HR@K and NDCG@K across K={5, 10, 20}."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    k_vals = [5, 10, 20]
    
    # HR@K Plot
    for model in master_df.index:
        hr_vals = [master_df.loc[model, f"HR@{k}"] for k in k_vals]
        axes[0].plot(k_vals, hr_vals, marker='o', linewidth=2, label=model)
        
    axes[0].set_title("Hit Ratio @ K (Higher is Better)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Hit Ratio")
    axes[0].set_xticks(k_vals)
    axes[0].legend()
    
    # NDCG@K Plot
    for model in master_df.index:
        ndcg_vals = [master_df.loc[model, f"NDCG@{k}"] for k in k_vals]
        axes[1].plot(k_vals, ndcg_vals, marker='s', linewidth=2, label=model)
        
    axes[1].set_title("NDCG @ K (Higher is Better)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("NDCG Score")
    axes[1].set_xticks(k_vals)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()