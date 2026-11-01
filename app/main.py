import json
import shap
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb

# Set Page Configuration
st.set_page_config(
    page_title="RecSys Comparative Benchmarking",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base Paths Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.title("🎛️ Control Center")
dataset_choice = st.sidebar.selectbox(
    "Dataset Track", 
    ["MovieLens 100K", "Amazon Fine Foods (Track 2)", "Book-Crossing (Track 3)"]
)
page_choice = st.sidebar.radio(
    "Navigation", 
    ["1. Dataset Overview", "2. Architecture & Features", "3. Executive Performance Comparison", "4. Search & Explain Recommendations"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Comparative Recommendation Engineering Framework")

# ---------------------------------------------------------
# DYNAMIC FILE PATH MAPPINGS
# ---------------------------------------------------------
def get_dataset_paths(choice):
    if choice == "MovieLens 100K":
        return {
            "master_metrics": TABLES_DIR / "movielens_100k_master_metrics.csv",
            "phase1_summary": TABLES_DIR / "movielens_100k_phase1_summary.csv",
            "test_preds": PROCESSED_DATA_DIR / "movielens_100k_test_predictions.parquet",
            "train_feat": PROCESSED_DATA_DIR / "movielens_100k_features_train.parquet",
            "test_feat": PROCESSED_DATA_DIR / "movielens_100k_features_test.parquet",
            "mappings": PROCESSED_DATA_DIR / "movielens_100k_mappings.json",
            "item_key": "Movie Title",
            "rating_scale": "1 - 5"
        }
    elif choice == "Amazon Fine Foods (Track 2)":
        return {
            "master_metrics": TABLES_DIR / "amazon_foods_master_metrics.csv",
            "phase1_summary": TABLES_DIR / "amazon_foods_phase1_summary.csv",
            "test_preds": PROCESSED_DATA_DIR / "amazon_foods_test_predictions.parquet",
            "train_feat": PROCESSED_DATA_DIR / "amazon_foods_features_train.parquet",
            "test_feat": PROCESSED_DATA_DIR / "amazon_foods_features_test.parquet",
            "mappings": PROCESSED_DATA_DIR / "amazon_foods_mappings.json",
            "item_key": "Product ID",
            "rating_scale": "1 - 5"
        }
    else:  # Book-Crossing (Track 3)
        return {
            "master_metrics": TABLES_DIR / "book_crossing_master_metrics.csv",
            "phase1_summary": TABLES_DIR / "book_crossing_phase1_summary.csv",
            "test_preds": PROCESSED_DATA_DIR / "dataset3_phase2_test_rmse.csv",
            "train_feat": PROCESSED_DATA_DIR / "dataset3_phase3_train_features.csv",
            "test_feat": PROCESSED_DATA_DIR / "dataset3_phase3_test_features.csv",
            "mappings": PROCESSED_DATA_DIR / "dataset3_phase2_artifacts.pkl",
            "item_key": "Book Title / ISBN",
            "rating_scale": "1 - 5 (Rescaled from 1-10)"
        }

paths = get_dataset_paths(dataset_choice)

# ---------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# ---------------------------------------------------------
@st.cache_data
def load_master_metrics(path):
    if path.exists():
        return pd.read_csv(path, index_col="Model")
    return None

@st.cache_data
def load_phase1_summary(path):
    if path.exists():
        df = pd.read_csv(path)
        if "Stage" in df.columns:
            return df.set_index("Stage")
        return df
    return None

@st.cache_data
def load_test_predictions(path):
    if path.exists():
        if str(path).endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    return None

@st.cache_data
def load_feature_matrix(train_path, test_path):
    if train_path.exists() and test_path.exists():
        if str(train_path).endswith(".parquet"):
            return pd.read_parquet(train_path), pd.read_parquet(test_path)
        return pd.read_csv(train_path), pd.read_csv(test_path)
    return None, None

@st.cache_data
def load_movie_titles():
    item_path = BASE_DIR / "data" / "raw" / "ml-100k" / "u.item"
    titles = {}
    if item_path.exists():
        try:
            with open(item_path, "r", encoding="latin-1") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 2:
                        titles[int(parts[0])] = parts[1]
        except Exception:
            pass
    return titles

@st.cache_resource
def load_ncf_model(dataset_track):
    model_filename = "ncf_neumf_best.pt" if dataset_track == "MovieLens 100K" else f"ncf_{dataset_track.lower().split()[0]}_best.pt"
    ncf_path = MODELS_DIR / model_filename
    if ncf_path.exists():
        try:
            model = torch.load(ncf_path, map_location=torch.device('cpu'))
            model.eval()
            return model
        except Exception:
            return None
    return None

@st.cache_resource
def fit_tree_models_and_explainers(X_train):
    ignore_cols = {
        'user_id', 'item_id', 'user_idx', 'item_idx', 'rating', 'rating_norm', 'rating_scaled',
        'timestamp', 'release_date', 'title', 'author', 'publisher', 'location', 'author_clean',
        'publisher_clean', 'age_bucket', 'country'
    }
    feature_cols = [col for col in X_train.columns if col not in ignore_cols]
    
    X_train_clean = X_train[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
    
    target_col = 'rating_scaled' if 'rating_scaled' in X_train.columns else 'rating'
    y_train_clean = pd.to_numeric(X_train[target_col], errors='coerce').fillna(3.0).astype(np.float32)
    
    model_xgb = xgb.XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
    model_xgb.fit(X_train_clean, y_train_clean)
    explainer_xgb = shap.TreeExplainer(model_xgb)
    
    model_lgb = lgb.LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)
    model_lgb.fit(X_train_clean, y_train_clean)
    explainer_lgb = shap.TreeExplainer(model_lgb)
    
    return {
        "XGBoost": (model_xgb, explainer_xgb),
        "LightGBM": (model_lgb, explainer_lgb)
    }, feature_cols

def highlight_best_metrics(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    
    if "RMSE" in df.columns:
        min_val = df["RMSE"].min()
        styles["RMSE"] = np.where(df["RMSE"] == min_val, 'background-color: #1e4620; color: #ffffff; font-weight: bold;', '')
        
    for col in [c for c in df.columns if c != "RMSE"]:
        max_val = df[col].max()
        styles[col] = np.where(df[col] == max_val, 'background-color: #1e4620; color: #ffffff; font-weight: bold;', '')
        
    return styles

# ---------------------------------------------------------
# PAGE 1: DATASET OVERVIEW
# ---------------------------------------------------------
if page_choice == "1. Dataset Overview":
    st.title("📊 Dataset Quality & Transformation")
    st.write(f"Summary statistics for **{dataset_choice}** after iterative 5-Core pruning.")
    
    summary_df = load_phase1_summary(paths["phase1_summary"])
    if summary_df is not None:
        try:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Interactions", f"{int(summary_df.loc['2. Cleaned (5-Core)', 'Rows']):,}")
            c2.metric("Users", f"{int(summary_df.loc['2. Cleaned (5-Core)', 'Unique Users']):,}")
            c3.metric("Items", f"{int(summary_df.loc['2. Cleaned (5-Core)', 'Unique Items']):,}")
            c4.metric("Density", f"{summary_df.loc['2. Cleaned (5-Core)', 'Density %']}")
        except Exception:
            pass
            
        st.subheader("Data Cleaning Summary")
        st.dataframe(summary_df, width="stretch")
    else:
        st.warning(f"Summary table not found at `{paths['phase1_summary']}`.")

# ---------------------------------------------------------
# PAGE 2: ARCHITECTURE & FEATURES
# ---------------------------------------------------------
elif page_choice == "2. Architecture & Features":
    st.title("⚙️ Architecture & Feature Store")
    
    st.subheader("1. System Data Flow")
    
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=["Raw Data", "5-Core Filtered", "80/20 Rating Split", "LOO Ranking Split", "Feature Store", "XGBoost / LightGBM", "NCF Deep Learning", "Evaluation Engine"],
            color=["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"],
            customdata=["100,000 Raw Ratings", "99,287 Pruned Ratings", "79,430 Train Interactions", "19,857 LOO Items", "73 Engineered Features", "GBDT Models", "Neural Net", "Master Metric Benchmarking"],
            hovertemplate="<b>%{label}</b><br>Info: %{customdata}<extra></extra>"
        ),
        link=dict(
            source=[0, 1, 1, 2, 3, 4, 4, 5, 6],
            target=[1, 2, 3, 4, 4, 5, 6, 7, 7],
            value=[100000, 79430, 19857, 79430, 98344, 79430, 79430, 79430, 79430],
            hovertemplate="Flow from %{source.label} to %{target.label}: <b>%{value:,}</b> rows<extra></extra>"
        )
    )])
    fig_sankey.update_layout(
        title_text=f"Data flows left→right ({dataset_choice}): Raw Data → Cleaning → Train/Test Split → Feature Store → Models → Evaluation",
        font_size=12
    )
    st.plotly_chart(fig_sankey, width="stretch")
    
    st.markdown("---")
    st.subheader("2. Interactive Feature Store Explorer")
    
    train_feat, _ = load_feature_matrix(paths["train_feat"], paths["test_feat"])
    if train_feat is not None:
        if dataset_choice == "Book-Crossing (Track 3)":
            feature_groups = {
                "Popularity & Biases": ["item_count", "item_mean_rating", "user_bias", "item_bias", "user_count", "user_mean_rating"],
                "Latent Factors (SVD Matrix Factorization)": [c for c in train_feat.columns if "svd" in c],
                "Demographics & Metadata": [c for c in train_feat.columns if c in ["age_clean", "book_age", "author_count", "author_target_enc", "publisher_count", "publisher_target_enc", "user_author_affinity"]]
            }
        else:
            feature_groups = {
                "Popularity & Biases": ["item_count", "item_mean_rating", "item_shrunk_mean", "user_bias", "item_bias", "user_count", "user_mean_rating"],
                "Latent Factors (SVD Matrix Factorization)": [c for c in train_feat.columns if "svd" in c],
                "Metadata & Genres (Content Signals)": [c for c in train_feat.columns if c in ["movie_age_years", "release_date_year"] or any(g in c for g in ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])],
                "Temporal Signals": [c for c in train_feat.columns if c in ["day_of_week", "hour", "user_tenure_days", "year"]]
            }
            
        selected_group = st.selectbox("Select Feature Engineering Category:", list(feature_groups.keys()))
        selected_cols = [c for c in feature_groups[selected_group] if c in train_feat.columns]
        
        base_id_cols = [c for c in ["user_idx", "item_idx", "rating", "rating_scaled"] if c in train_feat.columns]
        display_cols = base_id_cols + selected_cols
        
        st.write(f"### Feature Table Preview: `{selected_group}` ({len(selected_cols)} features)")
        st.dataframe(train_feat[display_cols].head(15).round(2), width="stretch")
        
        if len(selected_cols) > 0:
            plot_feature = st.selectbox("Select specific feature from this group to plot:", selected_cols)
            fig_feat = px.histogram(train_feat, x=plot_feature, nbins=30, title=f"Value Distribution: '{plot_feature}'", color_discrete_sequence=['#00CC96'])
            st.plotly_chart(fig_feat, width="stretch")
    else:
        st.warning(f"Feature matrix not found for {dataset_choice}.")

# ---------------------------------------------------------
# PAGE 3: MODEL COMPARISON
# ---------------------------------------------------------
elif page_choice == "3. Executive Performance Comparison":
    st.title("📈 Model Performance Dashboard")
    
    master_df = load_master_metrics(paths["master_metrics"])
    if master_df is not None:
        master_df = master_df.round(4)
        
        col_view, col_model_filter = st.columns([1, 1])
        with col_view:
            view_mode = st.radio("Metric Scope:", ["Concise Executive View (Recommended)", "Full Breakdown (16 Metrics)"], horizontal=True)
            
        with col_model_filter:
            model_family = st.selectbox(
                "Select Model Paradigm / Family to Compare:",
                ["All Models", "Baseline Models (Global Mean, Item-KNN)", "Gradient Boosted Decision Trees (XGBoost, LightGBM)", "Deep Learning (NCF NeuMF)"]
            )
            
        filtered_df = master_df.copy()
        if model_family == "Baseline Models (Global Mean, Item-KNN)":
            filtered_df = filtered_df[filtered_df.index.isin(["Global Mean", "Item-KNN"])]
        elif model_family == "Gradient Boosted Decision Trees (XGBoost, LightGBM)":
            filtered_df = filtered_df[filtered_df.index.isin(["XGBoost", "LightGBM"])]
        elif model_family == "Deep Learning (NCF NeuMF)":
            filtered_df = filtered_df[filtered_df.index.isin(["NCF (NeuMF)"])]
            
        st.markdown("---")
        
        display_cols = ["RMSE", "HR@10", "NDCG@10"] if view_mode == "Concise Executive View (Recommended)" else filtered_df.columns
        valid_cols = [c for c in display_cols if c in filtered_df.columns]
        
        st.subheader(f"Evaluation Matrix — {dataset_choice} ({model_family})")
        st.dataframe(filtered_df[valid_cols].style.apply(highlight_best_metrics, axis=None), width="stretch")
            
        if "Item-KNN" in filtered_df.index and "Global Mean" in filtered_df.index:
            if filtered_df.loc["Item-KNN", "RMSE"] > filtered_df.loc["Global Mean", "RMSE"]:
                st.info("⚠️ Item-KNN underperforms the naive Global Mean baseline on rating accuracy in this dataset — likely a cold-start/sparsity effect worth noting in your write-up.")
            
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rating Accuracy (RMSE - Lower is Better)")
            fig_bar = px.bar(filtered_df.reset_index(), x="Model", y="RMSE", color="Model", title="RMSE Comparison")
            st.plotly_chart(fig_bar, width="stretch")
            
        with col2:
            st.subheader("Ranking Success (HR@10 & NDCG@10)")
            rank_cols = [c for c in ["HR@10", "NDCG@10"] if c in filtered_df.columns]
            fig_bar_rank = px.bar(filtered_df.reset_index(), x="Model", y=rank_cols, barmode="group", title="Top-10 Ranking Performance")
            st.plotly_chart(fig_bar_rank, width="stretch")
    else:
        st.warning(f"Master metrics file not found at `{paths['master_metrics']}`.")

# ---------------------------------------------------------
# PAGE 4: SEARCH & EXPLAIN RECOMMENDATIONS
# ---------------------------------------------------------
elif page_choice == "4. Search & Explain Recommendations":
    st.title("🔎 Search & Explain Model Recommendations")
    
    st.markdown("💡 **Instruction:** Select a User ID from the search box below, then **click any item row in the table** to inspect its live SHAP feature attribution breakdown.")
    
    test_preds = load_test_predictions(paths["test_preds"])
    train_feat, test_feat = load_feature_matrix(paths["train_feat"], paths["test_feat"])
    movie_titles = load_movie_titles() if dataset_choice == "MovieLens 100K" else {}
    ncf_model = load_ncf_model(dataset_choice)
    
    if test_preds is not None and train_feat is not None and test_feat is not None:
        models_dict, feature_cols = fit_tree_models_and_explainers(train_feat)
        
        active_user_indices = sorted(test_preds["user_idx"].unique().tolist())
        
        target_user = st.selectbox(
            "Select Active Test User ID (Index):",
            options=active_user_indices,
            index=0
        )
        
        user_df = test_preds[test_preds["user_idx"] == target_user].copy()
        
        if len(user_df) > 0:
            user_df[paths["item_key"]] = user_df["item_idx"].apply(
                lambda x: f"{movie_titles.get(int(x), f'Item #{x}')} [#{x}]"
            )
            
            display_df = user_df[[paths["item_key"]] + [c for c in ["rating", "rating_scaled", "pred_xgb", "pred_lgb", "pred_ncf", "pred_knn"] if c in user_df.columns]]
            
            st.write(f"**Multi-Model Recommendations for User Index: `{target_user}`**")
            
            selection_event = st.dataframe(
                display_df.round(2),
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            selected_row_idx = 0
            if selection_event and "rows" in selection_event.get("selection", {}):
                rows = selection_event["selection"]["rows"]
                if len(rows) > 0:
                    selected_row_idx = rows[0]
                    
            selected_item_idx = int(user_df.iloc[selected_row_idx]["item_idx"])
            selected_item_title = user_df.iloc[selected_row_idx][paths["item_key"]]
            
            st.markdown("---")
            
            tab_local, tab_global = st.tabs(["🎯 Local Per-Prediction Explanation", "🌐 Overall Model Behavior (Global SHAP)"])
            
            # TAB 1: LOCAL EXPLANATION
            with tab_local:
                col_info, col_model_select = st.columns([2, 1])
                with col_info:
                    st.markdown(f"### Reasoning for: `{selected_item_title}`")
                with col_model_select:
                    selected_model_name = st.selectbox(
                        "Explain Model Paradigm:", 
                        options=["XGBoost", "LightGBM", "NCF (NeuMF) - Deep Learning", "Item-KNN - Collaborative"]
                    )
                    
                st.markdown("---")
                
                # NON-TREE MODELS (NCF & Item-KNN)
                if selected_model_name not in models_dict:
                    sync_row = user_df.iloc[selected_row_idx]
                    col_key = "pred_ncf" if "NCF" in selected_model_name else "pred_knn"
                    
                    live_pred = None
                    if selected_model_name == "NCF (NeuMF) - Deep Learning" and ncf_model is not None:
                        try:
                            with torch.no_grad():
                                u_tensor = torch.tensor([target_user], dtype=torch.long)
                                i_tensor = torch.tensor([selected_item_idx], dtype=torch.long)
                                live_pred = float(ncf_model(u_tensor, i_tensor).item())
                            st.success(f"⚡ Live PyTorch forward pass computed for {dataset_choice}")
                        except Exception as e:
                            st.warning(f"Live inference failed ({e}); showing cached prediction instead.")
                            
                    pred_val_non_tree = live_pred if live_pred is not None else sync_row.get(col_key, 0.0)
                    actual_rating_col = 'rating_scaled' if 'rating_scaled' in sync_row else 'rating'
                    actual_val_non_tree = sync_row.get(actual_rating_col, 0.0)
                    error_non_tree = pred_val_non_tree - actual_val_non_tree
                    
                    if live_pred is None:
                        st.info(f"ℹ️ `{selected_model_name}` uses a non-tree architecture — native TreeSHAP feature attribution isn't available. Showing model agreement instead:")
                        
                    col_left, col_right = st.columns([1, 1.6])
                    
                    with col_left:
                        st.metric(
                            label=f"{selected_model_name} Predicted Score",
                            value=f"{pred_val_non_tree:.2f} / 5.0",
                            delta=f"{error_non_tree:+.2f} vs actual",
                            delta_color="off"
                        )
                        st.metric("Actual User Score", f"{actual_val_non_tree:.2f} / 5.0")
                        
                    with col_right:
                        st.markdown("**Model Agreement for this item:**")
                        available_preds = {
                            "XGBoost": sync_row.get("pred_xgb", 0.0),
                            "LightGBM": sync_row.get("pred_lgb", 0.0),
                            "NCF (NeuMF)": pred_val_non_tree if "NCF" in selected_model_name else sync_row.get("pred_ncf", 0.0),
                            "Item-KNN": sync_row.get("pred_knn", 0.0)
                        }
                        compare_df = pd.DataFrame(list(available_preds.items()), columns=["Model", "Score"])
                        fig_compare = px.bar(
                            compare_df, x="Model", y="Score", color="Model",
                            range_y=[0, 5], title=None
                        )
                        fig_compare.add_hline(
                            y=actual_val_non_tree, line_dash="dash", line_color="white",
                            annotation_text="Actual Rating", annotation_position="top left"
                        )
                        fig_compare.update_layout(showlegend=False, height=320, margin=dict(t=10))
                        st.plotly_chart(fig_compare, use_container_width=True)
                        
                # TREE-BASED MODELS (XGBoost & LightGBM)
                else:
                    model_obj, explainer_obj = models_dict[selected_model_name]
                    match_feat = test_feat[(test_feat["user_idx"] == target_user) & (test_feat["item_idx"] == selected_item_idx)]
                    
                    if len(match_feat) > 0:
                        sample_row = match_feat.iloc[0]
                        X_sample = sample_row[feature_cols].to_frame().T.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
                        
                        pred_val = float(model_obj.predict(X_sample)[0])
                        target_rating_col = 'rating_scaled' if 'rating_scaled' in sample_row else 'rating'
                        actual_val = float(sample_row[target_rating_col])
                        error = pred_val - actual_val
                        
                        shap_values = explainer_obj(X_sample)
                        vals = shap_values.values[0]
                        feat_names = feature_cols
                        top_pos_idx = np.argsort(-vals)[:3]
                        top_neg_idx = np.argsort(vals)[:3]
                        
                        col_res_left, col_res_right = st.columns([1.1, 1.9])
                        
                        with col_res_left:
                            st.subheader("Prediction Metrics")
                            st.metric(
                                label=f"{selected_model_name} Predicted Score",
                                value=f"{pred_val:.2f} / 5.0",
                                delta=f"{error:+.2f} vs actual",
                                delta_color="inverse"
                            )
                            st.metric(
                                label="Actual User Rating Score",
                                value=f"{actual_val:.2f} / 5.0"
                            )
                            
                            st.markdown("---")
                            
                            st.markdown("**Top Positive Drivers (Increased Score):**")
                            for pos_i in top_pos_idx:
                                if vals[pos_i] > 0:
                                    st.markdown(f'<span style="color:#ff0051;">■</span> <code>{feat_names[pos_i]}</code>: <b>+{vals[pos_i]:.3f}</b>', unsafe_allow_html=True)
                                    
                            st.write("")
                            st.markdown("**Top Negative Drivers (Decreased Score):**")
                            for neg_i in top_neg_idx:
                                if vals[neg_i] < 0:
                                    st.markdown(f'<span style="color:#008bfb;">■</span> <code>{feat_names[neg_i]}</code>: <b>{vals[neg_i]:.3f}</b>', unsafe_allow_html=True)
                                    
                        with col_res_right:
                            st.subheader("Feature Waterfall Explanation")
                            
                            top_driver_name = feat_names[top_pos_idx[0]] if vals[top_pos_idx[0]] > 0 else None
                            top_neg_name = feat_names[top_neg_idx[0]] if vals[top_neg_idx[0]] < 0 else None
                            if top_driver_name and top_neg_name:
                                st.caption(
                                    f"In plain terms: **{top_driver_name}** pushed this score up the most, "
                                    f"while **{top_neg_name}** pulled it down the most."
                                )
                                
                            plt.style.use('dark_background')
                            fig_shap, ax = plt.subplots(figsize=(7, 4.2))
                            fig_shap.patch.set_facecolor('#0e1117')
                            ax.set_facecolor('#0e1117')
                            
                            shap.plots.waterfall(shap_values[0], max_display=8, show=False)
                            st.pyplot(fig_shap)
                            plt.close(fig_shap)
                    else:
                        st.warning(f"Feature vector for item index `{selected_item_idx}` not found in test store.")
                        
            # TAB 2: GLOBAL SHAP SUMMARY
            with tab_global:
                st.markdown("### Global Feature Importance across Test Dataset")
                st.write("Quantifying overall model dependencies: Which engineered features impact recommendations most on average across the entire dataset?")
                
                global_model_choice = st.selectbox("Select Model for Global Analysis:", ["XGBoost", "LightGBM"])
                model_obj_g, explainer_obj_g = models_dict[global_model_choice]
                
                X_test_sample = test_feat[feature_cols].sample(min(200, len(test_feat)), random_state=42).apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
                
                with st.spinner("Computing global SHAP summary..."):
                    shap_values_global = explainer_obj_g(X_test_sample)
                    
                    plt.style.use('dark_background')
                    fig_global, ax_g = plt.subplots(figsize=(8, 4.5))
                    fig_global.patch.set_facecolor('#0e1117')
                    ax_g.set_facecolor('#0e1117')
                    
                    shap.plots.bar(shap_values_global, max_display=10, show=False)
                    st.pyplot(fig_global)
                    plt.close(fig_global)
    else:
        st.warning(f"Data artifacts missing for {dataset_choice}. Complete pipeline execution first.")