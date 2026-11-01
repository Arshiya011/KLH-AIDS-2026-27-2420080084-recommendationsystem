import numpy as np
import pandas as pd

class GlobalMeanRecommender:
    """Predicts global mean or item mean rating."""
    def __init__(self):
        self.global_mean = 0.0
        self.item_means = {}

    def fit(self, train_df: pd.DataFrame):
        self.global_mean = train_df['rating'].mean()
        self.item_means = train_df.groupby('item_idx')['rating'].mean().to_dict()

    def predict(self, user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        preds = [self.item_means.get(item, self.global_mean) for item in item_indices]
        return np.array(preds)


class ItemKNNRecommender:
    """Cosine similarity Item-Item Collaborative Filtering baseline."""
    def __init__(self, k_neighbors: int = 20):
        self.k_neighbors = k_neighbors
        self.item_sim_matrix = None
        self.user_item_matrix = None
        self.global_mean = 0.0

    def fit(self, train_df: pd.DataFrame, n_users: int, n_items: int):
        self.global_mean = train_df['rating'].mean()
        self.user_item_matrix = np.zeros((n_users, n_items), dtype=np.float32)
        
        for row in train_df.itertuples():
            self.user_item_matrix[row.user_idx, row.item_idx] = row.rating

        # Item Cosine Similarity
        norms = np.linalg.norm(self.user_item_matrix, axis=0, keepdims=True) + 1e-9
        norm_matrix = self.user_item_matrix / norms
        self.item_sim_matrix = np.dot(norm_matrix.T, norm_matrix)
        np.fill_diagonal(self.item_sim_matrix, 0.0)

    def predict_pair(self, user_idx: int, item_idx: int) -> float:
        user_ratings = self.user_item_matrix[user_idx]
        sim_scores = self.item_sim_matrix[item_idx]
        
        # Non-zero rated items by user
        rated_indices = np.where(user_ratings > 0)[0]
        if len(rated_indices) == 0:
            return self.global_mean
            
        top_k = rated_indices[np.argsort(sim_scores[rated_indices])[-self.k_neighbors:]]
        weights = sim_scores[top_k]
        
        if np.sum(weights) == 0:
            return self.global_mean
            
        return float(np.sum(weights * user_ratings[top_k]) / np.sum(weights))

    def predict(self, user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        return np.array([self.predict_pair(u, i) for u, i in zip(user_indices, item_indices)])