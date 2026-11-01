import torch
import torch.nn as nn

class NeuralCollaborativeFiltering(nn.Module):
    """
    NeuMF / NCF Dual-Branch Architecture:
    - GMF Branch: Element-wise product of user and item latent embeddings.
    - MLP Branch: Concatenation of user and item embeddings fed through dense layers.
    - Final Fusion: Combines GMF and MLP vectors to predict ratings.
    """
    def __init__(self, num_users: int, num_items: int, latent_dim_gmf: int = 16, latent_dim_mlp: int = 32, mlp_hidden_dims: list = [64, 32, 16]):
        super().__init__()
        
        # GMF Embeddings
        self.user_embed_gmf = nn.Embedding(num_users, latent_dim_gmf)
        self.item_embed_gmf = nn.Embedding(num_items, latent_dim_gmf)
        
        # MLP Embeddings
        self.user_embed_mlp = nn.Embedding(num_users, latent_dim_mlp)
        self.item_embed_mlp = nn.Embedding(num_items, latent_dim_mlp)
        
        # MLP Dense Stack
        mlp_layers = []
        input_dim = latent_dim_mlp * 2
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Fusion Prediction Head
        final_dim = latent_dim_gmf + mlp_hidden_dims[-1]
        self.prediction_head = nn.Linear(final_dim, 1)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        # GMF
        u_gmf = self.user_embed_gmf(user_indices)
        i_gmf = self.item_embed_gmf(item_indices)
        gmf_vec = u_gmf * i_gmf
        
        # MLP
        u_mlp = self.user_embed_mlp(user_indices)
        i_mlp = self.item_embed_mlp(item_indices)
        mlp_vec = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_vec = self.mlp(mlp_vec)
        
        # NeuMF Fusion
        fusion_vec = torch.cat([gmf_vec, mlp_vec], dim=-1)
        logits = self.prediction_head(fusion_vec)
        return torch.sigmoid(logits).squeeze(-1)