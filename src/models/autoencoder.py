import torch
import torch.nn as nn

class AutoRec(nn.Module):
    """
    User-based AutoRec Denoising Autoencoder:
    Reconstructs partially observed user interaction vectors.
    """
    def __init__(self, num_items: int, hidden_dim: int = 128, dropout_prob: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(num_items, hidden_dim),
            nn.Sigmoid()
        )
        self.decoder = nn.Linear(hidden_dim, num_items)

    def forward(self, user_vectors: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(user_vectors)
        reconstruction = self.decoder(latent)
        return reconstruction