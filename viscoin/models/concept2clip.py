"""
Model to convert concept space embeddings to clip embeddings
"""

import torch.nn as nn
from torch import Tensor


class Concept2CLIP(nn.Module):
    """Basic adapter model with two linear layers"""

    def __init__(self, n_concepts: int, clip_dim: int):
        """
        Args:
            n_concepts: amount of viscoin concepts
            clip_dim: dimension of the clip embeddings
            hidden_dim: dimension of the hidden layer
        """
        super().__init__()

        hidden_dim = (n_concepts * 3 * 3 + clip_dim) // 2

        self.model = nn.Sequential(
            nn.Linear(n_concepts * 3 * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, clip_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: (batch_size, n_concepts, 3, 3): Concept embeddings
        """
        # Flatten the input tensor to (batch_size, n_concepts * 3 * 3)
        x = x.view(x.size(0), -1)

        return self.model(x)
