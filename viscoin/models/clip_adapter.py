""" 
Model to convert concept space embeddings to clip embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipAdapter(nn.Module):
    """Basic adapter model with two linear layers"""

    def __init__(self, in_features: int, out_features: int, hidden_size: int = 1024):
        """
        Args:
            in_features (int): dimension of the concept embeddings
            out_features (int): dimension of the clip embeddings
            hidden_size (int, optional): Defaults to 1024.
        """
        super(ClipAdapter, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class ClipAdapterVAE(torch.nn.Module):
    """Adapter model with a VAE architecture :
    Note that this VAE does not have the same input and output dimensions
    """

    def __init__(
        self, in_features: int, out_features: int, hidden_size: int = 1024, latent_size: int = 512
    ):
        """
        Args:
            in_features (int): dimension of the concept embeddings
            out_features (int): dimension of the clip embeddings
            hidden_size (int, optional): Defaults to 1024.
            latent_size (int, optional): Size of the latent space. Defaults to 512.
        """
        super(ClipAdapterVAE, self).__init__()

        # Encoder layers
        self.linear_enc = nn.Linear(in_features, hidden_size)

        # Latent space
        self.linear_mu = nn.Linear(hidden_size, latent_size)
        self.linear_logvar = nn.Linear(hidden_size, latent_size)

        # Decoder layers
        self.linear_dec = nn.Linear(latent_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, out_features)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.linear_enc(x))
        return self.linear_mu(h), self.linear_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.linear_dec(z))
        reconstruction = torch.sigmoid(self.linear_out(h))
        return reconstruction

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
