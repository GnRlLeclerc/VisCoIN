"""
The explainer models (called `h` in FLINT, or `Theta` in VisCoIN)
compute an output class from the latent concept dictionary spaces.

The output fidelity loss can be computed from the difference between the output class predicted
by the original classifier model `f` and the output class predicted by the explainer model.
"""

from torch import Tensor, nn
from torch.nn.utils import parametrizations as P


class Explainer(nn.Module):
    """Explainer network. Produces a class prediction from a latent concept space.

    The concept spaces are 3x3 convolutional maps per concept.

    The explainer models pool each concept space in order to reduce the whole latent space
    from (batch_size, n_concepts, 3, 3) to (batch_size, n_concepts).

    A final linear layer is applied to produce class predictions.
    """

    def __init__(self, n_concepts=16, n_classes=200, normalized=False):
        """Initialize an explainer network.

        Args:
            n_concepts: number of concepts in the latent space
            n_classes: number of output classes
            normalized: whether to wrap the last linear layer with a weight normalization function
        """
        super().__init__()

        self.normalized = normalized

        self.dropout = nn.Dropout(0.01)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.linear = nn.Linear(n_concepts, n_classes, bias=True)

        if normalized:
            self.linear = P.weight_norm(self.linear, name="weight")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the explainer network.
        Args:
            x: (batch_size, n_concepts, 3, 3): batched latent concept spaces.

        Returns: a tensor of shape (batch_size, n_classes) with unnormalized class logits.
        """
        x = self.dropout(x)

        # (batch_size, n_concepts)
        x = self.pool(x).flatten(start_dim=1)

        # (batch_size, n_classes)
        return self.linear(x)
