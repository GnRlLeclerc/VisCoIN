"""
The concept extractor models (called `Psi` in FLINT & VisCoIN)
compute latent concept embeddings from latent layers of an original classifier `f`.

Their primary output is `Phi(x) = (phi_1(x), ..., phi_K(x))`, where each phi_i is a dictionary function
representing a single concept, of dimension `b` represented as a 3x3 convolutional map.
Their secondary output is `Phi'`, a vector of same size, designed to help the GAN reconstruction.

The complete concept latent space is a matrix of dimension `K x b`.
In FLINT, the amount of concepts is `J`, whereas in VisCoIN, it is `K`.

Note that the concept extractor models are working with ResNet latent layers, which are are 2D tensors (images)
and not 1D tensors (vectors) like standard fully connected neural networks.
"""

import torch
from torch import Tensor, nn


class ConceptExtractor(nn.Module):
    """Concept Extractor model for ResNet-based models.
    This model takes as an input the 3 last intermediate feature maps from a ResNet-based model.

    It outputs two tensors:
    - `Phi`, the concept embedding tensor of `n_concepts` (`J` in FLINT, `b` in VisCoIN) concepts, each one embedded in a convolutional layer of size 3x3 = `b`
    - `Phi'`, a vector of size `n_concepts * 9` (same total size as `Phi`) designed to help the GAN reconstruction.

    Resnet-based model have the following shapes for their 3 last intermediate layers:
     - Layer 3: (batch_size, latent_3, 28, 28)
     - Layer 4: (batch_size, latent_4, 14, 14)
     - Layer 5: (batch_size, latent_5, 7, 7)

    The first convolutions transform these layers into the same latent space:
     - Layer 3: (batch_size, latent_channels, 7, 7)
     - Layer 4: (batch_size, latent_channels, 7, 7)
     - Layer 5: (batch_size, latent_channels, 7, 7)

    They can then be concatenated along the `latent_channels` dimensions, for further processing.
    """

    def __init__(
        self,
        n_concepts=16,
        latent_channels=512,
        input_channels1=512,
        input_channels2=1024,
        input_channels3=2048,
    ):
        """Instanciate a concept extractor model `Psi` for ResNet-based models.

        Note: the default arguments are configured for a ResNet-50 model.

        Args:
            n_concepts: Amount of output concepts.
            latent_channels: Number of channels for the latent feature maps.
            input_channels1: Number of channels for the [-3] resnet latent feature map.
            input_channels2: Number of channels for the [-2] resnet latent feature map.
            input_channels3: Number of channels for the [-1] resnet latent feature map.
        """
        super().__init__()

        self.latent_channels = latent_channels
        self.n_concepts = n_concepts

        # Define convolutional layers to process and normalize each of the input feature maps
        # Resulting shape (batch_size, latent_channels, 7, 7)
        self.conv1 = nn.Conv2d(input_channels1, latent_channels, kernel_size=5, padding=2, stride=4)
        self.conv2 = nn.Conv2d(input_channels2, latent_channels, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(input_channels3, latent_channels, kernel_size=3, padding=1, stride=1)

        # Define additional layers for feature fusion and output
        # still (batch_size, latent_channels, 7, 7)
        self.conv4 = nn.Conv2d(
            3 * latent_channels, 3 * latent_channels, kernel_size=3, padding=1, stride=1
        )

        # Shared pooling layer for both outputs
        # Maps to an output of shape 3x3, ie (batch_size, n_concepts, 3, 3)
        self.pool = nn.AdaptiveAvgPool2d(3)

        # ------------- Compute the first output x1 ------------ #
        self.conv5 = nn.Conv2d(3 * latent_channels, n_concepts, kernel_size=1, padding=0, stride=1)

        # ------------ Compute the second output x2 ------------ #
        # Hardcoded to 256 the size of the helper space Phi'
        # (batch_size, 9 * n_concepts)
        self.linear1 = nn.Linear(latent_channels * 3 * 9, 9 * 256, bias=True)
        # (batch_size, 9 * n_concepts)
        self.linear2 = nn.Linear(9 * 256, 9 * 256, bias=False)

        # Define activation and pooling layers
        self.activ = nn.ReLU()

    def forward(self, latent_layers: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Computes concept embeddings from the 3 latent layers of a classifier.

        Args:
            latent_layers: Tuple of 3 tensors, each one representing a latent feature map from the classifier. For a Resnet-based model, these are the layers [-3], [-2] and [-1] in this order.

        Returns: Tuple of 2 tensors.
         - `Phi` (batch_size, n_concepts, 3, 3): Concept convolutional embeddings.
         - `Phi'` (batch_size, 9 * n_concepts): Vector designed to help the GAN reconstruction.
        """

        assert len(latent_layers) == 3, "Input should be a tuple of 3 tensors"

        # Process each feature map with corresponding convolution
        # (batch_size, latent_channels, 7, 7) for the 3 of them
        x1 = self.activ(self.conv1(latent_layers[0]))
        x2 = self.activ(self.conv2(latent_layers[1]))
        x3 = self.activ(self.conv3(latent_layers[2]))

        # Concatenate feature maps along the channel dimension
        # (batch_size, 3 * latent_channels, 7, 7)
        x = torch.cat((x1, x2, x3), 1)

        # Mix the concatenated feature map further
        # (batch_size, 3 * latent_channels, 7, 7)
        x = self.activ(self.conv4(x))

        # ------------- Compute the first output x1 ------------ #
        # (batch_size, n_concepts, 7, 7)
        x1 = self.activ(self.conv5(x))

        # (batch_size, n_concepts, 3, 3)
        x1 = self.pool(x1)

        # ------------ Compute the second output x2 ------------ #
        # (batch_size, 3 * latent_channels, 3, 3)
        x2 = self.pool(x)
        # (batch_size, 3 * latent_channels * 9)
        x2 = x2.view(-1, self.latent_channels * 3 * 9)
        # (batch_size, 9 * n_concepts)
        x2 = self.activ(self.linear1(x2))
        # (batch_size, 9 * n_concepts)
        x2 = self.activ(self.linear2(x2))

        # End shapes
        # x1: (batch_size, n_concepts, 3, 3)
        # x2: (batch_size, 9 * n_concepts)

        return x1, x2
