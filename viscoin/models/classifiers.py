"""Classifier models `f` to be explained. In VisCoIN, we aim at explaining ResNet-based classifiers.
The classifiers are ResNet-18 or ResNet-50 models with a custom fully connected layer head on top for classification
with a custom amount of output classes.
"""

import torch
from torch import Tensor, nn
from transformers import ResNetConfig, ResNetModel  # type: ignore

# Hidden state tuple for ResNet models
HiddenStates = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


class Classifier(nn.Module):
    """Resnet-based classifier model with a custom head on top for classification.

    The model returns its 5 hidden states during the forward pass.

    Hidden states for some ResNet models (note that the first dimension is the batch size):

    Resnet-50:
     - Layer 1: (1, 64, 56, 56)
     - Layer 2: (1, 256, 56, 56)
     - Layer 3: (1, 512, 28, 28)
     - Layer 4: (1, 1024, 14, 14)
     - Layer 5: (1, 2048, 7, 7)
    hidden config [256, 512, 1024, 2048]
    """

    def __init__(self, output_classes=200, pretrained=True) -> None:
        """Initialize a ResNet-based classifier model with a custom head on top for classification.

        Args:
            output_classes: the number of output classes (size of the last layer)
            pretrained: whether to load pretrained weights or not

        Example usages:

        Pretrained model:
        ```python
        >>> model = Classifier(output_classes=200, pretrained=True)
        ```

        Custom model:
        ```python
        >>> model = Classifier(output_classes=200, pretrained=False)
        >>> model.load_weights("model.pt")
        ```
        """
        super().__init__()

        model_name = "microsoft/resnet-50"

        # Load the model
        if pretrained:
            self.resnet = ResNetModel.from_pretrained(model_name)
        else:
            config = ResNetConfig.from_pretrained(model_name)
            self.resnet = ResNetModel(config)

        # Create the last linear layer
        self.last_size = self.resnet.config.hidden_sizes[-1]
        self.linear = nn.Linear(self.last_size, output_classes)

        # Apply Kaiming initialization to the last layer (only effective if not pretrained)
        nn.init.kaiming_normal_(self.linear.weight)

    def load_weights(self, filename: str):
        """Load the weights of the model from a file."""
        self.load_state_dict(torch.load(filename, weights_only=True))

    def forward(self, x: Tensor) -> tuple[Tensor, HiddenStates]:
        """Do a classifier forward pass.
        The input must be an image tensor with the following shape:
        (batch_size, num_channels, height, width)
        - num_channels = 3

        Args:
            x: the input image tensor

        Returns: A tuple of the output tensor and the hidden states
        """
        results = self.resnet(x, output_hidden_states=True)

        # Reshape the output (batch_size, last_size, 1, 1) to (batch_size, last_size) after the pooler
        output_state = results["pooler_output"].view(-1, self.last_size)
        hidden_states = results["hidden_states"]  # Tuple of 5 hidden states

        # Compute the output class unnormalized logits
        logits = self.linear(output_state)

        return logits, hidden_states
