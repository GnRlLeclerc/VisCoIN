"""GradCAM heatmap analysis"""

import numpy as np
from torch import Tensor, nn


class GradCAM:
    def __init__(self, conv_layer: nn.Conv2d) -> None:
        # Register the hooks for the given convolutional layer
        conv_layer.register_full_backward_hook(self._save_gradients_hook)  # type: ignore
        conv_layer.register_forward_hook(self._save_activations_hook)

        self.activation = np.array([])
        self.gradient = np.array([])

    def _save_gradients_hook(
        self, _module: nn.Module, _grad_input: tuple[Tensor], grad_output: tuple[Tensor]
    ):
        self.gradient = grad_output[0].detach().cpu().numpy()

    def _save_activations_hook(
        self, _module: nn.Module, _grad_input: tuple[Tensor], output: Tensor
    ):
        self.activation = output.detach().cpu().numpy()

    def compute(self) -> np.ndarray:
        """Computes the GradCAM heatmap for each image in the processed batch"""
        # Compute gradient weights for each channel
        weights = self.gradient.mean(axis=(2, 3))  # shape: [batch_size, 128]

        # Sum the weighted activations for each channel into maps
        # (batch_size, height, width)
        heatmaps = self.activation * weights[:, :, None, None]  # .sum(axis=1)

        # Remove negative values (relu)
        heatmaps[heatmaps < 0] = 0

        # Normalize to [0, 1]
        batchwise_max = heatmaps.max(axis=(1, 2, 3))[:, None, None, None]
        batchwise_min = heatmaps.min(axis=(1, 2, 3))[:, None, None, None]

        divider = batchwise_max - batchwise_min
        divider[divider == 0] = 1  # Avoid division by zero
        heatmaps = (heatmaps - batchwise_min) / divider

        return heatmaps
