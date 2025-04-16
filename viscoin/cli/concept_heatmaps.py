import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import torch
import torch.nn.functional as F

from viscoin.cli.utils import device, viscoin_pickle_path
from viscoin.datasets.cub import CUB_200_2011
from viscoin.datasets.utils import DEFAULT_CHECKPOINTS
from viscoin.models.utils import load_viscoin_pickle
from viscoin.utils.gradcam import GradCAM
from viscoin.utils.images import from_torch, heatmap_to_img, overlay


@click.command()
@viscoin_pickle_path
@device
def concept_heatmaps(device: str, viscoin_pickle_path: str = DEFAULT_CHECKPOINTS["cub"]["viscoin"]):
    """Generate heatmaps for random images of the dataset, for the 5 convolutional layers of the concept extractor,
    using GradCAM."""

    n_samples = 5

    # Load dataset and models
    models = load_viscoin_pickle(viscoin_pickle_path)
    dataset = CUB_200_2011(mode="test")

    # Move models to device
    classifier = models.classifier.to(device)
    concept_extractor = models.concept_extractor.to(device)
    explainer = models.explainer.to(device)

    # GradCAM for each convolutional layer
    gradcam1 = GradCAM(concept_extractor.conv1)
    gradcam2 = GradCAM(concept_extractor.conv2)
    gradcam3 = GradCAM(concept_extractor.conv3)
    gradcam4 = GradCAM(concept_extractor.conv4)
    gradcam5 = GradCAM(concept_extractor.conv5)

    # Choose random images to amplify
    indices = rd.choice(len(dataset), n_samples, replace=False)
    # indices = [0, 1, 2, 100, 200]
    images = torch.zeros(n_samples, 3, 256, 256).to(device)
    labels = torch.zeros(n_samples, dtype=torch.int64).to(device)
    for i, index in enumerate(indices):
        images[i] = dataset[index][0].to(device)
        labels[i] = dataset[index][1]

    # Do a forward pass
    _, hidden_states = classifier.forward(images)
    concept_maps, _ = concept_extractor.forward(hidden_states[-3:])
    explainer_classes = explainer.forward(concept_maps)

    explainer_labels = explainer_classes.argmax(dim=1)

    # Compute loss
    loss = F.cross_entropy(explainer_classes, labels)
    loss.backward()

    # Compute heatmaps
    heatmaps = [
        gradcam1.compute(),
        gradcam2.compute(),
        gradcam3.compute(),
        gradcam4.compute(),
        gradcam5.compute(),
    ]

    columns = [
        "original",
        "conv1 from hidden_state[-3]",
        "conv2 from hidden_state[-2]",
        "conv3 from hidden_state[-1]",
        "conv4 after concat",
        "conv5 after conv4",
    ]

    fig, axs = plt.subplots(n_samples, 6, figsize=(20, 10))
    fig.suptitle("GradCAM heatmaps of the concept extractor convolutional layers")

    for row in range(n_samples):
        # Set the row label
        is_correct = labels[row] == explainer_labels[row]
        confidence = F.softmax(explainer_classes[row], dim=0).max().item()

        axs[row, 0].set_ylabel(f"{is_correct} with {100 * confidence:.0f}%", fontsize=8)

        for column in range(6):
            if column == 0:
                # Display the original image
                axs[row, column].imshow(from_torch(images[row]))
            else:
                # Display the relevant heatmap
                axs[row, column].axis("off")
                axs[row, column].imshow(
                    overlay(
                        (from_torch(images[row]) * 255).astype(np.uint8),
                        heatmap_to_img(heatmaps[column - 1][row]),
                    )
                )

            if row == 0:
                # Set the title a bit smaller
                axs[row, column].set_title(columns[column], fontsize=8)

    plt.show()
