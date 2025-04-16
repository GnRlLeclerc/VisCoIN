import os
import pickle
import random

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from viscoin.cli.utils import batch_size, device, viscoin_pickle_path
from viscoin.datasets.cub import CUB_200_2011
from viscoin.models.utils import load_viscoin_pickle
from viscoin.testing.concepts import test_concepts
from viscoin.testing.viscoin import (
    ThresholdSelection,
    TopKSelection,
    amplify_concepts,
    amplify_single_concepts,
    plot_amplified_images_batch,
)
from viscoin.utils.gradcam import GradCAM
from viscoin.utils.images import from_torch, heatmap_to_img, overlay


@click.command()
@viscoin_pickle_path
@device
@click.option(
    "--concept-threshold",
    help="Use a concept activation threshold to select the concepts to amplify. In [-1, 1], prefer 0.2 as a default. Exclusive with concept-top-k",
    type=float,
)
@click.option(
    "--concept-top-k",
    help="The amount of most activated concepts to amplify. Exclusive with concept-threshold",
    type=int,
)
def amplify(
    viscoin_pickle_path: str,
    concept_threshold: float | None,
    concept_top_k: int | None,
    device: str,
):
    """Amplify the concepts of random images from a dataset (showcase)"""
    n_samples = 5

    # Load dataset and models
    models = load_viscoin_pickle(viscoin_pickle_path)
    dataset = CUB_200_2011(mode="test")

    # Move models to device
    classifier = models.classifier.to(device)
    concept_extractor = models.concept_extractor.to(device)
    explainer = models.explainer.to(device)
    gan = models.gan.to(device)

    # Choose random images to amplify
    indices = rd.choice(len(dataset), n_samples, replace=False)
    originals = [dataset[i][0].to(device) for i in indices]
    amplified: list[list[Tensor]] = []
    multipliers: list[float] = []

    if concept_threshold is not None:
        concept_selection: ThresholdSelection | TopKSelection = {
            "method": "threshold",
            "threshold": concept_threshold,
        }
    elif concept_top_k is not None:
        concept_selection: ThresholdSelection | TopKSelection = {
            "method": "top_k",
            "k": concept_top_k,
        }
    else:
        raise ValueError("You must provide either concept-threshold or concept-top-k")

    for image in originals:
        results = amplify_concepts(
            image,
            classifier,
            concept_extractor,
            explainer,
            gan,
            concept_selection,
            device,
        )
        amplified.append(results.amplified_images)
        multipliers = results.multipliers

    plot_amplified_images_batch(originals, amplified, multipliers)


@click.command()
@viscoin_pickle_path
@batch_size
@device
@click.option(
    "--force",
    help="Recompute the concept through the dataset, even if cached",
    is_flag=True,
)
def concepts(
    viscoin_pickle_path: str,
    batch_size: int,
    force: bool,
    device: str,
):
    """Analyse the distribution of concepts across the test dataset, and how well they separate classes."""

    if force or not os.path.isfile("concept_results.pkl"):
        # Recompute the concept results

        dataset = CUB_200_2011(mode="test")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        viscoin = load_viscoin_pickle(viscoin_pickle_path)

        classifier = viscoin.classifier.to(device)
        concept_extractor = viscoin.concept_extractor.to(device)
        explainer = viscoin.explainer.to(device)

        results = test_concepts(classifier, concept_extractor, explainer, dataloader, device)

        # Pickle the results for later use
        pickle.dump(results, open("concept_results.pkl", "wb"))

    else:
        results = pickle.load(open("concept_results.pkl", "rb"))

    results.print_accuracies()
    results.plot_concept_activation_per_concept()
    results.plot_concept_activation_per_image()
    results.plot_class_concept_correlations()
    results.plot_concept_class_correlations()
    results.plot_concept_entropies()


@click.command()
@viscoin_pickle_path
@device
def concept_heatmaps(viscoin_pickle_path: str, device: str):
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


@click.command()
@viscoin_pickle_path
@click.option(
    "--concept-labels-path",
    help="The path to the concept labels file",
    type=str,
    default="./concept_labels.csv",
)
@click.option(
    "--concept-indices",
    help="The indices of the concepts to amplify : eg. 1,2,3,4,5",
    type=str,
    required=True,
)
@click.option(
    "--image-indices", help="The indices of the images to amplify : eg. 1,2,3,4,5", type=str
)
@device
def amplify_single(
    viscoin_pickle_path: str,
    concept_labels_path: str,
    concept_indices: str,
    image_indices: str,
    device: str,
):
    """Similar to amplify, but instead of amplifying multiple concepts for a given image, we amplify only a single concept per image.

    TODO : quel indice ? Quelle image ?
    """
    concept_labels: list[str] = []

    if concept_indices == "random":
        concept_indices_ints = random.sample(range(0, 256), 5)
    else:
        concept_indices_ints = [int(i) for i in concept_indices.split(",")]

    # If a path to the concept labels file is provided, we load the concept labels and the most activating images
    saved_image_indices = None
    if concept_labels_path:
        concept_labels_df = pd.read_csv(concept_labels_path)

        concept_labels = list(concept_labels_df["description"].values[concept_indices_ints])

        # Retrieve the most activating images for the selected concepts
        saved_image_indices = concept_labels_df["most-activating-images"].values[
            concept_indices_ints
        ]
        saved_image_indices = [int(l.split(":")[random.randint(0, 5)]) for l in saved_image_indices]

    # Either use the provided image indices or the ones given in the concept labels file
    if image_indices:
        image_indices_ints = [int(i) for i in image_indices.split(",")]
    elif saved_image_indices is not None:
        image_indices_ints = saved_image_indices
    else:
        raise ValueError(
            "You must provide the concept labels file if you do not provide the image indices"
        )

    assert len(concept_indices_ints) == len(
        image_indices
    ), "The number of concepts and images must be the same"

    print("Selected image indices: ", image_indices_ints)

    viscoin = load_viscoin_pickle(viscoin_pickle_path)
    dataset = CUB_200_2011(mode="test")

    images_batch: list[Tensor] = []
    amplified_images_batch: list[list[Tensor]] = []
    multipliers = [0.0, 1.0, 2.0, 4.0]

    for i, image_idx in enumerate(image_indices_ints):
        image = dataset[image_idx][0].to(device)

        amplified_images = amplify_single_concepts(
            image.to(device),
            viscoin.gan.to(device),
            viscoin.classifier.to(device),
            viscoin.concept_extractor.to(device),
            concept_indices_ints[i],
            multipliers,
        )

        images_batch.append(image)
        amplified_images_batch.append(amplified_images)

    plot_amplified_images_batch(images_batch, amplified_images_batch, multipliers, concept_labels)
