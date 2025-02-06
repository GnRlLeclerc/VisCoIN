"""
CLI entrypoint for interacting with the viscoin library.

Example usage:
```bash
python main.py test resnet50 --batch-size 32 --dataset-path datasets/CUB_200_2011/
```

Will be subject to many changes as the project evolves.
"""

import os
import pickle

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
import clip

from viscoin.datasets.cub import CUB_200_2011
from viscoin.models.classifiers import Classifier
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer
from viscoin.models.gan import GeneratorAdapted
from viscoin.models.clip_adapter import ClipAdapter, ClipAdapterVAE
from viscoin.models.utils import load_viscoin, load_viscoin_pickle, save_viscoin_pickle
from viscoin.testing.classifiers import test_classifier
from viscoin.testing.concepts import test_concepts
from viscoin.testing.clip_adapter import get_concept_labels_vocab
from viscoin.testing.viscoin import (
    ThresholdSelection,
    TopKSelection,
    amplify_concepts,
    plot_amplified_images_batch,
)
from viscoin.training.classifiers import train_classifier_cub
from viscoin.training.viscoin import TrainingParameters, train_viscoin_cub
from viscoin.training.clip_adapter import (
    train_clip_adapter_cub,
    ClipAdapterTrainingParams,
    ClipAdapterVAETrainingParams,
)
from viscoin.utils.cli import (
    batch_size,
    dataset_path,
    device,
    viscoin_pickle_path,
    clip_adapter_path,
    vocab_path,
)
from viscoin.utils.gradcam import GradCAM
from viscoin.utils.images import from_torch, heatmap_to_img, overlay
from viscoin.utils.logging import configure_score_logging
from viscoin.utils.types import TestingResults, TrainingResults


@click.group()
def main():
    pass


@main.command()
@batch_size
@device
@dataset_path
@click.argument("model_name")
@click.option(
    "--epochs",
    help="The amount of epochs to train the model for",
    default=30,
    type=int,
)
@click.option(
    "--learning-rate",
    help="The optimizer learning rate",
    default=0.0001,
    type=float,
)
@click.option(
    "--output-weights",
    help="The path/filename where to save the weights",
    type=str,
    default="output-weights.pt",
)
@click.option(
    "--gradient-accumulation-steps",
    help="The amount of steps to accumulate gradients before stepping the optimizers",
    type=int,
    default=1,
)
@click.option("--checkpoints", help="The path to load the checkpoints", type=str)
def train(
    model_name: str,
    batch_size: int,
    device: str,
    dataset_path: str,
    checkpoints: str | None,
    epochs: int,
    learning_rate: float,
    output_weights: str,
    gradient_accumulation_steps: int,
):
    """Train a model on a dataset.

    A progress bar is displayed during training.
    Metrics are logged to a file.
    """
    train_dataset = CUB_200_2011(dataset_path, mode="train")
    test_dataset = CUB_200_2011(dataset_path, mode="test")
    batch_size = batch_size // gradient_accumulation_steps
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    pretrained = checkpoints is not None

    match model_name:
        case "classifier":
            model = Classifier(output_classes=200, pretrained=pretrained).to(device)

            if pretrained:
                model.load_state_dict(torch.load(checkpoints, weights_only=True))

            model = model.to(device)

            configure_score_logging(f"{model_name}_{epochs}.log")
            train_classifier_cub(
                model,
                train_loader,
                test_loader,
                device,
                epochs,
                learning_rate,
            )

            weights = model.state_dict()
            torch.save(weights, output_weights)

        case "viscoin":
            classifier = torch.load("checkpoints/cub/classifier-cub.pkl", weights_only=False).to(
                device
            )
            concept_extractor = ConceptExtractor().to(device)
            explainer = Explainer().to(device)
            viscoin_gan = torch.load("checkpoints/cub/gan-adapted-cub.pkl", weights_only=False).to(
                device
            )
            generator_gan = torch.load("checkpoints/cub/gan-cub.pkl", weights_only=False).to(device)

            if pretrained:
                load_viscoin(classifier, concept_extractor, explainer, viscoin_gan, checkpoints)

            configure_score_logging(f"{model_name}_{epochs}.log")

            # Using the default parameters for training on CUB
            params = TrainingParameters()
            params.gradient_accumulation = gradient_accumulation_steps

            # The training saves the viscoin model regularly
            train_viscoin_cub(
                classifier,
                concept_extractor,
                explainer,
                viscoin_gan,
                generator_gan,
                train_loader,
                test_loader,
                params,
                device,
            )

        case "clip_adapter" | "clip_adapter_vae":

            viscoin = load_viscoin_pickle("checkpoints/cub/viscoin-cub.pkl")

            clip_model, preprocess = clip.load("ViT-B/32", device=device)

            # Loading the appropriate clip adapter model
            n_concepts = viscoin.concept_extractor.n_concepts
            clip_embedding_dim = clip_model.visual.output_dim

            if model_name == "clip_adapter":
                clip_adapter = ClipAdapter(n_concepts * 9, clip_embedding_dim)
                params = ClipAdapterTrainingParams(epochs=epochs, learning_rate=learning_rate)
            elif model_name == "clip_adapter_vae":
                clip_adapter = ClipAdapterVAE(
                    n_concepts * 9, clip_embedding_dim, hidden_size=512, latent_size=128
                )
                params = ClipAdapterVAETrainingParams(epochs=epochs, learning_rate=learning_rate)

            clip_adapter = clip_adapter.to(device)

            configure_score_logging(f"{model_name}_{epochs}.log")

            # Creating new dataloader with the clip preprocess as clip does not work with all image sizes
            train_dataset = CUB_200_2011(dataset_path, mode="train", transform=preprocess)
            test_dataset = CUB_200_2011(dataset_path, mode="test", transform=preprocess)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # The training saves the viscoin model regularly
            train_clip_adapter_cub(
                clip_adapter,
                viscoin.concept_extractor.to(device),
                viscoin.classifier.to(device),
                clip_model,
                train_loader,
                test_loader,
                device,
                params,
            )

            torch.save(clip_adapter, output_weights)

        case _:
            raise ValueError(f"Unknown model name: {model_name}")


@main.command()
@batch_size
@device
@dataset_path
@click.argument("model_name")
@click.option("--checkpoints", help="The path to load the checkpoints", type=str)
def test(
    model_name: str,
    batch_size: int,
    device: str,
    dataset_path: str,
    checkpoints: str | None,
):
    """Test a model on a dataset"""

    dataset = CUB_200_2011(dataset_path, mode="test")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pretrained = checkpoints is not None

    match model_name:
        case "classifier":
            model = Classifier(output_classes=200, pretrained=pretrained).to(device)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    if pretrained:
        model.load_state_dict(torch.load(checkpoints, weights_only=True))

    # TODO : if viscoin, specific stuff has to be done to ensure the model parameters are compatible

    model = model.to(device)

    accuracy, loss = test_classifier(model, dataloader, device)

    click.echo(f"Accuracy: {100*accuracy:.2f}%")
    click.echo(f"Loss: {loss}")


@main.command()
@dataset_path
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
    dataset_path: str,
    viscoin_pickle_path: str,
    concept_threshold: float | None,
    concept_top_k: int | None,
    device: str,
):
    """Amplify the concepts of random images from a dataset (showcase)"""
    n_samples = 5

    # Load dataset and models
    models = load_viscoin_pickle(viscoin_pickle_path)
    dataset = CUB_200_2011(dataset_path, mode="test")

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


@main.command()
@dataset_path
@viscoin_pickle_path
@batch_size
@device
@click.option(
    "--force",
    help="Recompute the concept through the dataset, even if cached",
    is_flag=True,
)
def concepts(
    dataset_path: str,
    viscoin_pickle_path: str,
    batch_size: int,
    force: bool,
    device: str,
):
    """Analyse the distribution of concepts across the test dataset, and how well they separate classes."""

    if force or not os.path.isfile("concept_results.pkl"):
        # Recompute the concept results

        dataset = CUB_200_2011(dataset_path, mode="test")
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


@main.command()
@click.option(
    "--logs-path",
    help="The path to the logs file",
    required=True,
    type=str,
)
def logs(logs_path: str):
    """Parse a viscoin training log file and plot the losses and metrics"""

    training_results: list[TrainingResults] = []
    testing_results: list[TestingResults] = []

    # Read the log file
    with open(logs_path, "r") as f:
        for line in f:
            if line.startswith("TestingResults"):
                testing_results.append(eval(line))
            elif line.startswith("TrainingResults"):
                training_results.append(eval(line))
            else:
                continue

    # Plot the losses
    TrainingResults.plot_losses(training_results)
    TestingResults.plot_losses(testing_results)

    # Plot the metrics
    TestingResults.plot_preds_overlap(testing_results)


@main.command()
@click.option("--checkpoints", help="The path to load the checkpoints", type=str)
@click.option("--output", help="The path to generate the pickle to", type=str)
def to_pickle(checkpoints: str, output: str):
    """Convert safetensors to a pickled viscoin model using default parameters"""

    classifier = Classifier()
    concept_extractor = ConceptExtractor()
    explainer = Explainer()
    gan = GeneratorAdapted()

    load_viscoin(classifier, concept_extractor, explainer, gan, checkpoints)
    save_viscoin_pickle(classifier, concept_extractor, explainer, gan, output)


@main.command()
@dataset_path
@viscoin_pickle_path
@device
def concept_heatmaps(dataset_path: str, viscoin_pickle_path: str, device: str):
    """Generate heatmaps for random images of the dataset, for the 5 convolutional layers of the concept extractor,
    using GradCAM."""

    n_samples = 5

    # Load dataset and models
    models = load_viscoin_pickle(viscoin_pickle_path)
    dataset = CUB_200_2011(dataset_path, mode="test")

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


@main.command()
@clip_adapter_path
@vocab_path
@click.option(
    "--n-concepts",
    help="The number of concepts",
    type=int,
    default=256,
)
@device
def clip_concept_labels(clip_adapter_path: str, vocab_path: str, n_concepts: int, device: str):
    """
    Generate concept labels from a given vocabulary using the clip adapter model :
        The clip adapter model is used to generate clip embeddings from concept space embeddings.
        Those clip embeddings are then compared to the text embeddings of the vocabulary to get the concept labels.

        We generate the concept space embeddings for each concept for instance by putting a 1 in the 9 corresponding positions of a vector of size n_concepts*9.

    Args:
        clip_adapter_path (str): Path to the clip adapter model
        vocab_path (str): Path to the vocabulary file (.txt)
        n_concepts (int): The number of concepts
    """

    clip_adapter = torch.load(clip_adapter_path).to(device)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    with open(vocab_path, "r") as f:
        vocab = f.readlines()

    vocab = [v.strip() for v in vocab]

    concept_labels, probs = get_concept_labels_vocab(
        clip_adapter,
        clip_model,
        vocab,
        n_concepts,
        device,
    )

    # Save to file
    with open("concept_labels.csv", "w") as f:
        for i, label in enumerate(concept_labels):
            probs_line = [str(v) for v in probs[i][0]]
            f.write(f"{i}, {label}, [{' '.join(probs_line)}]\n")


if __name__ == "__main__":
    main()
