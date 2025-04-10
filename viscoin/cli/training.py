import os
import sys

import click
import clip
import torch
from torch.utils.data import DataLoader

from viscoin.cli.utils import batch_size, dataset_type, device
from viscoin.datasets.cub import CUB_200_2011
from viscoin.datasets.funnybirds import FunnyBirds
from viscoin.models.classifiers import Classifier
from viscoin.models.concept2clip import Concept2CLIP
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer
from viscoin.models.gan import GeneratorAdapted
from viscoin.models.utils import load_viscoin, load_viscoin_pickle, save_viscoin_pickle
from viscoin.testing.classifiers import test_classifier
from viscoin.training.classifiers import train_classifier
from viscoin.training.concept2clip import Concept2ClipTrainingParams, train_concept2clip
from viscoin.training.viscoin import TrainingParameters, train_viscoin
from viscoin.utils.logging import configure_score_logging

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "stylegan2_ada"))
from stylegan2_ada.training.networks import Generator

DEFAULT_CHECKPOINTS = {
    "cub": {
        "classifier": "checkpoints/cub/classifier-cub.pkl",
        "gan": "checkpoints/cub/gan-cub.pkl",
        "gan_adapted": "checkpoints/cub/gan-adapted-cub.pkl",
    },
    "funnybirds": {
        "classifier": "checkpoints/funnybirds/classifier-fb.pkl",
        "gan": "checkpoints/funnybirds/gan-fb.pkl",
        "gan_adapted": "checkpoints/funnybirds/gan-adapted-fb.pkl",
    },
}

DATASET_CLASSES = {
    "cub": 200,
    "funnybirds": 50,
}


@click.command()
@batch_size
@device
@dataset_type
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
    dataset_type: str,
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

    match dataset_type:
        case "cub":
            train_dataset = CUB_200_2011(mode="train")
            test_dataset = CUB_200_2011(mode="test")
        case "funnybirds":
            train_dataset = FunnyBirds(mode="train")
            test_dataset = FunnyBirds(mode="test")
        case _:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    batch_size = batch_size // gradient_accumulation_steps
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    match model_name:
        case "classifier":
            setup_classifier_training(
                checkpoints,
                device,
                train_loader,
                test_loader,
                epochs,
                learning_rate,
                output_weights,
            )

        case "viscoin":
            setup_viscoin_training(
                checkpoints,
                dataset_type,
                device,
                train_loader,
                test_loader,
                epochs,
                gradient_accumulation_steps,
            )

        case "concept2clip":
            setup_concept2clip_training(
                model_name,
                device,
                dataset_type,
                epochs,
                learning_rate,
                output_weights,
                batch_size,
            )

        case _:
            raise ValueError(f"Unknown model name: {model_name}")


def setup_classifier_training(
    checkpoints: str | None,
    device: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    output_weights: str,
):
    """Helper function to setup the training of a classifier"""

    model = Classifier(output_classes=200, pretrained=checkpoints is None).to(device)

    if checkpoints is not None:
        model.load_state_dict(torch.load(checkpoints, weights_only=True))

    model = model.to(device)

    configure_score_logging(f"classifier_{epochs}.log")
    train_classifier(
        model,
        train_loader,
        test_loader,
        device,
        epochs,
        learning_rate,
    )

    weights = model.state_dict()
    torch.save(weights, output_weights)


def setup_concept2clip_training(
    model_type: str,
    device: str,
    dataset_type: str,
    epochs: int,
    learning_rate: float,
    output_weights: str,
    batch_size: int,
):
    """Helper function to setup the training of a clip adapter"""

    viscoin = load_viscoin_pickle("checkpoints/cub/viscoin-cub.pkl")

    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Loading the appropriate clip adapter model
    n_concepts = viscoin.concept_extractor.n_concepts
    clip_embedding_dim = clip_model.visual.output_dim

    concept2clip = Concept2CLIP(n_concepts * 9, clip_embedding_dim)
    params = Concept2ClipTrainingParams(epochs=epochs, learning_rate=learning_rate)

    concept2clip = concept2clip.to(device)

    configure_score_logging(f"{model_type}_{epochs}.log")

    # Creating new dataloader with the clip preprocess as clip does not work with all image sizes
    match dataset_type:
        case "cub":
            train_dataset = CUB_200_2011(mode="train", transform=preprocess)
            test_dataset = CUB_200_2011(mode="test", transform=preprocess)
        case "funnybirds":
            train_dataset = FunnyBirds(mode="train", transform=preprocess)
            test_dataset = FunnyBirds(mode="test", transform=preprocess)
        case _:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # The training saves the viscoin model regularly
    train_concept2clip(
        viscoin.classifier.to(device),
        viscoin.concept_extractor.to(device),
        concept2clip,
        clip_model,
        train_loader,
        test_loader,
        device,
        params,
    )

    torch.save(concept2clip, output_weights)


def load_classifier(path: str, n_classes: int) -> Classifier:
    if not os.path.exists(path):
        return Classifier(output_classes=n_classes)
    else:
        return torch.load(path, weights_only=False)


def load_gan(path_viscoin_gan: str, path_generator_gan: str) -> tuple[GeneratorAdapted, Generator]:
    # Generator GAN must be provided
    assert os.path.exists(path_generator_gan), f"Generator GAN not found."

    generator_gan = torch.load(path_generator_gan, weights_only=False)

    # If the adapted generator does not exist, create it from the generator GAN
    if not os.path.exists(path_viscoin_gan):
        viscoin_gan = GeneratorAdapted.from_gan(generator_gan)
        torch.save(viscoin_gan, path_viscoin_gan)
    else:
        viscoin_gan = torch.load(path_viscoin_gan, weights_only=False)

    return viscoin_gan, generator_gan


def setup_viscoin_training(
    checkpoints: str | None,
    dataset_type: str,
    device: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    gradient_accumulation_steps: int,
):
    """Helper function to setup the training of viscoin"""

    concept_extractor = ConceptExtractor().to(device)
    explainer = Explainer(n_classes=DATASET_CLASSES[dataset_type]).to(device)

    classifier = load_classifier(
        DEFAULT_CHECKPOINTS[dataset_type]["classifier"], DATASET_CLASSES[dataset_type]
    ).to(device)
    viscoin_gan, generator_gan = load_gan(
        DEFAULT_CHECKPOINTS[dataset_type]["gan_adapted"],
        DEFAULT_CHECKPOINTS[dataset_type]["gan"],
    )

    viscoin_gan = viscoin_gan.to(device)
    generator_gan = generator_gan.to(device)

    if checkpoints is not None:
        load_viscoin(classifier, concept_extractor, explainer, viscoin_gan, checkpoints)

    configure_score_logging(f"viscoin_{epochs}.log")

    # Using the default parameters for training on CUB
    params = TrainingParameters()
    params.gradient_accumulation = gradient_accumulation_steps

    # The training saves the viscoin model regularly
    train_viscoin(
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


@click.command()
@batch_size
@device
@dataset_type
@click.argument("model_name")
@click.option("--checkpoints", help="The path to load the checkpoints", type=str)
def test(
    model_name: str,
    batch_size: int,
    device: str,
    dataset_type: str,
    checkpoints: str | None,
):
    """Test a model on a dataset"""

    match dataset_type:
        case "cub":
            dataset = CUB_200_2011(mode="test")
        case "funnybirds":
            dataset = FunnyBirds(mode="test")
        case _:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pretrained = checkpoints is not None

    match model_name:
        case "classifier":
            model = Classifier(
                output_classes=DATASET_CLASSES[dataset_type], pretrained=pretrained
            ).to(device)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    if pretrained:
        model.load_state_dict(torch.load(checkpoints, weights_only=True))

    model = model.to(device)

    accuracy, loss = test_classifier(model, dataloader, device)

    click.echo(f"Accuracy: {100*accuracy:.2f}%")
    click.echo(f"Loss: {loss}")


@click.command()
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
