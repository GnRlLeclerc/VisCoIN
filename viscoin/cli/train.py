"""Model training CLI command"""

import click
import torch

from viscoin.cli.utils import (
    batch_size,
    checkpoints,
    dataset,
    device,
    epochs,
    learning_rate,
    output_weights,
)
from viscoin.datasets.utils import (
    DATASET_CLASSES,
    DEFAULT_CHECKPOINTS,
    DatasetType,
    get_dataloaders,
)
from viscoin.models.classifiers import Classifier
from viscoin.models.clip import CLIP
from viscoin.models.concept2clip import Concept2CLIP
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer
from viscoin.models.gan import GeneratorAdapted
from viscoin.models.utils import load_viscoin_pickle
from viscoin.training.classifiers import ClassifierTrainingParams, train_classifier
from viscoin.training.concept2clip import Concept2ClipTrainingParams, train_concept2clip
from viscoin.training.viscoin import VisCoINTrainingParams, train_viscoin
from viscoin.utils.logging import configure_score_logging


@click.command()
@click.argument("model_name")
@batch_size
@device
@dataset
@epochs
@learning_rate
@output_weights
@click.option(
    "--gradient-accumulation-steps",
    help="The amount of steps to accumulate gradients before stepping the optimizers",
    type=int,
    default=1,
)
@checkpoints
def train(
    model_name: str,
    dataset: DatasetType,
    device: str,
    checkpoints: str | None,
    batch_size: int | None,
    learning_rate: float | None,
    epochs: int | None,
    output_weights: str,
    gradient_accumulation_steps: int,
):
    """Train a model on a dataset

    Metrics are logged to a file.
    """

    match model_name:
        case "classifier":
            _train_classifier(
                dataset,
                device,
                checkpoints,
                batch_size,
                learning_rate,
                epochs,
                output_weights,
            )

        case "concept2clip":
            _train_concept2clip(
                device,
                dataset,
                epochs,
                learning_rate,
                batch_size,
                output_weights,
            )

        case "viscoin":
            _train_viscoin(
                dataset,
                device,
                learning_rate,
                epochs,
                batch_size,
                gradient_accumulation_steps,
            )

        case _:
            raise ValueError(f"Unknown model name: {model_name}")


###################################################################################################
#                                          HELPER METHODS                                         #
###################################################################################################


def _train_classifier(
    dataset: DatasetType,
    device: str,
    checkpoints: str | None,
    batch_size: int | None,
    learning_rate: float | None,
    epochs: int | None,
    output_weights: str,
):
    """Train the VisCoIN classifier"""

    configure_score_logging(f"classifier_{epochs}.jsonl")

    params = ClassifierTrainingParams(epochs, learning_rate, batch_size, device)  # type: ignore
    train, test = get_dataloaders(dataset, params.batch_size)

    model = Classifier(output_classes=DATASET_CLASSES[dataset], pretrained=checkpoints is None)

    if checkpoints is not None:
        model.load_state_dict(torch.load(checkpoints, weights_only=True))

    train_classifier(model.to(device), train, test, params)

    weights = model.state_dict()
    torch.save(weights, output_weights)


def _train_concept2clip(
    device: str,
    dataset: DatasetType,
    epochs: int | None,
    learning_rate: float | None,
    batch_size: int | None,
    output_weights: str,
):
    """Train a concept2clip model"""

    viscoin = load_viscoin_pickle(DEFAULT_CHECKPOINTS[dataset]["viscoin"])
    clip_model = CLIP(device)

    # Loading the appropriate clip adapter model
    n_concepts = viscoin.concept_extractor.n_concepts
    concept2clip = Concept2CLIP(n_concepts, clip_model.embedding_size)
    params = Concept2ClipTrainingParams(
        epochs=epochs, learning_rate=learning_rate, batch_size=batch_size  # type: ignore
    )

    configure_score_logging(f"concept2clip_{epochs}.jsonl")

    # The training saves the viscoin model regularly
    train_concept2clip(
        viscoin.classifier.to(device),
        viscoin.concept_extractor.to(device),
        concept2clip.to(device),
        clip_model,
        dataset,
        device,
        params,
    )

    torch.save(concept2clip, output_weights)


def _train_viscoin(
    dataset: DatasetType,
    device: str,
    learning_rate: float | None,
    epochs: int | None,
    batch_size: int | None,
    gradient_accumulation_steps: int | None,
):
    """Train a VisCoIN model"""

    """Helper function to setup the training of viscoin"""

    concept_extractor = ConceptExtractor()
    explainer = Explainer(n_classes=DATASET_CLASSES[dataset])
    classifier = torch.load(DEFAULT_CHECKPOINTS[dataset]["classifier"], weights_only=False)

    generator_gan = torch.load(DEFAULT_CHECKPOINTS[dataset]["gan"])
    viscoin_gan = GeneratorAdapted.from_gan(generator_gan)

    configure_score_logging(f"viscoin_{epochs}.jsonl")

    # Using the default parameters for training on CUB
    params = VisCoINTrainingParams(
        learning_rate=learning_rate,  # type: ignore
        iterations=epochs,  # type: ignore
        gradient_accumulation=gradient_accumulation_steps,  # type: ignore
        batch_size=batch_size,  # type: ignore
    )

    train, test = get_dataloaders(dataset, params.batch_size)

    # The training saves the viscoin model regularly
    train_viscoin(
        classifier.to(device),
        concept_extractor.to(device),
        explainer.to(device),
        viscoin_gan.to(device),
        generator_gan.to(device),
        train,
        test,
        params,
    )
