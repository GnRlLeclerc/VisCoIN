import click
import numpy.random as rd
from torch import Tensor

from viscoin.cli.utils import device, viscoin_pickle_path
from viscoin.datasets.cub import CUB_200_2011
from viscoin.models.utils import load_viscoin_pickle
from viscoin.testing.viscoin import (
    Selection,
    amplify_concepts,
    amplify_single_concepts,
    plot_amplified_images_batch,
)


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
@click.option(
    "--concept-indices",
    help="The indices of the concepts to amplify : eg. 1,2,3,4,5",
    type=str,
)
@click.option(
    "--image-indices", help="The indices of the images to amplify : eg. 1,2,3,4,5", type=str
)
def amplify(
    concept_threshold: float | None,
    concept_top_k: int | None,
    concept_indices: str | None,
    image_indices: str | None,
    device: str,
    viscoin_pickle_path: str,
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
    if image_indices is not None:
        indices = [int(i) for i in image_indices.split(",")]
    else:
        indices = rd.choice(len(dataset), n_samples, replace=False)

    originals = [dataset[i][0].to(device) for i in indices]
    amplified: list[list[Tensor]] = []

    if concept_indices is not None:
        concept_selection: Selection = {
            "method": "indices",
            "indices": [int(i) for i in concept_indices.split(",")],
        }
    elif concept_threshold is not None:
        concept_selection: Selection = {
            "method": "threshold",
            "threshold": concept_threshold,
        }
    elif concept_top_k is not None:
        concept_selection: Selection = {
            "method": "top_k",
            "k": concept_top_k,
        }
    else:
        raise ValueError("You must provide either concept-threshold or concept-top-k")

    multipliers = [0.0, 1.0, 2.0, 4.0]

    if concept_selection["method"] != "indices":
        for image in originals:
            results = amplify_concepts(
                image,
                classifier,
                concept_extractor,
                explainer,
                gan,
                concept_selection,
                multipliers,
                device,
            )
            amplified.append(results.amplified_images)
    else:
        for concept_id, image in zip(concept_selection["indices"], originals):
            results = amplify_single_concepts(
                image,
                gan,
                classifier,
                concept_extractor,
                concept_id,
                multipliers,
            )
            amplified.append(results)

    plot_amplified_images_batch(originals, amplified, multipliers)
