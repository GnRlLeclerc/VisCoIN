import os
import pickle

import click
from torch.utils.data import DataLoader

from viscoin.cli.utils import batch_size, device, viscoin_pickle_path
from viscoin.datasets.cub import CUB_200_2011
from viscoin.models.utils import load_viscoin_pickle
from viscoin.testing.concepts import test_concepts


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
    force: bool,
    device: str,
    viscoin_pickle_path: str,
    batch_size: int = 32,
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
