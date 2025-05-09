"""Model utilities."""

import os
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from tqdm import tqdm

from viscoin.datasets.utils import DatasetType, get_dataloaders
from viscoin.models.classifiers import Classifier
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer
from viscoin.models.gan import GeneratorAdapted


@dataclass
class VisCoINModels:
    """Dataclass to store the VisCoIN models together for pickling."""

    classifier: Classifier
    concept_extractor: ConceptExtractor
    explainer: Explainer
    gan: GeneratorAdapted

    def compute_w_space(
        self,
        dataset: DatasetType,
        device: str,
    ) -> tuple[Tensor, Tensor]:
        """Compute and return the train and test W+ spaces of the GAN for the given dataset.
        The results are cached under checkpoints/gan-w/

        Returns:
            train_w: W+ space for the training set
            test_w: W+ space for the testing set
        """

        try:
            return (
                torch.load(f"checkpoints/gan-w/{dataset}-train.pt", weights_only=True),
                torch.load(f"checkpoints/gan-w/{dataset}-test.pt", weights_only=True),
            )
        except FileNotFoundError:
            pass

        gan = self.gan.to(device)
        classifier = self.classifier.to(device)
        concept_extractor = self.concept_extractor.to(device)
        batch_size = 4

        train_loader, test_loader = get_dataloaders(dataset, batch_size, "test")

        train_w = torch.zeros((len(train_loader.dataset), gan.num_ws, gan.w_dim))  # type: ignore
        test_w = torch.zeros((len(test_loader.dataset), gan.num_ws, gan.w_dim))  # type: ignore

        for i, (img, _) in enumerate(
            tqdm(
                train_loader,
                desc=f"Computing W+ space for {dataset} - train",
            )
        ):
            img = img.to(device)

            with torch.no_grad():
                _, latents = classifier.forward(img)
                concepts, extra_info = concept_extractor.forward(latents[-3:])
                _, latents = gan.forward(
                    z1=concepts,
                    z2=extra_info,
                    return_latents=True,
                )

            train_w[i * batch_size : (i + 1) * batch_size] = latents.detach().cpu()

        for i, (img, _) in enumerate(
            tqdm(
                test_loader,
                desc=f"Computing W+ space for {dataset} - test",
            )
        ):
            img = img.to(device)

            with torch.no_grad():
                _, latents = classifier.forward(img)
                concepts, extra_info = concept_extractor.forward(latents[-3:])
                _, latents = gan.forward(
                    z1=concepts,
                    z2=extra_info,
                    return_latents=True,
                )

            test_w[i * batch_size : (i + 1) * batch_size] = latents.detach().cpu()

        # Save the W+ spaces
        os.makedirs("checkpoints/gan-w", exist_ok=True)
        torch.save(train_w, f"checkpoints/gan-w/{dataset}-train.pt")
        torch.save(test_w, f"checkpoints/gan-w/{dataset}-test.pt")

        # Move models back to cpu to free gpu memory
        gan.cpu()
        classifier.cpu()
        concept_extractor.cpu()

        return train_w, test_w

    def compute_concept_space(
        self,
        dataset: DatasetType,
        device: str,
        batch_size: int,
    ):
        """Precompute the concept spaces for the whole training and testing image sets.

        Args:
            dataset: name of the dataset (for cache path)
            device: device to use for training
            batch_size: batch size to use for computation

        The results are cached under checkpoints/concepts/

        Returns:
            train_concept_spaces: concept spaces for the training set (on CPU)
            test_concept_spaces: concept spaces for the testing set (on CPU)
        """

        try:
            train_embeddings = torch.load(_cache("train", dataset), weights_only=True)
            test_embeddings = torch.load(_cache("test", dataset), weights_only=True)
            return train_embeddings, test_embeddings
        except FileNotFoundError:
            pass

        # Use both datasets in test transform mode, no shuffling
        train, test = get_dataloaders(dataset, batch_size, "test")

        concept_extractor = self.concept_extractor.to(device)
        classifier = self.classifier.to(device)

        n_concepts = concept_extractor.n_concepts
        len_train = len(train_loader.dataset)  # type: ignore
        len_test = len(test_loader.dataset)  # type: ignore

        train_concept_spaces = torch.zeros((len_train, n_concepts, 3, 3))  # type: ignore
        test_concept_spaces = torch.zeros((len_test, n_concepts, 3, 3))  # type: ignore

        classifier.eval()
        concept_extractor.eval()

        for i, (inputs, _) in enumerate(tqdm(train, desc="Precomputing training embeddings")):
            inputs = inputs.to(device)
            _, hidden = classifier.forward(inputs)
            concept_space, _ = concept_extractor.forward(hidden[-3:])
            train_concept_spaces[i * batch_size : (i + 1) * batch_size] = (
                concept_space.detach().cpu()
            )

        for i, (inputs, _) in enumerate(tqdm(test, desc="Precomputing testing embeddings")):
            inputs = inputs.to(device)
            _, hidden = classifier.forward(inputs)
            concept_space, _ = concept_extractor.forward(hidden[-3:])
            test_concept_spaces[i * batch_size : (i + 1) * batch_size] = (
                concept_space.detach().cpu()
            )

        os.makedirs("checkpoints/concepts", exist_ok=True)
        torch.save(train_concept_spaces, _cache("train", dataset))
        torch.save(test_concept_spaces, _cache("test", dataset))

        # Move models back to cpu to free gpu memory
        concept_extractor.cpu()
        classifier.cpu()

        return train_concept_spaces, test_concept_spaces


def save_viscoin(
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    explainer: Explainer,
    gan: GeneratorAdapted,
    path: str,
):
    """Jointly save the checkpoints of the VisCoIN models."""
    checkpoints = {
        "classifier": classifier.state_dict(),
        "concept_extractor": concept_extractor.state_dict(),
        "explainer": explainer.state_dict(),
        "gan": gan.state_dict(),
    }

    torch.save(checkpoints, path)


def load_viscoin(
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    explainer: Explainer,
    gan: GeneratorAdapted,
    path: str,
):
    """Jointly load the checkpoints of the VisCoIN models."""
    checkpoints = torch.load(path, weights_only=True)

    classifier.load_state_dict(checkpoints["classifier"])
    concept_extractor.load_state_dict(checkpoints["concept_extractor"])
    explainer.load_state_dict(checkpoints["explainer"])
    gan.load_state_dict(checkpoints["gan"])


def save_viscoin_pickle(
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    explainer: Explainer,
    gan: GeneratorAdapted,
    path: str,
):
    """Jointly pickle the VisCoIN models to also store their parameters."""
    bundle = VisCoINModels(
        classifier=classifier,
        concept_extractor=concept_extractor,
        explainer=explainer,
        gan=gan,
    )

    torch.save(bundle, path)


def load_viscoin_pickle(
    path: str,
) -> VisCoINModels:
    """Jointly load the VisCoIN models from a pickle file."""
    return torch.load(path, weights_only=False)


def _cache(mode: Literal["train", "test"], dataset: str) -> str:
    """Concept space cache path."""
    return f"checkpoints/concepts/{dataset}_{mode}.pt"
