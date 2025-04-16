import os
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from viscoin.datasets.utils import DatasetType, get_dataloaders
from viscoin.models.classifiers import Classifier
from viscoin.models.clip import CLIP
from viscoin.models.concept2clip import Concept2CLIP
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.testing.concept2clip import test_concept2clip
from viscoin.utils.dataclasses import IgnoreNone
from viscoin.utils.logging import get_logger


@dataclass
class Concept2ClipTrainingParams(IgnoreNone):
    """Training parameters for the concept2clip model.

    The default parameters were observed to be the best for training on CUB-200-2011."""

    epochs: int = 30
    learning_rate: float = 1e-5
    batch_size: int = 32


def train_concept2clip(
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    concept2clip: Concept2CLIP,
    clip_model: CLIP,
    dataset: DatasetType,
    device: str,
    params: Concept2ClipTrainingParams,
):
    """Train the concept2clip model to rebuild CLIP embeddings from concept spaces.

    Args:
        classifier: viscoin classifier
        concept_extractor: viscoin concept extractor
        concept2clip: concept2clip model to train
        clip_model: CLIP model
        dataset_type: name of the dataset (for CLIP)
        train_loader: DataLoader containing the training dataset
        test_loader: DataLoader containing the testing dataset
        device: device to use for training
        params: training parameters
    """

    batch_size = params.batch_size

    ###############################################################################################
    #                                   PRECOMPUTE CONCEPT SPACES                                 #
    ###############################################################################################

    train_concept_spaces, test_concept_spaces = _compute_concept_spaces(
        classifier,
        concept_extractor,
        dataset,
        device,
        batch_size,
    )

    del classifier, concept_extractor  # free GPU memory
    torch.cuda.empty_cache()

    ###############################################################################################
    #                                  PRECOMPUTE CLIP EMBEDDINGS                                 #
    ###############################################################################################

    train_embeddings, test_embeddings = clip_model.compute_image_embeddings(dataset)
    del clip_model  # free GPU memory
    torch.cuda.empty_cache()

    ###############################################################################################
    #                                 ACTUAL CONCEPT2CLIP TRAINING                                #
    ###############################################################################################

    # Create dataloaders from the precomputed concept spaces and clip embeddings
    train_loader = DataLoader(
        TensorDataset(train_concept_spaces, train_embeddings), batch_size, shuffle=True
    )
    # NOTE: the test dataloader is shuffled to avoid testing contrastive batches of birds of the same class
    test_loader = DataLoader(
        TensorDataset(test_concept_spaces, test_embeddings), batch_size, shuffle=True
    )

    best_loss = float("inf")
    best_model = concept2clip.state_dict()
    logger = get_logger()

    optimizer = optim.Adam(concept2clip.parameters(), lr=params.learning_rate)

    for _ in (progress := tqdm(range(1, params.epochs + 1), "Training Concept2CLIP")):
        ###########################################################################################
        #                                      TRAINING STEP                                      #
        ###########################################################################################

        concept2clip.train()

        # Training metrics for this epoch
        train_loss = 0

        for concepts, embeddings in train_loader:
            # Move batch to device
            concepts, embeddings = concepts.to(device), embeddings.to(device)

            # Generate clip embeddings from concept embeddings
            output = concept2clip(concepts)

            # Optimize the model
            optimizer.zero_grad()
            loss = F.mse_loss(output, embeddings)
            loss.backward()
            optimizer.step()

            # Accumulate loss for metrics
            train_loss += loss.item() / batch_size

        # Compute the mean loss for this epoch
        train_loss /= len(train_loader)

        ###########################################################################################
        #                                       TESTING STEP                                      #
        ###########################################################################################

        test_loss, matching_accuracy = test_concept2clip(
            concept2clip,
            test_loader,
            device,
            False,
        )

        # Save the model state_dict if it performs best
        if test_loss < best_loss:  # type: ignore
            best_model = concept2clip.state_dict()
            best_loss = test_loss

        data = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "matching_accuracy": matching_accuracy,
        }

        # Log the current state of training in jsonl format for easy plotting
        logger.info(data)

        progress.set_postfix(
            train_loss=train_loss,
            test_loss=test_loss,
            best_loss=best_loss,
            matching_accuracy=matching_accuracy,
        )

    # Load the best model
    print(f"Best test loss: {best_loss:.4f}")
    concept2clip.load_state_dict(best_model)


def _cache(mode: Literal["train", "test"], dataset: str) -> str:
    return f"checkpoints/concepts/{dataset}_{mode}.pt"


def _compute_concept_spaces(
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    dataset: DatasetType,
    device: str,
    batch_size,
) -> tuple[Tensor, Tensor]:
    """Precompute the concept spaces for the whole training and testing image sets.

    Args:
        classifier: viscoin classifier
        concept_extractor: viscoin concept extractor
        dataset: name of the dataset (for cache path)
        device: device to use for training

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
        train_concept_spaces[i * batch_size : (i + 1) * batch_size] = concept_space.detach().cpu()

    for i, (inputs, _) in enumerate(tqdm(test, desc="Precomputing testing embeddings")):
        inputs = inputs.to(device)
        _, hidden = classifier.forward(inputs)
        concept_space, _ = concept_extractor.forward(hidden[-3:])
        test_concept_spaces[i * batch_size : (i + 1) * batch_size] = concept_space.detach().cpu()

    os.makedirs("checkpoints/concepts", exist_ok=True)
    torch.save(train_concept_spaces, _cache("train", dataset))
    torch.save(test_concept_spaces, _cache("test", dataset))

    return train_concept_spaces, test_concept_spaces
