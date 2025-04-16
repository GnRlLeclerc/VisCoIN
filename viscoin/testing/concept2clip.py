"""Classifiers testing functions"""

import clip
import numpy as np
import torch
from clip.model import CLIP
from torch.nn import functional as F
from torch.types import Number
from torch.utils.data import DataLoader
from tqdm import tqdm

from viscoin.datasets.cub import CUB_200_2011
from viscoin.models.classifiers import Classifier
from viscoin.models.concept2clip import Concept2CLIP
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.utils.metrics import cosine_matching


def test_concept2clip(
    concept2clip: Concept2CLIP,
    loader: DataLoader,
    device: str,
    verbose: bool = True,
) -> tuple[float, float]:
    """Test concept2clip model on the given dataset.

    Cosine matching metric is computed accross batches.

    WARNING: SHUFFLE THE TEST DATALOADER!
    On some datasets like CUB, pictures of birds of the same classes follow each other
    in the dataset. Because concept2clip is about contrastive learning, testing it on
    batches of very similar pictures will inevitably yield bad scores.

    Args:
        concept2clip: trained clip adapter model to be tested
        loader: TensorDataset DataLoader of precomputed concept spaces and CLIP embeddings
        device: the device to use for the testing
        verbose: whether to print the progress bar (default: True)

    Returns:
        loss: average l2 mse loss
        accuracy: average cosine matching accuracy
    """

    concept2clip.eval()
    batch_size = loader.batch_size

    assert batch_size is not None

    with torch.no_grad():
        loss = 0
        matching_accuracy = 0

        for concepts, embeddings in tqdm(loader, desc="Test batches", disable=not verbose):
            # Move batch to device
            concepts, embeddings = concepts.to(device), embeddings.to(device)

            output = concept2clip(concepts)

            # Update metrics
            loss += F.mse_loss(output, embeddings).item() / batch_size
            # No need to divide by batch size, cosine matching is already normalized
            matching_accuracy += cosine_matching(output, embeddings)

    loss /= len(loader)
    matching_accuracy /= len(loader)

    return loss, matching_accuracy
