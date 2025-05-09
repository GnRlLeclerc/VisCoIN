import json
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from viscoin.datasets.utils import DatasetType
from viscoin.models.clip import CLIP
from viscoin.models.utils import VisCoINModels
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
    models: VisCoINModels,
    concept2clip: nn.Module,
    clip_model: CLIP,
    latent_type: Literal["viscoin", "gan"],
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

    train_concept_spaces, test_concept_spaces = None, None

    match latent_type:
        case "viscoin":
            train_concept_spaces, test_concept_spaces = models.compute_concept_space(
                dataset, device, batch_size
            )
        case "gan":
            train_concept_spaces, test_concept_spaces = models.compute_w_space(dataset, device)

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
        logger.info(json.dumps(data))

        progress.set_postfix(
            train_loss=train_loss,
            test_loss=test_loss,
            best_loss=best_loss,
            matching_accuracy=matching_accuracy,
        )

    # Load the best model
    print(f"Best test loss: {best_loss:.4f}")
    concept2clip.load_state_dict(best_model)
