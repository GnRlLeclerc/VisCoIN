import json
from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from viscoin.models.classifiers import Classifier
from viscoin.models.clip import CLIP
from viscoin.models.concept2clip import Concept2CLIP
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.testing.concept2clip import test_concept2clip
from viscoin.training.losses import InfoNCE
from viscoin.utils.logging import get_logger


@dataclass
class Concept2ClipTrainingParams:
    epochs: int = 30
    learning_rate: float = 1e-5
    criterion: nn.Module = InfoNCE()


def train_concept2clip(
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    concept2clip: Concept2CLIP,
    clip_model: CLIP,
    dataset_type: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
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

    ###############################################################################################
    #                                   PRECOMPUTE CONCEPT SPACES                                 #
    ###############################################################################################

    # 1. Precompute the CLIP embeddings & concept spaces for the whole training and testing sets
    # We assume that the whole dataset fits into CPU memory when converted into these 2 spaces
    n_concepts = concept_extractor.n_concepts
    len_train = len(train_loader.dataset)  # type: ignore
    len_test = len(test_loader.dataset)  # type: ignore
    batch_size = train_loader.batch_size

    assert (train_loader.batch_size == test_loader.batch_size) and (batch_size is not None)

    train_concept_spaces = torch.zeros((len_train, n_concepts, 3, 3))  # type: ignore
    test_concept_spaces = torch.zeros((len_test, n_concepts, 3, 3))  # type: ignore

    classifier.eval()
    concept_extractor.eval()

    for i, (inputs, _) in enumerate(tqdm(train_loader, desc="Precomputing training embeddings")):
        inputs = inputs.to(device)
        _, hidden = classifier.forward(inputs)
        concept_space, _ = concept_extractor.forward(hidden[-3:])
        train_concept_spaces[i * batch_size : (i + 1) * batch_size] = concept_space.detach().cpu()

    for i, (inputs, _) in enumerate(tqdm(test_loader, desc="Precomputing testing embeddings")):
        inputs = inputs.to(device)
        _, hidden = classifier.forward(inputs)
        concept_space, _ = concept_extractor.forward(hidden[-3:])
        test_concept_spaces[i * batch_size : (i + 1) * batch_size] = concept_space.detach().cpu()

    del train_loader, test_loader  # free memory
    del classifier, concept_extractor  # free GPU memory
    torch.cuda.empty_cache()

    ###############################################################################################
    #                                  PRECOMPUTE CLIP EMBEDDINGS                                 #
    ###############################################################################################

    train_embeddings, test_embeddings = clip_model.compute_embeddings(dataset_type)  # type: ignore
    del clip_model  # free GPU memory
    torch.cuda.empty_cache()

    ###############################################################################################
    #                                 ACTUAL CONCEPT2CLIP TRAINING                                #
    ###############################################################################################

    # Create dataloaders from the precomputed concept spaces and clip embeddings
    train_concept_loader = DataLoader(TensorDataset(train_concept_spaces), batch_size)
    train_clip_loader = DataLoader(TensorDataset(train_embeddings), batch_size)
    test_concept_loader = DataLoader(TensorDataset(test_concept_spaces), batch_size)
    test_clip_loader = DataLoader(TensorDataset(test_embeddings), batch_size)

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

        for concepts, embeddings in zip(train_concept_loader, train_clip_loader):
            # Move batch to device
            concepts, embeddings = concepts.to(device), embeddings.to(device)

            # Generate clip embeddings from concept embeddings
            output = concept2clip(concepts)

            # Optimize the model
            optimizer.zero_grad()
            loss = params.criterion(output, embeddings)
            loss.backward()
            optimizer.step()

            # Accumulate loss for metrics
            train_loss += loss.item() / batch_size

        # Compute the mean loss for this epoch
        train_loss /= len(train_concept_loader)

        ###########################################################################################
        #                                       TESTING STEP                                      #
        ###########################################################################################

        test_loss, matching_accuracy = test_concept2clip(
            concept2clip,
            test_concept_loader,
            test_clip_loader,
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
