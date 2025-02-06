from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass

import torch
from viscoin.models.clip_adapter import ClipAdapter, ClipAdapterVAE
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.classifiers import Classifier
from viscoin.utils.logging import get_logger

from viscoin.training.losses import vae_reconstruction_loss

from viscoin.testing.clip_adapter import test_adapter

from clip.model import CLIP


@dataclass
class ClipAdapterTrainingParams:
    epochs: int = 30
    learning_rate: float = 1e-5

    train_criterion = nn.MSELoss()
    test_criterion = nn.MSELoss()


@dataclass
class ClipAdapterVAETrainingParams:
    epochs: int = 30
    learning_rate: float = 1e-5

    # output = (reconstruction, mu, logvar)
    def train_criterion(self, output, clip_embedd):
        return vae_reconstruction_loss(output[0], clip_embedd, output[1], output[2])

    def test_criterion(self, output, clip_embedd):
        return nn.functional.mse_loss(output[0], clip_embedd)


def train_clip_adapter_cub(
    clip_adapter: ClipAdapter | ClipAdapterVAE,
    concept_extractor: ConceptExtractor,
    classifier: Classifier,
    clip_model: CLIP,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    params: ClipAdapterTrainingParams | ClipAdapterVAETrainingParams,
):
    """Train the adapter to convert concept embeddings to clip embeddings.

    Note: the losses are averaged over batches.

    Args:
        model: the classifier model to train
        clip_model: the loaded CLIP model
        train_loader: the DataLoader containing the training dataset
        test_loader: the DataLoader containing the testing dataset
        device: the device to use for training
        epochs: the number of epochs to train the model
        learning_rate: the learning rate for the optimizer
    """
    best_loss = float("inf")
    best_model = clip_adapter.state_dict()
    logger = get_logger()

    # Optimizer and scheduler
    optimizer = optim.Adam(clip_adapter.parameters(), lr=params.learning_rate)

    for epoch in (progress := tqdm(range(1, params.epochs + 1), "Training epochs")):
        ###########################################################################################
        #                                      TRAINING STEP                                      #
        ###########################################################################################

        clip_adapter.train()

        # Training metrics for this epoch
        total_loss = 0
        total_samples = 0

        for inputs, _ in train_loader:
            # Move batch to device
            inputs = inputs.to(device)

            # Compute real clip embeddings
            clip_embeddings = clip_model.encode_image(inputs).float()

            # Predicted clip embeddings
            classes, hidden = classifier.forward(inputs)
            concept_space, gan_helper_space = concept_extractor.forward(hidden[-3:])

            output = clip_adapter(concept_space.view(-1, concept_extractor.n_concepts * 9))

            # Compute logits
            optimizer.zero_grad()

            loss = params.train_criterion(output, clip_embeddings)

            current_loss = loss.item()

            # Compute loss and backpropagate
            loss.backward()
            optimizer.step()

            # Update training metrics
            total_loss += current_loss
            total_samples += inputs.size(0)

        # Append training metrics
        batch_mean_loss = total_loss / len(train_loader)

        ###########################################################################################
        #                                       TESTING STEP                                      #
        ###########################################################################################

        mean_loss = test_adapter(
            clip_adapter,
            classifier,
            concept_extractor,
            clip_model,
            test_loader,
            device,
            params.test_criterion,
            False,
        )

        if loss < best_loss:
            best_model = clip_adapter.state_dict()
            best_loss = mean_loss

        # Log the current state of training
        logger.info(
            f"Epoch {epoch}/{params.epochs} - Train Loss: {batch_mean_loss:.4f} - Test Loss: {mean_loss:.4f}"
        )

        progress.set_postfix(train_loss=batch_mean_loss, test_loss=mean_loss, best_loss=best_loss)

    # Load the best model
    print(f"Best test loss: {best_loss:.4f}")
    clip_adapter.load_state_dict(best_model)
