from dataclasses import dataclass

import torch
import torchvision.transforms
from clip.model import CLIP
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from viscoin.models.classifiers import Classifier
from viscoin.models.concept2clip import Concept2CLIP
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.gan import GeneratorAdapted
from viscoin.testing.concept2clip import test_concept2clip
from viscoin.training.losses import InfoNCE
from viscoin.utils.logging import get_logger


@dataclass
class Concept2ClipTrainingParams:
    epochs: int = 30
    learning_rate: float = 1e-5

    train_criterion = nn.MSELoss()
    test_criterion = nn.MSELoss()


def train_concept2clip(
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    concept2clip: Concept2CLIP,
    clip_model: CLIP,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    params: Concept2ClipTrainingParams,
):
    """Train the adapter to convert concept embeddings to clip embeddings.

    Note: the losses are averaged over batches.

    Args:
        classifier: viscoin classifier
        concept_extractor: viscoin concept extractor
        concept2clip: concept2clip model to train
        clip_model: CLIP model used to compute embeddings
        train_loader: DataLoader containing the training dataset
        test_loader: DataLoader containing the testing dataset
        device: device to use for training
        params: training parameters
    """
    best_loss = float("inf")
    best_model = concept2clip.state_dict()
    logger = get_logger()

    # Optimizer and scheduler
    optimizer = optim.Adam(concept2clip.parameters(), lr=params.learning_rate)

    resizer = torchvision.transforms.Resize((224, 224))

    contrastive_criterion = InfoNCE()

    for epoch in (progress := tqdm(range(1, params.epochs + 1), "Training epochs")):
        ###########################################################################################
        #                                      TRAINING STEP                                      #
        ###########################################################################################

        concept2clip.train()

        # Training metrics for this epoch
        total_loss = 0
        total_samples = 0

        for inputs, _ in train_loader:
            # Move batch to device
            inputs = inputs.to(device)

            # Compute real clip embeddings
            clip_embeddings = clip_model.encode_image(inputs).float()

            # Predicted clip embeddings
            _, hidden = classifier.forward(inputs)
            concept_space, extra_info = concept_extractor.forward(hidden[-3:])

            # Compute the reconstructed images and their clip embeddings
            rebuilt_images, _ = viscoin_gan.forward(
                z1=concept_space, z2=extra_info, return_latents=True
            )

            # Resize images to 224x224 to match CLIP input size
            rebuilt_images = resizer(rebuilt_images)

            rebuilt_images_clip_embeddings = clip_model.encode_image(rebuilt_images).float()

            # Generate clip embeddings from concept embeddings
            output = concept2clip(concept_space.view(-1, concept_extractor.n_concepts * 9))

            optimizer.zero_grad()

            # Computing the loss between the predicted clip embeddings and the real clip embeddings and the contrastive loss
            embedding_loss = params.train_criterion(
                output, clip_embeddings
            ) + params.train_criterion(output, rebuilt_images_clip_embeddings)

            contrastive_loss = contrastive_criterion(output, clip_embeddings)

            loss = embedding_loss + contrastive_loss

            current_loss = loss.item()

            # Compute loss and backpropagate
            loss.backward()

            optimizer.step()

            # Update training metrics
            total_loss += current_loss
            total_samples += inputs.size(0)

            progress.set_description_str(
                f"Training epochs {total_samples}/{len(train_loader.dataset)}"  # type: ignore
            )

        # Append training metrics
        batch_mean_loss = total_loss / len(train_loader)

        ###########################################################################################
        #                                       TESTING STEP                                      #
        ###########################################################################################

        mean_loss, matching_accuracy = test_concept2clip(
            concept2clip,
            classifier,
            concept_extractor,
            clip_model,
            test_loader,
            device,
            params.test_criterion,  # type: ignore
            False,
        )

        if mean_loss < best_loss:  # type: ignore
            best_model = concept2clip.state_dict()
            best_loss = mean_loss

        # Log the current state of training
        logger.info(
            f"Epoch {epoch}/{params.epochs} - Train Loss: {batch_mean_loss:.4f} - Test Loss: {mean_loss:.4f} - Matching Accuracy: {matching_accuracy:.4f}"
        )

        progress.set_postfix(
            train_loss=batch_mean_loss,
            test_loss=mean_loss,
            best_loss=best_loss,
            matching_accuracy=matching_accuracy,
        )

    # Load the best model
    print(f"Best test loss: {best_loss:.4f}")
    concept2clip.load_state_dict(best_model)


def get_average_clip_embedding(loader: DataLoader, clip_model: CLIP, device: str) -> torch.Tensor:
    """Compute the average clip embedding of a dataset

    Args:
        loader: the DataLoader containing the dataset
        clip_model: the loaded CLIP model
        device
    """
    clip_model.eval()

    with torch.no_grad():
        total_clip_embeddings = torch.zeros(512).to(device)
        total_samples = 0

        for inputs, _ in tqdm(loader, desc="Computing average clip embedding"):
            # Move batch to device
            inputs = inputs.to(device)

            # Compute real clip embeddings
            clip_embeddings = clip_model.encode_image(inputs).float()

            total_clip_embeddings += clip_embeddings.sum(dim=0)
            total_samples += inputs.size(0)

    return total_clip_embeddings / total_samples
