"""Classifiers testing functions"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from viscoin.models.classifiers import Classifier
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.clip_adapter import ClipAdapter

from clip.model import CLIP
import clip


def test_adapter(
    clip_adapter: ClipAdapter,
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    clip_model: CLIP,
    dataloader: DataLoader,
    device: str,
    criterion: nn.Module | None = None,
    verbose: bool = True,
) -> tuple[float, float]:
    """Test the clip adapter performance across a testing Dataloader

    Args:
        model: the classifier model to test
        dataloader: the DataLoader containing the testing dataset
        device: the device to use for the testing
        criterion: the loss function to use for the test (default: nn.CrossEntropyLoss)
        verbose: whether to print the progress bar (default: True)

    Returns:
        accuracy: the accuracy of the model on the testing dataset
        batch_mean_loss: the mean loss per batch on the testing dataset
    """
    if criterion is None:
        criterion = nn.MSELoss()

    clip_adapter.eval()

    with torch.no_grad():
        total_loss = 0
        total_samples = 0

        for inputs, targets in tqdm(dataloader, desc="Test batches", disable=not verbose):
            # Move batch to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute logits & predictions
            # Compute real clip embeddings
            clip_embeddings = clip_model.encode_image(inputs)

            # Predicted clip embeddings
            classes, hidden = classifier.forward(inputs)
            concept_space, gan_helper_space = concept_extractor.forward(hidden[-3:])

            output = clip_adapter(concept_space.view(-1, concept_extractor.n_concepts * 9))

            # Update metrics
            total_loss += criterion(output, clip_embeddings).item()
            total_samples += targets.size(0)

    batch_mean_loss = total_loss / len(dataloader)

    return batch_mean_loss


def get_concept_labels_vocab(
    clip_adapter: ClipAdapter, clip_model: CLIP, vocab: list[str], n_concepts: int, device: str
) -> tuple[list[str], list[torch.Tensor]]:
    """Get the concept labels vocabulary for the adapter

    Args:
        clip_adapter: the trained clip adapter model
        clip_model: the loaded CLIP model
        vocab: the vocabulary from which to chose the concept labels
        n_concepts: the number of concepts
        device: the device to use for the computation

    Returns:
        concept_labels: a list containing the concept labels
        probs: the probabilities of the concept labels
    """

    concept_labels = []
    probs_per_concept = []

    clip_adapter.eval()

    # Tokenize the vocabulary and embed it
    tokenized_vocab = clip.tokenize(vocab).to(device)

    text_features = clip_model.encode_text(tokenized_vocab).float()
    normalized_text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    concept_space_embeddings = generate_concept_embeddings(n_concepts, method="soft")

    with torch.no_grad():
        for concept_embedding in concept_space_embeddings:

            # Get the clip embeddings from the concept embeddings using the adapter
            predicted_clip_embedding, mu, logvar = clip_adapter(concept_embedding.to(device))

            normalized_image_features = predicted_clip_embedding / predicted_clip_embedding.norm(
                dim=-1, keepdim=True
            )

            # Compute logits and probabilities
            logits_per_image = normalized_image_features @ normalized_text_features.T

            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            idx = probs.argmax()

            concept_labels.append(vocab[idx])

            probs_per_concept.append(list(probs))

    return concept_labels, probs_per_concept


def generate_concept_embeddings(n_concepts: int, method: str = "ones") -> list[torch.Tensor]:
    """Generate fake concept embeddings to obtain a label

    Args:
        n_concepts (int)
        method (str, optional): the method to use to generate the embeddings, by default, all values are zero except for the 9 corresponding to the concept which are 1.0.

    Returns:
        A list of tensors containing the concept embeddings
    """

    concepts = [torch.zeros(1, n_concepts * 9) for _ in range(n_concepts)]

    match method:

        case "ones":

            for j in range(n_concepts):
                concepts[j][0, j * 9 : (j + 1) * 9] = 1.0

        case "soft":

            for j in range(n_concepts):
                concepts[j][0, :] = 0.2
                concepts[j][0, j * 9 : (j + 1) * 9] = 0.8

        case _:
            raise ValueError(f"Invalid method: {method}")

    return concepts
