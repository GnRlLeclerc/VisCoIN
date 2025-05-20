"""Classifiers testing functions"""

import clip
import torch
from clip.model import CLIP
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from viscoin.models.classifiers import Classifier
from viscoin.models.concept2clip import Concept2CLIP
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.utils.metrics import cosine_matching
from viscoin.datasets.cub import CUB_200_2011


def test_adapter(
    clip_adapter: Concept2CLIP,
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
        matching_accuracy = 0

        for inputs, targets in tqdm(dataloader, desc="Test batches", disable=not verbose):
            # Move batch to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute logits & predictions
            # Compute real clip embeddings
            clip_embeddings = clip_model.encode_image(inputs)

            # Predicted clip embeddings
            _, hidden = classifier.forward(inputs)
            concept_space, _ = concept_extractor.forward(hidden[-3:])

            output = clip_adapter(concept_space.view(-1, concept_extractor.n_concepts * 9))

            # Update metrics
            total_loss += criterion(output, clip_embeddings).item()
            matching_accuracy += cosine_matching(output, clip_embeddings)
            total_samples += targets.size(0)

    batch_mean_loss = total_loss / len(dataloader)

    return (batch_mean_loss, matching_accuracy / len(dataloader))


def get_concept_labels_vocab(
    clip_adapter: Concept2CLIP,
    concept_extractor: ConceptExtractor,
    classifier: Classifier,
    clip_model: CLIP,
    vocab: list[str],
    n_concepts: int,
    dataset: CUB_200_2011,
    multiplier: int,
    selection_n: int,
    device: str,
) -> tuple[list[str], list[torch.Tensor]]:
    """Retrieve concept labels for VisCoIN concepts from CLIP embeddings

    Args:
        clip_adapter: the trained clip adapter model
        clip_model: the loaded CLIP model
        vocab: the vocabulary from which to chose the concept labels
        n_concepts: the number of concepts
        dataset
        multiplier: the multiplier to amplify to the concept
        selection_n: the number of most activating images to select per concept
        device: the device to use for the computation

    Returns:
        concept_labels: a list containing the concept labels
        probs: the probabilities of the concept labels
    """

    n_vocab = len(vocab)
    batch_size = 128
    mean_concept_label_similarity = torch.zeros((n_concepts, n_vocab)).to(device)
    image_concepts = torch.zeros((len(dataset), n_concepts, 3, 3))
    most_activating_images_per_concept = []  # List to store selected image indices for each concept

    clip_adapter.eval()

    with torch.no_grad():

        # Pre-compute vocabulary CLIP embeddings:
        vocab_embeddings = torch.zeros((n_vocab, 512)).to(device)
        for i, label in enumerate(vocab):
            text_features = clip_model.encode_text(clip.tokenize(label).to(device)).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            vocab_embeddings[i] = text_features

        # Step 1: Extract Concepts for all images
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for i, (image, _) in tqdm(
            enumerate(dataloader), desc="Computing concepts for each image", total=len(dataloader)
        ):
            image = image.to(device)
            _, hidden = classifier.forward(image)
            concept_space, _ = concept_extractor.forward(hidden[-3:])
            image_concepts[i * batch_size : (i + 1) * batch_size] = concept_space

        # Step 2: Select selection_n most activating images for each concept
        pooled_image_concepts = torch.mean(image_concepts, dim=(2, 3)).transpose(0, 1)

        most_activating_images_per_concept = torch.argsort(
            pooled_image_concepts, dim=1, descending=True
        )[:, :selection_n]

        # Step 3: Process each concept and selected images
        for concept_idx in tqdm(range(n_concepts), desc="Processing concepts"):

            selected_indices = most_activating_images_per_concept[concept_idx]

            embedding_differences = torch.zeros((len(selected_indices), 512)).to(device)

            for i, image_idx in enumerate(selected_indices):

                image = dataset[image_idx][0].to(device)
                original_concept = image_concepts[image_idx]

                # Compute original CLIP embedding
                clip_embedding = clip_adapter(
                    original_concept.view(-1, concept_extractor.n_concepts * 9).to(device)
                )

                # Compute amplified concept and its CLIP embedding ---
                amplified_concept = original_concept.clone()
                amplified_concept[concept_idx] *= multiplier
                amplified_clip_embedding = clip_adapter(
                    amplified_concept.view(-1, concept_extractor.n_concepts * 9).to(device)
                )

                # Compute embedding difference
                embedding_difference = amplified_clip_embedding - clip_embedding
                if embedding_difference.norm(dim=-1, keepdim=True) != 0:
                    embedding_difference /= embedding_difference.norm(dim=-1, keepdim=True)

                if embedding_difference.isnan().any():
                    print("NANs in embedding difference")

                embedding_differences[i] = embedding_difference

            # Compute similarity with vocab
            similarities = embedding_differences @ vocab_embeddings.T

            mean_concept_label_similarity[concept_idx] = torch.nn.functional.softmax(
                torch.mean(similarities, dim=0), dim=0
            )

    # Step 5: Get top vocab words
    concept_labels = []
    probs = []
    for concept_idx in range(n_concepts):
        top_idx = torch.argmax(mean_concept_label_similarity[concept_idx])
        concept_labels.append(vocab[top_idx])
        probs.append(mean_concept_label_similarity[concept_idx, top_idx])

    return concept_labels, probs, most_activating_images_per_concept.cpu().numpy()
