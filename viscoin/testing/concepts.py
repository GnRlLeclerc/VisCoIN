"""Testing concept repartition in viscoin"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from viscoin.models.classifiers import Classifier
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer
from viscoin.utils.maths import normalize


@dataclass
class ConceptTestResults:
    """Results of the concept test
    Every result is normalized between 0 and 1.

    Args:
        classifier_accuracy: The accuracy of the classifier model.
        explainer_accuracy: The accuracy of the explainer model.
        concept_activation_per_image: (n_concepts) The sorted curve of average concept activation per image (see how many concepts are used per image).
        concept_activation_per_concept: (n_concepts) The sorted average activation of each concept per image (see dead concepts).
        raw_concept_mean_activation (n_concepts) The mean activation of each concept over the whole dataset, in the order of concept_correlations.
        concept_correlations: (n_concepts, n_concepts) The correlation of each concept with each other. Normalized.
        class_concept_correlations: (n_classes, n_concepts) The correlation of each concept with each class. Normalized per class, so that each row (class) represents the relative activation of every concept for this class.
        concept_class_correlations: (n_concepts, n_classes) The correlation of each class with each concept. Normalized per concept, so that each row (concept) represents the relative activation of this concept for every class.
        concept_entropy: (n_concepts) The entropy of each concept activation, normalized over the concepts.
        class_counts: (n_classes) The number of images per class in the dataset.
    """

    classifier_accuracy: float
    explainer_accuracy: float
    concept_activation_per_image: np.ndarray
    concept_activation_per_concept: np.ndarray
    raw_concept_mean_activation: np.ndarray
    concept_correlations: np.ndarray
    class_concept_correlations: np.ndarray
    concept_class_correlations: np.ndarray
    concept_entropy: np.ndarray
    class_counts: np.ndarray


def test_concepts(
    # Models
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    explainer: Explainer,
    # Loader
    dataloader: DataLoader,
    device: str,
) -> ConceptTestResults:
    """Test viscoin concept repartition.

    Args:
        classifier: The classifier model.
        concept_extractor: The concept extractor model.
        explainer: The explainer model.
        dataloader: The testing DataLoader.
        device: The device to use.

    Returns:
        ConceptTestResults: The results of the concept test.
    """

    # Put the models in evaluation mode
    classifier.eval()
    concept_extractor.eval()
    explainer.eval()

    n_concepts = concept_extractor.n_concepts
    n_classes = classifier.linear.weight.shape[0]

    # For each image, add the sorted probabilities of the concept to have an idea of how many concepts are activated at once
    concept_activation_per_image = np.zeros(n_concepts)
    # For each concept, measure the activation intensity (via probability), averaged over all images to compute 'dead concepts'
    concept_activation_per_concept = np.zeros(n_concepts)
    # Correlated activation of concepts in a heatmap
    concept_correlations = np.zeros((n_concepts, n_concepts))
    # Correlated activation of concepts for each class
    class_concept_correlations = np.zeros((n_classes, n_concepts))
    # Class counts in the dataset
    class_counts = np.zeros(n_classes)

    # Compare accuracies
    classifier_accuracies: list[float] = []
    explainer_accuracies: list[float] = []

    for images, labels in tqdm(dataloader, desc="Concept test batches"):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            classes, latent = classifier.forward(images)
            encoded_concepts, _ = concept_extractor.forward(latent[-3:])
            explainer_classes = explainer.forward(encoded_concepts)

        preds = classes.argmax(dim=1, keepdim=True)
        preds_expl = explainer_classes.argmax(dim=1, keepdim=True)

        # Compute the accuracy
        classifier_accuracy = preds.eq(labels.view_as(preds)).sum().item() / len(labels)
        explainer_accuracy = preds_expl.eq(labels.view_as(preds_expl)).sum().item() / len(labels)
        classifier_accuracies.append(classifier_accuracy)
        explainer_accuracies.append(explainer_accuracy)

        # Compute concept activations and correlations
        for image_concepts, label in zip(encoded_concepts, labels):
            # (n_concepts)
            activations = F.adaptive_max_pool2d(image_concepts, 1).squeeze().cpu().numpy()
            label = label.cpu().item()

            concept_activation_per_image += np.sort(activations)
            concept_activation_per_concept += activations
            concept_correlations += np.outer(activations, activations)
            class_concept_correlations[label] += activations
            class_counts[label] += 1

    # (n_classes, n_concepts)
    balanced_class_concept_correlations = (
        class_concept_correlations / class_counts[:, None] * class_counts.max()
    )

    # Now, before normalizing, compute the entropy of each concept
    entropies = -np.sum(
        balanced_class_concept_correlations * np.log(balanced_class_concept_correlations + 1e-6),
        axis=0,
    )

    # Normalize by sum to obtain probabilities
    return ConceptTestResults(
        classifier_accuracy=float(np.mean(classifier_accuracies)),
        explainer_accuracy=float(np.mean(explainer_accuracies)),
        concept_activation_per_image=normalize(concept_activation_per_image),
        concept_activation_per_concept=np.sort(normalize(concept_activation_per_concept)),
        raw_concept_mean_activation=normalize(concept_activation_per_concept),
        concept_correlations=normalize(concept_correlations),
        # Normalize concept activation per class (insensitive to class imbalance)
        class_concept_correlations=normalize(class_concept_correlations, axis=1),
        # Normalize concept activation per concept (sensitive to class imbalance, hence the use of class counts)
        concept_class_correlations=normalize(balanced_class_concept_correlations, axis=0).T,
        class_counts=class_counts,
        concept_entropy=normalize(entropies),
    )
