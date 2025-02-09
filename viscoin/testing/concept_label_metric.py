import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from viscoin.models.classifiers import Classifier
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.datasets.cub import CUB_200_2011

from viscoin.models.utils import load_viscoin_pickle

import numpy as np
import os
import random
import pandas as pd

BATCH_SIZE = 1000


class ViscoinWrapper(torch.nn.Module):
    """
    Wrapper class for VisCoIN model to pool the 3x3 concept embeddings to single values
    """

    def __init__(
        self,
        classifier: Classifier,
        concept_extractor: ConceptExtractor,
        pooling_method: str = "L2",
    ):
        """
        Args:
            classifier (Classifier)
            concept_extractor (ConceptExtractor)
            pooling_method (str, optional): Should be "L2", "L1", "avg" or "max". Defaults to "L2".
        """

        super(ViscoinWrapper, self).__init__()
        self.classifier = classifier
        self.concept_extractor = concept_extractor

        match pooling_method:
            case "L2":
                self.pool = torch.nn.LPPool1d(2, 9)
            case "L1":
                self.pool = torch.nn.LPPool1d(1, 9)
            case "avg":
                self.pool = torch.nn.AdaptiveAvgPool1d(1)
            case "max":
                self.pool = torch.nn.AdaptiveMaxPool1d(1)
            case _:
                raise ValueError("Pooling method not recognized")

        # Identity layer for which to save the activations
        self.identity = torch.nn.Identity()

    def to(self, device):
        self.classifier.to(device)
        self.concept_extractor.to(device)
        return self

    def forward(self, x):
        """
        Args:
            x (Tensor): Image tensor of shape (B, 3, 224, 224)
        """
        classes, hidden = self.classifier.forward(x)
        concept_space, gan_helper_space = self.concept_extractor.forward(hidden[-3:])

        # Concept Space is orginally of shape (B, n_concepts * 9), we convert it to (B, n_concepts, 9)
        concept_space = concept_space.view(-1, self.concept_extractor.n_concepts, 9)
        # We pool the 9x1 vectors to get a single value for each concept
        concept_space = self.pool(concept_space).squeeze(-1)
        # Identity layer for CLIP-Dissect to work
        concept_space = self.identity(concept_space)

        return concept_space, gan_helper_space


def get_activations(
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    dataset: CUB_200_2011,
    device: str,
    concept_pooling_method: str = "L2",
):
    """Returns the activations of the last layer of the concept extractor for the given dataset.
    This tells us for each image, how much it activates each concept.

    Args:
        classifier (Classifier)
        concept_extractor (ConceptExtractor)
        dataset (CUB_200_2011)
        device (str)
        concept_pooling_method (str) : Should be "L2", "L1", "avg" or "max". Defaults to "L2".
    """

    target_model = ViscoinWrapper(classifier, concept_extractor, concept_pooling_method).to(device)

    activations = []

    # Hook function to save the activations
    def hook_function(model, input, output):
        activations.append(output.detach())

    hook = target_model.identity.register_forward_hook(hook_function)

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, BATCH_SIZE, num_workers=8, pin_memory=True)):
            _, _ = target_model(images.to(device))

    hook.remove()

    return torch.cat(activations, dim=0).cpu().numpy()


def save_expert_annotations_score(
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    dataset: CUB_200_2011,
    device: str,
    save_path: str,
    concept_pooling_method: str = "L2",
) -> np.ndarray:
    """Save the expert annotations score for each neuron in the last layer of the concept extractor.

    Args:
        classifier (Classifier)
        concept_extractor (ConceptExtractor)
        dataset (CUB_200_2011)
        device (str)
        save_path (str): Path to save the expert annotations score
        concept_pooling_method (str) : Should be "L2", "L1", "avg" or "max". Defaults to "L2".
    """

    # Get the activations of the last layer of the concept extractor
    activations = get_activations(
        classifier, concept_extractor, dataset, device, concept_pooling_method
    )

    # Get the dataset's attributes
    indices = dataset.test_indexes if dataset.mode == "test" else dataset.train_indexes
    attributes = [np.array(attrs) for i, attrs in enumerate(dataset.attributes) if i in indices]

    # For each neuron, get the expert annotations from the images that activate it.
    expert_annotations_score = np.zeros(
        (activations.shape[1], len(attributes))
    )  # For each neuron, the score of each attribute
    # The score is computed by adding the similarity of a given image to its attributes indices in expert_annotations_score

    # Create a binary matrix (images x attributes) where each entry is 1 if the attribute is present
    attribute_matrix = np.zeros(
        (activations.shape[0], len(dataset.attributes_labels.keys())), dtype=np.float32
    )

    for img_idx, img_attrs in enumerate(attributes):
        attribute_matrix[img_idx, img_attrs - 1] = 1

    # Multiply activations with the attribute matrix
    expert_annotations_score = activations.T @ attribute_matrix

    # Normalize the scores
    expert_annotations_score /= expert_annotations_score.sum(axis=1, keepdims=True)

    # Save the expert annotations score
    np.save(save_path, expert_annotations_score)

    return expert_annotations_score


def topk_evaluation(
    expert_annotations_score: np.ndarray,
    concept_labels: pd.DataFrame,
    attribute_caption_to_id: dict,
    k: int = 5,
):
    """Evaluate the concept labels using the expert annotations score and the top-k method

    Args:
        expert_annotations_score (np.ndarray): expert annotations score for each neuron
        concept_labels (pd.DataFrame): concept labels to evaluate
        attribute_caption_to_id (dict): mapping of attribute captions to their ids
        k (int, optional): Defaults to 5.
    """

    # Get the top-k attributes for each neuron
    topk_attributes = np.argsort(expert_annotations_score, axis=1)[:, -k:]

    # Check if the predicted attribute is in the top-k expert attributes for each neuron
    accuracy = 0

    for neuron in range(expert_annotations_score.shape[0]):

        if (
            attribute_caption_to_id[concept_labels.iloc[neuron]["description"]]
            in topk_attributes[neuron]
        ):
            accuracy += 1

    return accuracy / expert_annotations_score.shape[1]


def study_neurons(
    neurons: list[int],
    expert_annotations_score: np.ndarray,
    concept_labels: pd.DataFrame,
    attribute_caption_to_id: dict,
    topk_value: int = 5,
):
    """Print the top-k expert attributes and the predicted description of the neurons provided

    Args:
        neurons (list[int])
        expert_annotations_score (np.ndarray)
        concept_labels (pd.DataFrame)
        attribute_caption_to_id (dict)
    """

    mean_similarity = 0
    top_k_proba = 0

    attribute_captions = list(attribute_caption_to_id.keys())

    # Get the top-k attributes for each neuron
    topk_attributes = np.argsort(expert_annotations_score, axis=1)[:, -topk_value:]

    for neuron in neurons:

        print("Neuron", neuron)
        print("Top k attributes :", [attribute_captions[j] for j in topk_attributes[neuron]])
        print(
            "Description :",
            concept_labels.iloc[neuron]["description"],
            " - similarity :",
            concept_labels.iloc[neuron]["similarity"],
        )

        mean_similarity += concept_labels.iloc[neuron]["similarity"]

        intopk = False

        if (
            attribute_caption_to_id[concept_labels.iloc[neuron]["description"]]
            in topk_attributes[neuron]
        ):
            intopk = True
            top_k_proba += 1

        print("In top k :", intopk)

    print(" ====== ")

    print("Mean similarity :", mean_similarity / len(neurons))
    print("Top k probability :", top_k_proba / len(neurons))


def evaluate_concept_labels(
    expert_annotations_score_path: str,
    viscoin_pkl_path: str,
    dataset_path: str,
    concept_labels_path: str,
    device: str,
    evaluation_method: str = "topk",
    topk_value: int = 5,
    neurons_to_study: list[int] = None,
):
    """Evaluate the concept labels using the expert annotations score and the specified method

    Args:
        expert_annotations_score_path (str): path to the expert annotations score file
        viscoin_pkl_path (str)
        dataset_path (str)
        concept_labels_path (str): path to the concept labels file to evaluate, should be a CSV file with header, "unit,description,similarity"
        method (str, optional): Defaults to "topk".
    """

    viscoin = load_viscoin_pickle(viscoin_pkl_path)
    dataset = CUB_200_2011(dataset_path, mode="test")

    if not os.path.isfile(expert_annotations_score_path):

        expert_annotations_score = save_expert_annotations_score(
            viscoin.classifier,
            viscoin.concept_extractor,
            dataset,
            device,
            expert_annotations_score_path,
            "L2",
        )

    else:

        expert_annotations_score = np.load(expert_annotations_score_path)

    # Load the concept labels
    concept_labels = pd.read_csv(concept_labels_path)

    get_attribute_caption = (
        lambda attr: f"{attr[1].replace("_", " ")}{attr[0].lstrip("has").replace("_", " ")}"
    )

    attributes_caption_to_id = {
        get_attribute_caption(attr.split("::")): i
        for i, attr in enumerate(dataset.attributes_labels.values())
    }

    match evaluation_method:
        case "topk":

            accuracy = topk_evaluation(
                expert_annotations_score, concept_labels, attributes_caption_to_id, topk_value
            )

            print(f"Top-{topk_value} accuracy: {accuracy}")

            if neurons_to_study is None or len(neurons_to_study) == 0:
                neurons_to_study = random.sample(range(expert_annotations_score.shape[0]), 5)

            study_neurons(
                neurons_to_study,
                expert_annotations_score,
                concept_labels,
                attributes_caption_to_id,
                topk_value,
            )

        case _:
            raise ValueError("Evaluation method not recognized")
