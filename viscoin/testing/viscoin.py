"""Testing functions for the viscoin ensemble."""

# pyright: reportPossiblyUnboundVariable=false

from dataclasses import dataclass
from typing import Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torch import Tensor
from torch.types import Number
from torch.utils.data import DataLoader
from tqdm import tqdm

from viscoin.models.gan import GeneratorAdapted, fix_path
from viscoin.utils.images import from_torch
from viscoin.utils.types import TestingResults

fix_path()

from stylegan2_ada.dnnlib.util import open_url
from stylegan2_ada.metrics.metric_utils import FeatureStats
from viscoin.models.classifiers import Classifier
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer
from viscoin.training.losses import (
    concept_regularization_loss,
    lpips_loss,
    output_fidelity_loss,
)


def test_viscoin(
    # Models
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    explainer: Explainer,
    viscoin_gan: GeneratorAdapted,
    # Loader
    dataloader: DataLoader,
    device: str,
    # FID loss
    compute_fid: bool = False,
    verbose: bool = True,
) -> TestingResults:
    """Test the classifier performance across a testing Dataloader

    Args:
        classifier: The classifier model.
        concept_extractor: The concept extractor model.
        explainer: The explainer model.
        viscoin_gan: The VisCoIN GAN model.
        dataloader: The testing DataLoader.
        device: The device to use.
        compute_fid: Whether to compute the Fréchet Inception Distance score.
        verbose: Whether to display the progress bar.

    Returns:
        The testing results in a dataclass.
    """

    # Put the models in evaluation mode
    classifier.eval()
    concept_extractor.eval()
    explainer.eval()
    viscoin_gan.eval()

    # Create the loss arrays
    acc_loss: list[Number] = []
    cr_loss: list[Number] = []
    of_loss: list[Number] = []
    lp_loss: list[Number] = []
    rec_loss_l1: list[Number] = []
    rec_loss_l2: list[Number] = []
    preds_overlap: list[Number] = []
    correct_preds: list[Number] = []
    correct_expl_preds: list[Number] = []

    # Prepare the feature detector model
    if compute_fid:
        detector_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"
        detector_kwargs = {"return_features": True}
        with open_url(detector_url, verbose=True) as f_detector:
            feature_detector = torch.jit.load(f_detector).eval().to(device)
        stats_real = FeatureStats(max_items=len(dataloader.dataset), capture_mean_cov=True)  # type: ignore
        stats_fake = FeatureStats(max_items=len(dataloader.dataset), capture_mean_cov=True)  # type: ignore

    for images, labels in tqdm(dataloader, desc="Viscoin test batches", disable=not verbose):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            classes, latent = classifier.forward(images)
            encoded_concepts, extra_info = concept_extractor.forward(latent[-3:])
            explainer_classes = explainer.forward(encoded_concepts)
            rebuilt_images: Tensor = viscoin_gan.forward(z1=encoded_concepts, z2=extra_info)  # type: ignore
        preds = classes.argmax(dim=1, keepdim=True)
        preds_expl = explainer_classes.argmax(dim=1, keepdim=True)

        # Compute the different losses
        acc_loss.append(F.cross_entropy(classes, labels).item())
        cr_loss.append(concept_regularization_loss(encoded_concepts).item())
        of_loss.append(output_fidelity_loss(classes, explainer_classes).item())
        lp_loss.append(lpips_loss(rebuilt_images, images).item())
        rec_loss_l1.append(F.l1_loss(rebuilt_images, images).item())
        rec_loss_l2.append(F.mse_loss(rebuilt_images, images).item())
        preds_overlap.append(torch.sum(preds == preds_expl).item())
        correct_preds.append(torch.sum(preds == labels).item())
        correct_expl_preds.append(torch.sum(preds_expl == labels).item())

        if compute_fid:
            fake_features = feature_detector(rebuilt_images, **detector_kwargs)
            real_features = feature_detector(images, **detector_kwargs)
            stats_fake.append_torch(fake_features)
            stats_real.append_torch(real_features)

    # Aggregate the different losses
    results = TestingResults(
        acc_loss=np.mean(acc_loss),
        cr_loss=np.mean(cr_loss),
        of_loss=np.mean(of_loss),
        lp_loss=np.mean(lp_loss),
        rec_loss_l1=np.mean(rec_loss_l1),
        rec_loss_l2=np.mean(rec_loss_l2),
        preds_overlap=100 * np.mean(preds_overlap),
        correct_preds=100 * np.mean(correct_preds),
        correct_expl_preds=100 * np.mean(correct_expl_preds),
    )

    if compute_fid:
        mu_real, sigma_real = stats_real.get_mean_cov()
        mu_fake, sigma_fake = stats_fake.get_mean_cov()
        m = np.square(mu_fake - mu_real).sum()
        s, _ = sqrtm(np.dot(sigma_fake, sigma_real), disp=False)
        fid = np.real(m + np.trace(sigma_fake + sigma_real - s * 2))
        results.fid_score = fid

    return results


@dataclass
class AmplifiedConceptsResults:
    """Amplified concepts results.

    Args:
        image: The image sample tensor that was amplified.
        default_probas: (n_concepts,) The default class probabilities for all the concepts (computed on an image generated with zeroed inputs with the adaped gan)
        multipliers: The multipliers used to amplify the concepts
        best_concept_probas_best: Given the initial best concept for the image, contains the probabilities of said concept for all multipliers after targeted best concept amplification.
        best_concept_probas_rand: Given the initial best concept for the image, contains the probabilities of said concept for all multipliers after random concept amplification.
        amplified_images: (n_multipliers, 3, 256, 256) The amplified images for all multipliers
    """

    image: Tensor
    default_probas: Tensor
    multipliers: list[float]
    best_concept_probas_best: list[Number]
    best_concept_probas_rand: list[Number]
    amplified_images: list[Tensor]


class ThresholdSelection(TypedDict):
    """Select all concepts above a threshold"""

    method: Literal["threshold"]
    threshold: float  # Default: 0.2


class TopKSelection(TypedDict):
    """Select the top k concepts (use k=1 to amplify only the best concept)"""

    method: Literal["top_k"]
    k: int  # Default: 3


def amplify_concepts(
    image: Tensor,
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    explainer: Explainer,
    generator: GeneratorAdapted,
    concept_selection: ThresholdSelection | TopKSelection,
    device: str,
) -> AmplifiedConceptsResults:
    """
    Amplify the best concepts of a given image sample, and compare the regenerated images.
    The best concepts are chosen based on a threshold between [-1, 1] on normalised max concept activations.

    Args:
        image: The image sample tensor to amplify.
        classifier: The classifier model.
        concept_extractor: The concept extractor model.
        explainer: The explainer model.
        generator: The VisCoIN GAN model.
        device: The device to use.
        threshold: [-1, 1] - The threshold to select the best concepts.

    Returns: AmplifiedConceptsResults, a dataclass containing the results.
    """
    # Put the models in evaluation mode
    classifier = classifier.eval()
    concept_extractor = concept_extractor.eval()
    explainer = explainer.eval()
    generator = generator.eval()

    multipliers = [0.0, 1.0, 2.0, 4.0]
    results = AmplifiedConceptsResults(
        image=image.clone(),
        default_probas=torch.tensor([]),
        multipliers=multipliers,
        best_concept_probas_best=[],
        best_concept_probas_rand=[],
        amplified_images=[],
    )

    # Unsqueeze a batch dimension if needed
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    synth_kwargs = {"noise_mode": "const"}
    expl_weights = explainer.linear.weight  # (n_classes, n_concepts)

    # Forward pass through the models
    with torch.no_grad():
        _, classifier_latents = classifier.forward(image)
        concept_embeddings, extra_info = concept_extractor.forward(classifier_latents[-3:])
        expl_logits = explainer.forward(concept_embeddings)  # output shape: (1, n_classes)
    expl_class_probas = F.softmax(expl_logits, dim=1)  # (1, n_classes)
    class_pred = expl_class_probas.argmax(dim=1)  # (1) - predicted best class for this image
    concept_pred = expl_logits.argmax(dim=1)  # (1) - predicted best concept for this image

    # Compute the concept intensities (for each concept, max of the 3x3 embedding map, and then normalize by the max)
    # (n_concepts)
    concept_pooling = F.adaptive_max_pool2d(concept_embeddings.squeeze(0), 1).squeeze()
    # (n_concepts)
    concept_intensities = concept_pooling * expl_weights[class_pred].squeeze()  # type: ignore
    concept_intensities /= concept_intensities.abs().max()  # (n_concepts) in [-1, 1]

    # Get the concept index whose intensity is above the threshold
    # (n_best_concepts) in [0, n_concepts[
    if concept_selection["method"] == "threshold":
        best_concepts = torch.where(concept_intensities > concept_selection["threshold"])[0]
    elif concept_selection["method"] == "top_k":
        best_concepts = torch.topk(concept_intensities, k=concept_selection["k"]).indices

    # Choose as many random concepts (will serve as a baseline)
    rand_concepts = rd.choice(len(concept_intensities), len(best_concepts), replace=False)

    # Concept intensity multipliers
    for multiplier in multipliers:
        # concept embeddings (1, n_concepts, 3, 3)
        embeddings_best = concept_embeddings.clone()
        embeddings_rand = concept_embeddings.clone()

        # Multiply the selected concept embeddings by the multiplier
        for attr in best_concepts:
            embeddings_best[0, attr] *= multiplier
        for attr in rand_concepts:
            embeddings_rand[0, attr] *= multiplier

        # Generate the images with the modified embeddings
        with torch.no_grad():
            new_image_best = generator(z1=embeddings_best, z2=extra_info, **synth_kwargs)
            new_image_rand = generator(z1=embeddings_rand, z2=extra_info, **synth_kwargs)
        results.amplified_images.append(new_image_best)

        # Do a full forward pass for the generated images
        with torch.no_grad():
            _, latent_best = classifier.forward(new_image_best)
            _, latent_rand = classifier.forward(new_image_rand)
            concepts_best, _ = concept_extractor.forward(latent_best[-3:])
            concepts_rand, _ = concept_extractor.forward(latent_rand[-3:])
            classes_best = explainer.forward(concepts_best)
            classes_rand = explainer.forward(concepts_rand)
        probas_best = F.softmax(classes_best, dim=1)
        probas_rand = F.softmax(classes_rand, dim=1)

        # Accumulate the statistics for all multipliers
        results.best_concept_probas_best.append(probas_best[0, concept_pred].item())
        results.best_concept_probas_rand.append(probas_rand[0, concept_pred].item())

    # Generate and analyze the "default" image (all embeddings to 0)
    default_z1 = torch.zeros_like(concept_embeddings)
    default_z2 = torch.zeros_like(extra_info)
    default_img = generator(z1=default_z1, z2=default_z2, **synth_kwargs)
    _, default_latents = classifier.forward(default_img)
    default_concepts, _ = concept_extractor.forward(default_latents[-3:])
    default_classes = explainer.forward(default_concepts)
    results.default_probas = F.softmax(default_classes, dim=1).squeeze(0)

    return results


def amplify_single_concepts(
    image: Tensor,
    generator: GeneratorAdapted,
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    concept_index: int,
    multipliers: list[float],
) -> list[Tensor]:

    _, classifier_latents = classifier.forward(image.unsqueeze(0))
    concept_embeddings, extra_info = concept_extractor.forward(classifier_latents[-3:])

    amplified_images = []
    for multiplier in multipliers:

        embeddings = concept_embeddings.clone()
        embeddings[0, concept_index] *= multiplier

        with torch.no_grad():
            new_image = generator(z1=embeddings, z2=extra_info, noise_mode="const")

        amplified_images.append(new_image)

    return amplified_images


def plot_amplified_images(original: Tensor, images: list[Tensor], multipliers: list[float]):
    """Plot amplified images in a row, with their corresponding multiplier in the title"""

    np_images = [from_torch(image) for image in images]
    np_original = from_torch(original)

    fig, axs = plt.subplots(1, len(multipliers) + 1, figsize=(15, 5))
    fig.suptitle("Amplification of best concepts for an image")

    axs[0].imshow(np_original)
    axs[0].set_title("Original")
    axs[0].axis("off")

    for i, (image, multiplier) in enumerate(zip(np_images, multipliers)):
        axs[i + 1].imshow(image)
        axs[i + 1].set_title(f"Multiplier: {multiplier:.2f}")
        axs[i + 1].axis("off")

    plt.show()


def plot_amplified_images_batch(
    originals: list[Tensor],
    images: list[list[Tensor]],
    multipliers: list[float],
    labels: list[str] | None = None,
):
    """Plot amplified images in a row, with their corresponding multiplier in the title.
    This function can plot multiple rows."""

    np_images = [[from_torch(image) for image in images[i]] for i in range(len(images))]
    np_original = [from_torch(original) for original in originals]

    fig, axs = plt.subplots(len(images), len(multipliers) + 1, figsize=(15, 5 * len(images)))
    fig.suptitle("Amplification of best concepts for an image")

    for i, (original, row_images) in enumerate(zip(np_original, np_images)):
        axs[i, 0].imshow(original)
        axs[i, 0].set_title("Original")
        axs[i, 0].axis("off")

        if labels is not None:
            axs[i, 0].set_title(f"Predicted Label : {labels[i]}\nOriginal")

        for j, (image, multiplier) in enumerate(zip(row_images, multipliers)):
            axs[i, j + 1].imshow(image)
            if i == 0:  # Only display the header multiplier for the first row
                axs[i, j + 1].set_title(f"Multiplier: {multiplier:.2f}")
            axs[i, j + 1].axis("off")

    plt.show()
