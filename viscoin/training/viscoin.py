"""
Training function for the whole viscoin model.
"""

import itertools
from dataclasses import dataclass

import numpy as np
import numpy.random as rd
import torch
import torch.nn.functional as F
from torch import optim
from torch.types import Number
from torch.utils.data import DataLoader
from tqdm import tqdm

from viscoin.models.gan import GeneratorAdapted, fix_path
from viscoin.utils.dataclasses import IgnoreNone
from viscoin.utils.types import TrainingResults

fix_path()

from stylegan2_ada.training.networks import Generator
from viscoin.models.classifiers import Classifier
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer
from viscoin.models.utils import save_viscoin
from viscoin.testing.viscoin import amplify_concepts, test_viscoin
from viscoin.training.losses import (
    concept_orthogonality_loss,
    concept_regularization_loss,
    cross_cross_entropy_loss,
    gan_regularization_loss,
    output_fidelity_loss,
    reconstruction_loss,
)
from viscoin.training.utils import (
    Accumulator,
    loop_iter,
    requires_grad,
    synthetic_samples,
    update_lr,
)
from viscoin.utils.logging import get_logger


@dataclass
class VisCoINTrainingParams(IgnoreNone):
    """Training parameters for the VisCoIN ensemble.
    The default are for CUB (see page 25 of the paper).

    Args:
        iterations: The number of iterations to train the ensemble.
        learning_rate: The learning rate for the optimizers.
        cd_fid_iteration: The iteration at which we start to compute the concept diversity and fidelity loss.
        batch_size: The batch size for the training.
        alpha: The output fidelity loss coefficient.
        beta: The LPIPS loss coefficient.
        gamma: The reconstruction classification loss coefficient.
        delta: The sparsity loss coefficient.
        gradient_accumulation: The number of steps to accumulate gradients for.
        device: The device to use for training.
    """

    # Generic params
    iterations: int = 100_000
    learning_rate: float = 0.0001
    cd_fid_iteration: int = 100
    batch_size: int = 8

    # Loss coefficients
    alpha: float = 0.5  # Output fidelity loss
    beta: float = 3.0  # LPIPS loss
    gamma: float = 0.1  # Reconstruction classification loss
    delta: float = 0.2  # Sparsity loss

    # Gradient accumulation
    gradient_accumulation: int = 1  # Step value of 1: no accumulation

    device = "cuda"


def train_viscoin(
    # Models
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    explainer: Explainer,
    viscoin_gan: GeneratorAdapted,
    generator_gan: Generator,
    # Loaders
    train_loader: DataLoader,
    test_loader: DataLoader,
    # Training parameters
    params: VisCoINTrainingParams,
):
    """Train the VisCoIN ensemble on the CUB or FunnyBirds dataset, using the parameters defined in the paper.

    Args:
        classifier: The classifier model to explain.
        concept_extractor: The concept extractor model.
        explainer: The explainer model.
        viscoin_gan: The adapted GAN model that will try to reconstruct the input.
        generator_gan: The GAN model that will generate synthetic samples for the training loop.
        train_loader: The training dataloader.
        test_loader: The test dataloader.
        test_loader: The test dataloader.
        device: The device to use.
    """
    logger = get_logger()
    device = params.device

    # Model preparations
    classifier.eval()  # Freeze the classifier
    requires_grad(classifier, False)
    requires_grad(viscoin_gan.synthesis, False)  # Freeze the synthesis part of the GAN

    # Move the models to the device
    classifier = classifier.to(device)
    concept_extractor = concept_extractor.to(device)
    explainer = explainer.to(device)
    viscoin_gan = viscoin_gan.to(device)
    generator_gan = generator_gan.to(device)

    # Make dataloaders iterable
    train_loader_iter = loop_iter(train_loader)

    # Define the optimizers
    learning_rate = params.learning_rate
    gan_optimizer = optim.Adam(viscoin_gan.mapping.parameters(), lr=learning_rate)
    optimizer = optim.Adam(
        itertools.chain(concept_extractor.parameters(), explainer.parameters()), lr=learning_rate
    )
    accumulator = Accumulator(params.gradient_accumulation)

    ###########################################################################
    #                               TRAINING LOOP                             #
    ###########################################################################

    for i in tqdm(range(params.iterations), "Training VisCoIN"):

        # Put models in training mode
        concept_extractor.train()
        explainer.train()
        viscoin_gan.train()

        ###################################################
        #            LEARNING RATE SCHEDULING             #
        ###################################################

        # Update the learning rate all 1000 iterations after the first half
        if i > params.iterations // 2 and i % 1000 == 0:
            learning_rate *= 0.8

            update_lr(optimizer, learning_rate)
            update_lr(gan_optimizer, learning_rate)

        ###################################################
        #                  SAMPLE INPUTS                  #
        ###################################################

        # Gather real image samples (1 batch worth) and mix them with GAN generated images
        real_images, labels = next(train_loader_iter)
        real_images, labels = real_images.to(device), labels.to(device)
        fake_images = synthetic_samples(generator_gan, len(real_images), device)
        all_images = torch.cat([real_images, fake_images]).to(device)

        ###################################################
        #                   FORWARD PASS                  #
        ###################################################

        # Forward pass of the VisCoIN ensemble
        classes, hidden_states = classifier.forward(all_images)
        encoded_concepts, extra_info = concept_extractor.forward(hidden_states[-3:])
        explainer_classes = explainer.forward(encoded_concepts)

        ###################################################
        #                 LOSS COMPUTATION                #
        ###################################################

        acc_loss = F.cross_entropy(classes[: len(labels)], labels)

        if i > params.cd_fid_iteration:
            cr_loss = params.delta * concept_regularization_loss(encoded_concepts)
            of_loss = params.alpha * output_fidelity_loss(classes, explainer_classes)
        else:
            cr_loss = torch.tensor(0.0).to(device)
            of_loss = torch.tensor(0.0).to(device)

        ortho_loss = concept_orthogonality_loss(concept_extractor)

        # Reconstruction loss
        rebuilt_images, gan_latents = viscoin_gan.forward(
            z1=encoded_concepts, z2=extra_info, return_latents=True
        )
        rebuilt_classes, _ = classifier.forward(rebuilt_images)

        rec_loss = reconstruction_loss(
            rebuilt_images,
            all_images,
            rebuilt_classes,
            classes,
            params.gamma,
            params.beta,
        )

        gan_loss = gan_regularization_loss(gan_latents, viscoin_gan)

        total_loss = acc_loss + cr_loss + of_loss + ortho_loss + rec_loss + gan_loss

        ###################################################
        #                 BACKPROPAGATION                 #
        ###################################################

        total_loss.backward()

        # Step the optimizers
        if accumulator.step():
            optimizer.step()
            gan_optimizer.step()

            optimizer.zero_grad()
            gan_optimizer.zero_grad()

        ###################################################
        #                      TESTING                    #
        ###################################################

        # Every 2000 iterations, test the models
        if i % 2000 == 0:

            # Compare classifier prediction on the original vs reconstructed images
            inter_loss = cross_cross_entropy_loss(rebuilt_classes, classes)
            results = TrainingResults(
                acc_loss.item(),
                cr_loss.item(),
                of_loss.item(),
                ortho_loss.item(),
                rec_loss.item(),
                gan_loss.item(),
                inter_loss.item(),
            )

            data = {f"train_{key}": value for key, value in results.__dict__.items()}

            test_results = test_viscoin(
                classifier,
                concept_extractor,
                explainer,
                viscoin_gan,
                test_loader,
                device,
                compute_fid=True,
                verbose=False,
            )

            # Merge the training and testing results
            data.update({f"test_{key}": value for key, value in test_results.__dict__.items()})
            logger.info(data)

        # Every 20_000 iterations, save the model checkpoints
        # NOTE: was 50_000 in the original code
        if i % 20_000 == 0:
            save_viscoin(
                classifier,
                concept_extractor,
                explainer,
                viscoin_gan,
                f"viscoin{i//20_000}-{params.iterations//20_000}.pth",
            )

        # Every 25_000 iterations, generate 200 amplified samples (~ faithfullness)
        if i % 25000 == 0 and i > 0:
            best_concept_proba: list[Number] = []

            # Select 200 random images from the test set
            for i in rd.choice(len(test_loader), 200, replace=False):

                results = amplify_concepts(
                    test_loader.dataset[i][0],
                    classifier,
                    concept_extractor,
                    explainer,
                    viscoin_gan,
                    {"method": "threshold", "threshold": 0.2},
                    [0.0, 1.0, 2.0, 4.0],
                    device,
                )
                # The element of index 1 corresponds to the rebuild image for x1 intensity
                # this evaluates faithfullness
                best_concept_proba.append(results.best_concept_probas_best[1])

            print(
                f"Faithfullness stats (probability of best concept after reconstruction): mean = {np.mean(best_concept_proba)} --- std = {np.std(best_concept_proba)}"
            )
