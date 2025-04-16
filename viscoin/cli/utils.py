"""
Command line helpers for the viscoin cli.
Notably, flag wrappers.

NOTE: training parameter flags are optional, so that different models may specify
different defaults when they are not present.
"""

import click

from viscoin.datasets.utils import DEFAULT_VISCOIN

###########################################################
#         OPTIONAL TRAINING / TESTING PARAMETERS          #
###########################################################


def batch_size(func):
    return click.option(
        "--batch-size",
        required=False,
        help="The batch size to use for training/testing",
        type=int,
    )(func)


def epochs(func):
    return click.option(
        "--epochs",
        help="The amount of epochs to train the model for",
        required=False,
        type=int,
    )(func)


def learning_rate(func):
    return click.option(
        "--learning-rate",
        help="The optimizer learning rate",
        required=False,
        type=float,
    )(func)


###########################################################
#           REQUIRED / DEFAULT MODEL PARAMETERS           #
###########################################################


def device(func):
    return click.option(
        "--device",
        default="cuda",
        help="The device to use for training/testing",
        type=str,
    )(func)


def checkpoints(func):
    return click.option("--checkpoints", help="The path to load the checkpoints", type=str)(func)


def output_weights(func):
    return click.option(
        "--output-weights",
        help="The path/filename where to save the weights",
        type=str,
        default="output-weights.pt",
    )(func)


###########################################################
#                       MODEL PATHS                       #
###########################################################


def viscoin_pickle_path(func):
    return click.option(
        "--viscoin-pickle-path",
        help="The path to the viscoin pickle file",
        default=DEFAULT_VISCOIN,
        required=False,
        type=str,
    )(func)


def concept2clip_pickle_path(func):
    return click.option(
        "--clip-adapter-path",
        help="The path to the concept2clip pickle file",
        required=False,
        type=str,
    )(func)


def dataset(func):
    return click.option(
        "--dataset",
        help="The dataset to use",
        default="cub",
        type=click.Choice(["cub", "funnybirds"]),
    )(func)
