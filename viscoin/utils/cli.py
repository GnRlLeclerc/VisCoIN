"""
Command line helpers for the viscoin cli.
Notably, flag wrappers.
"""

import click


def batch_size(func):
    return click.option(
        "--batch-size",
        default=32,
        help="The batch size to use for training/testing",
        type=int,
    )(func)


def device(func):
    return click.option(
        "--device",
        default="cuda",
        help="The device to use for training/testing",
        type=str,
    )(func)


def dataset_path(func):
    return click.option(
        "--dataset-path",
        help="The path to the dataset to use for training/testing",
        default="./datasets/CUB_200_2011/",
        required=True,
        type=str,
    )(func)


def epochs(func):
    return click.option(
        "--epochs",
        help="The amount of epochs to train the model for",
        default=30,
        type=int,
    )(func)


def output_weights(func):
    return click.option(
        "--output-weights",
        help="The path/filename where to save the weights",
        type=str,
        default="output-weights.pt",
    )(func)


def viscoin_pickle_path(func):
    return click.option(
        "--viscoin-pickle-path",
        help="The path to the viscoin pickle file",
        default="./checkpoints/cub/viscoin-cub.pkl",
        required=True,
        type=str,
    )(func)


def clip_adapter_path(func):
    return click.option(
        "--clip-adapter-path",
        help="The path to the clip adapter model",
        required=True,
        type=str,
    )(func)


def vocab_path(func):
    return click.option(
        "--vocab-path",
        help="The path to the vocab file",
        required=True,
        type=str,
    )(func)
