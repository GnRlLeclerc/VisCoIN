"""
Utilities to download, unzip datasets.
Because some of our datasets are on Kaggle, for ease of use when changing the destination
of datasets in limited disk space machines, we download datasets that are not on Kaggle
into the Kaggle cache path with username "viscoin".
"""

import os
import zipfile
from typing import Literal

import kagglehub
import requests
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    Compose as ComposeV1,  # for the CLIP model that uses legacy compose
)
from torchvision.transforms.v2 import Compose as ComposeV2
from tqdm import tqdm

from viscoin.datasets.transforms import RESNET_TEST_TRANSFORM, RESNET_TRAIN_TRANSFORM

Compose = ComposeV1 | ComposeV2

DatasetType = Literal["cub", "funnybirds"]


DATASET_CLASSES = {
    "cub": 200,
    "funnybirds": 50,
}

DEFAULT_CHECKPOINTS = {
    dataset: {
        "classifier": f"checkpoints/{dataset}/classifier-{dataset}.pkl",
        "gan": f"checkpoints/{dataset}/gan-{dataset}.pkl",
        "gan_adapted": f"checkpoints/{dataset}/gan-adapted-{dataset}.pkl",
        "viscoin": f"checkpoints/{dataset}/viscoin-{dataset}.pkl",
    }
    for dataset in DATASET_CLASSES.keys()
}


def download(url: str):
    """Download a ZIP file from a URL and extract it to the specified destination."""

    output_filename = "temp.zip"

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192
    t = tqdm(total=total_size, unit="i", unit_scale=True, desc=output_filename)

    with open(output_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=block_size):
            f.write(chunk)
            t.update(len(chunk))
    t.close()

    # Unzip the downloaded file
    print("Unzipping dataset...")
    with zipfile.ZipFile(output_filename, "r") as zip_ref:
        zip_ref.extractall(dataset_path(""))

    # Remove the zip file after extraction
    os.remove(output_filename)

    print("Done.")


def dataset_exists(name: str) -> bool:
    """Check if a dataset has already been downloaded."""

    path = dataset_path(name)

    return os.path.exists(path) and os.path.isdir(path)


def dataset_path(name: str) -> str:
    """Get the full path to a custom downloaded dataset, inside the Kagglehub cache path."""

    kaggle_cache = kagglehub.config.DEFAULT_CACHE_FOLDER  # type: ignore

    return os.path.join(kaggle_cache, "datasets", "viscoin", name)


def _get_transform(
    default: Compose, train: Compose, test: Compose, mode: Literal["train", "test"] | Compose | None
) -> Compose:
    """Auxiliary function to load the correct image transformation pipeline"""

    match mode:
        case None:
            return default
        case "train":
            return train
        case "test":
            return test
        case _:
            return mode


def get_datasets(
    name: DatasetType, transform: Literal["train", "test"] | Compose | None = None
) -> tuple[Dataset, Dataset]:
    """Load the train and test datasets for the given dataset name.

    Reminder:
    - train transforms usually apply random cropping, resizing, etc
    - test transforms usually apply resizing and normalization

    When doing fine-tuning of additional models (such as concept2clip), use "test" for both datasets
    in order to match the CLIP model's test transforms.

    Args:
        name: dataset variant
        transform: the transform to apply to all images
            - None: applies the default train/test transform to the train/test datasets
            - "train": applies the train transform to both datasets
            - "test": applies the test transform to both datasets
            - Compose: applies the given transform to both datasets
    """

    train_tf = _get_transform(
        RESNET_TRAIN_TRANSFORM, RESNET_TRAIN_TRANSFORM, RESNET_TEST_TRANSFORM, transform
    )
    test_tf = _get_transform(
        RESNET_TEST_TRANSFORM, RESNET_TRAIN_TRANSFORM, RESNET_TEST_TRANSFORM, transform
    )

    match name:
        case "cub":
            from viscoin.datasets.cub import CUB_200_2011

            train = CUB_200_2011("train", transform=train_tf)
            test = CUB_200_2011("test", transform=test_tf)

        case "funnybirds":
            from viscoin.datasets.funnybirds import FunnyBirds

            train = FunnyBirds("train", transform=train_tf)
            test = FunnyBirds("test", transform=test_tf)

        case _:
            raise ValueError(f"Unknown dataset: {name}")

    return train, test


def get_dataloaders(
    name: DatasetType, batch_size: int, transform: Literal["train", "test"] | Compose | None = None
) -> tuple[DataLoader, DataLoader]:
    """Load the train and test datasets for the given dataset name into dataloaders.

    Both dataloaders will be shuffled (just in case of batch-dependent testing).

    Reminder:
    - train transforms usually apply random cropping, resizing, etc
    - test transforms usually apply resizing and normalization

    When doing fine-tuning of additional models (such as concept2clip), use "test" for both datasets
    in order to match the CLIP model's test transforms.

    Args:
        name: dataset variant
        batch_size: batch size for the dataloaders
        transform: the transform to apply to all images
            - None: applies the default train/test transform to the train/test datasets
            - "train": applies the train transform to both datasets
            - "test": applies the test transform to both datasets
            - Compose: applies the given transform to both datasets
    """
    train, test = get_datasets(name, transform)

    return DataLoader(train, batch_size, shuffle=True), DataLoader(test, batch_size, shuffle=True)
