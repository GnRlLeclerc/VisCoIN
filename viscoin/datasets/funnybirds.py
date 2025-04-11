import io
import json
import os
import zipfile

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose as ComposeV1,  # for the CLIP model that uses legacy compose
)
from torchvision.transforms.v2 import Compose as ComposeV2

from viscoin.datasets.transforms import RESNET_TEST_TRANSFORM, RESNET_TRAIN_TRANSFORM
from viscoin.datasets.utils import dataset_exists, dataset_path, download
from viscoin.utils.types import Mode

Compose = ComposeV1 | ComposeV2

DATASET_NAME = "FunnyBirds"


class FunnyBirds(Dataset):
    """FunnyBirds dataset loader."""

    def __init__(
        self,
        mode: Mode = "train",
        transform: Compose | None = None,
    ) -> None:
        """Instantiate a FunnyBirds dataset. Its result is saved in a pickle file for faster reloading.

        Args:
            dataset_path: Path to the downloaded dataset. Defaults to "datasets/FunnyBirds".
            mode: Whether to consider training or testing data. Defaults to "train".
            bbox_only: Whether to crop the images to include only the bounding box of the bird.
            transform: Additional optional transformations to perform on loaded images. Will default to the appropriate one given the mode.
        """

        if not dataset_exists(DATASET_NAME):
            download(
                "https://download.visinf.tu-darmstadt.de/data/funnybirds/FunnyBirds.zip",
            )

        self.dataset_path = dataset_path(DATASET_NAME)
        self.mode: Mode = mode

        # Load appropriate transformations if none are provided
        if transform is None:
            if self.mode == "train":
                transform = RESNET_TRAIN_TRANSFORM
            else:
                transform = RESNET_TEST_TRANSFORM
        self.transform = transform

        # Number of classes defined in the dataset
        self.n_classes = 50

        self.image_paths = []
        self.labels = []

        # Select the appropriate data folder
        data_folder = os.path.join(self.dataset_path, mode)

        # Loop through the subfolders
        for folder in os.listdir(data_folder):
            for image in os.listdir(os.path.join(data_folder, folder)):
                self.image_paths.append(os.path.join(folder, image))
                self.labels.append(int(folder))

        # Image cache
        # The whole dataset is not loaded instantly in memory, but only when needed.
        self.image_cache: dict[int, Tensor] = {}

    def load_image(self, index: int) -> Tensor:
        """Load an image by index, and apply the specified transformations.
        Note that tensors have reversed dimensions (C, H, W) instead of (H, W, C)."""

        # Load the image
        image = Image.open(os.path.join(self.dataset_path, self.mode, self.image_paths[index]))

        image = image.convert("RGB")

        # Apply the transformations
        tensor_image = self.transform(image)

        # NOTE: actually returns a torchvision.tv_tensors._image.Image
        return tensor_image  # type: ignore

    def __len__(self):
        """Returns the length of the dataset (depends on the test/train mode)."""

        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Get an image and its label by index.

        Returns:
            image: The image tensor.
            label: The label tensor.
        """

        # Load the image
        if index in self.image_cache:
            image = self.image_cache[index]
        else:
            image = self.load_image(index)
            self.image_cache[index] = image

        label = torch.as_tensor(self.labels[index])

        return image, label


if __name__ == "__main__":
    """
    Saves all the images in the dataset as a zip file in order to train a StyleGAN2-ADA model.
    """

    train_dataset = FunnyBirds(mode="train")
    test_dataset = FunnyBirds(mode="test")
    print("Dataset size:", len(train_dataset))

    label_file_path = "dataset.json"
    labels_data = {"labels": []}

    # Save as a zip file for stylegan2-ada-pytorch
    with zipfile.ZipFile("funnybirds_dataset.zip", "w") as zipf:
        for dataset in [train_dataset, test_dataset]:
            for idx in range(len(dataset)):
                path = os.path.join(dataset.dataset_path, dataset.mode, dataset.image_paths[idx])
                file_name = f"{dataset.mode}_{path.split("/")[-1]}"

                zipf.write(path, arcname=file_name)  # Store only the filename in the zip

                labels_data["labels"].append([file_name, dataset.labels[idx]])

        # Save the labels as a json file
        json_bytes = json.dumps(labels_data, indent=4).encode("utf-8")  # Convert JSON to bytes
        with io.BytesIO(json_bytes) as json_file:
            zipf.writestr(label_file_path, json_file.getvalue())  # Write JSON file to ZIP

    print("Dataset saved as funnybirds_dataset.zip")
