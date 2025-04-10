"""CLIP variant used in the VisCoIN project."""

import os
from typing import Literal

import clip
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from viscoin.datasets.cub import CUB_200_2011
from viscoin.datasets.funnybirds import FunnyBirds


def _cache(mode: Literal["train", "test"], dataset: str, model: str) -> str:
    model = model.replace("/", "-")
    return f"checkpoints/clip/{model}_{dataset}_{mode}.pt"


class CLIP(nn.Module):
    """CLIP model wrapper to make it explicit which version of CLIP we are using, and the embedding size."""

    def __init__(self, device: str = "cuda"):
        super().__init__()

        self.kind = "ViT-B/32"
        model, preprocess = clip.load(self.kind, device=device)
        self.model = model
        self.preprocess = preprocess
        self.embedding_size = model.visual.output_dim
        self.device = device

    def compute_embeddings(self, dataset: Literal["cub", "funnybirds"]) -> tuple[Tensor, Tensor]:
        """Compute CLIP embeddings on the given datasets.
        We define this helper because CLIP needs its own transforms,
        that may not align with the transforms VisCoIN is using for the given dataset.

        The results are cached under checkpoints/clip/

        Returns:
            train_embeddings: CLIP embeddings for the training set (on CPU)
            test_embeddings: CLIP embeddings for the testing set (on CPU)
        """

        try:
            train_embeddings = torch.load(_cache("train", dataset, self.kind), weights_only=True)
            test_embeddings = torch.load(_cache("test", dataset, self.kind), weights_only=True)
            return train_embeddings, test_embeddings
        except FileNotFoundError:
            pass

        match dataset:
            case "cub":
                train = CUB_200_2011("train", transform=self.preprocess)
                test = CUB_200_2011("test", transform=self.preprocess)
            case "funnybirds":
                train = FunnyBirds("train", transform=self.preprocess)
                test = FunnyBirds("test", transform=self.preprocess)
            case _:
                raise ValueError(f"Unknown dataset: {dataset}")

        self.eval()

        batch_size = 32
        train_loader = DataLoader(train, batch_size)
        test_loader = DataLoader(test, batch_size)

        train_embeddings = torch.zeros((len(train_loader.dataset), self.embedding_size))  # type: ignore
        test_embeddings = torch.zeros((len(test_loader.dataset), self.embedding_size))  # type: ignore

        for i, (batch, _) in enumerate(
            tqdm(train_loader, desc=f"Computing CLIP embeddings for {dataset} - train")
        ):
            batch = batch.to(self.device)
            with torch.no_grad():
                train_embeddings[i * batch_size : (i + 1) * batch_size] = (
                    self.model.encode_image(batch).detach().cpu()
                )

        for i, (batch, _) in enumerate(
            tqdm(test_loader, desc=f"Computing CLIP embeddings for {dataset} - test")
        ):
            batch = batch.to(self.device)
            with torch.no_grad():
                test_embeddings[i * batch_size : (i + 1) * batch_size] = (
                    self.model.encode_image(batch).detach().cpu()
                )

        # Save the embeddings to cache
        os.makedirs("checkpoints/clip", exist_ok=True)
        torch.save(train_embeddings, _cache("train", dataset, self.kind))
        torch.save(test_embeddings, _cache("test", dataset, self.kind))

        return train_embeddings, test_embeddings
