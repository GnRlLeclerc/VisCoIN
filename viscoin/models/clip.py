"""CLIP variant used in the VisCoIN project."""

import os
from typing import Literal

import clip
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from viscoin.datasets.utils import get_dataloaders


def _img_cache(mode: Literal["train", "test"], dataset: str, model: str) -> str:
    """Returns the path to a cached file for CLIP image embeddings of a specific dataset"""
    model = model.replace("/", "-")
    return f"checkpoints/clip/{model}_{dataset}_img_{mode}.pt"


def _txt_cache(key: str, dataset: str, model: str) -> str:
    """Returns the path to a cached file for CLIP text embeddings of a set of labels / captions"""
    model = model.replace("/", "-")
    return f"checkpoints/clip/{model}_{dataset}_txt_{key}.pt"


class CLIP(nn.Module):
    """CLIP model wrapper to make it explicit which version of CLIP we are using, and the embedding size.

    NOTE: CLIP embeddings for a same image or text are not deterministic and depend on the batch size.
    The difference is usually of the order of 1e-3.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()

        self.kind = "ViT-B/32"
        model, preprocess = clip.load(self.kind, device=device)
        self.model = model
        self.preprocess = preprocess
        self.embedding_size = model.visual.output_dim
        self.device = device

    def encode_image(self, x: Tensor) -> Tensor:
        return self.model.encode_image(x)

    def encode_text(self, x: Tensor) -> Tensor:
        return self.model.encode_text(x)

    def compute_image_embeddings(
        self, dataset: Literal["cub", "funnybirds"]
    ) -> tuple[Tensor, Tensor]:
        """Compute CLIP embeddings on the given datasets.
        We define this helper because CLIP needs its own transforms,
        that may not align with the transforms VisCoIN is using for the given dataset.

        The results are cached under checkpoints/clip/

        Returns:
            train_embeddings: CLIP embeddings for the training set (on CPU)
            test_embeddings: CLIP embeddings for the testing set (on CPU)
        """

        try:
            train_embeddings = torch.load(
                _img_cache("train", dataset, self.kind), weights_only=True
            )
            test_embeddings = torch.load(_img_cache("test", dataset, self.kind), weights_only=True)
            return train_embeddings, test_embeddings
        except FileNotFoundError:
            pass

        self.eval()

        batch_size = 32
        train, test = get_dataloaders(dataset, batch_size, self.preprocess, shuffle=False)

        train_embeddings = torch.zeros((len(train.dataset), self.embedding_size))  # type: ignore
        test_embeddings = torch.zeros((len(test.dataset), self.embedding_size))  # type: ignore

        for i, (batch, _) in enumerate(
            tqdm(train, desc=f"Computing CLIP embeddings for {dataset} - train")
        ):
            batch = batch.to(self.device)
            with torch.no_grad():
                train_embeddings[i * batch_size : (i + 1) * batch_size] = (
                    self.encode_image(batch).detach().cpu()
                )

        for i, (batch, _) in enumerate(
            tqdm(test, desc=f"Computing CLIP embeddings for {dataset} - test")
        ):
            batch = batch.to(self.device)
            with torch.no_grad():
                test_embeddings[i * batch_size : (i + 1) * batch_size] = (
                    self.encode_image(batch).detach().cpu()
                )

        # Save the embeddings to cache
        os.makedirs("checkpoints/clip", exist_ok=True)
        torch.save(train_embeddings, _img_cache("train", dataset, self.kind))
        torch.save(test_embeddings, _img_cache("test", dataset, self.kind))

        return train_embeddings, test_embeddings

    def compute_text_embeddings(
        self, captions: list[str], dataset: Literal["cub", "funnybirds"], cache_key: str
    ) -> Tensor:
        """Compute CLIP embeddings of the given captions / sentences.

        The captions are inserted into the template "a photo of a <text>" before encoding.

        The results are cached under checkpoints/clip/

        Args:
            captions: list of captions to encode
            dataset: dataset which those captions are for
            cache_key: key to use for caching the results

        Returns:
            text_embeddings: CLIP embeddings for the captions (on CPU)
        """

        try:
            return torch.load(_txt_cache(cache_key, dataset, self.kind), weights_only=True)
        except FileNotFoundError:
            pass

        batch_size = 32

        text = clip.tokenize(captions)
        token_dataset = TensorDataset(text)
        dataloader = DataLoader(token_dataset, batch_size)

        embeddings = torch.zeros((len(captions), self.embedding_size))

        self.model.eval()

        for i, batch in enumerate(
            tqdm(
                dataloader,
                desc=f"Computing caption CLIP embeddings for {dataset} - {cache_key}",
            )
        ):
            with torch.no_grad():
                batch = batch[0].to(self.device)
                embeddings[i * batch_size : (i + 1) * batch_size] = (
                    self.encode_text(batch).detach().cpu()
                )

        # Save the embeddings to cache
        os.makedirs("checkpoints/clip", exist_ok=True)
        torch.save(embeddings, _txt_cache(cache_key, dataset, self.kind))

        return embeddings
