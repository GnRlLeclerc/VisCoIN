import torch
import clip
import random
import numpy as np
from torch.utils.data import DataLoader
import click
import sys
import os
from typing import List, Optional
from argparse import Namespace

# Adds Viscoin to the sys path to load the dataset
sys.path.append("./../../")
from viscoin.datasets.cub import Labeled_CUB_200_2011

from trainer import run_lora


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@click.command()
@click.option("--seed", default=1, type=int, help="Random seed for reproducibility.")
@click.option("--root_path", default="", type=str, help="Root path for datasets.")
@click.option("--shots", default=16, type=int, help="Number of shots for training.")
@click.option("--backbone", default="ViT-B/16", type=str, help="Model backbone for CLIP.")
@click.option("--lr", default=2e-4, type=float, help="Learning rate.")
@click.option("--n_iters", default=500, type=int, help="Number of training iterations.")
@click.option("--batch_size", default=32, type=int, help="Batch size.")
@click.option("--device", default="cuda", type=str, help="Device to run the experiment.")
@click.option(
    "--position",
    default="all",
    type=click.Choice(["bottom", "mid", "up", "half-up", "half-bottom", "all", "top3"]),
    help="Where to place the LoRA modules.",
)
@click.option(
    "--encoder",
    default="both",
    type=click.Choice(["text", "vision", "both"]),
    help="Encoder type for LoRA.",
)
@click.option(
    "--params",
    multiple=True,
    default=["q", "k", "v"],
    help="Attention matrices to apply LoRA (e.g., q, k, v).",
)
@click.option("--r", default=2, type=int, help="Rank of the low-rank matrices.")
@click.option("--alpha", default=1, type=int, help="Scaling factor (see LoRA paper).")
@click.option("--dropout_rate", default=0.25, type=float, help="Dropout rate before LoRA module.")
@click.option("--save_path", default="./", type=str, help="Path to save trained LoRA modules.")
@click.option(
    "--filename",
    default="lora_weights",
    type=str,
    help="Filename to save the LoRA weights (extension .pt will be added).",
)
@click.option(
    "--eval_only",
    is_flag=True,
    default=False,
    help="Flag to evaluate LoRA modules without training.",
)
def main(
    seed: int,
    root_path: str,
    shots: int,
    backbone: str,
    lr: float,
    n_iters: int,
    batch_size: int,
    device: str,
    position: str,
    encoder: str,
    params: List[str],
    r: int,
    alpha: int,
    dropout_rate: float,
    save_path: Optional[str],
    filename: str,
    eval_only: bool,
) -> None:
    """
    Main function to run the LoRA experiment with provided configurations.
    """
    # Set random seed
    set_random_seed(seed)

    # Load CLIP model
    clip_model, preprocess = clip.load(backbone)
    clip_model.eval()
    logit_scale = 10

    # Prepare dataset
    click.echo("Preparing dataset.")
    dataset_path = os.path.join(os.path.dirname(__file__), "./../../datasets/CUB_200_2011/")

    train_dataset = Labeled_CUB_200_2011(dataset_path, mode="train")
    test_dataset = Labeled_CUB_200_2011(dataset_path, mode="test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Run LoRA
    run_lora(
        args=Namespace(
            **{
                "seed": seed,
                "root_path": root_path,
                "shots": shots,
                "logit_scale": logit_scale,
                "backbone": backbone,
                "lr": lr,
                "n_iters": n_iters,
                "batch_size": batch_size,
                "position": position,
                "encoder": encoder,
                "params": params,
                "r": r,
                "alpha": alpha,
                "dropout_rate": dropout_rate,
                "save_path": save_path,
                "filename": filename,
                "eval_only": eval_only,
                "device": device,
            }
        ),
        clip_model=clip_model,
        dataset=train_dataset,
        train_loader=train_loader,
        test_loader=test_loader,
    )


if __name__ == "__main__":
    main()
