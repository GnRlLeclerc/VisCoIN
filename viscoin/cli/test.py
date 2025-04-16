import click
import torch

from viscoin.cli.utils import batch_size, checkpoints, dataset, device
from viscoin.datasets.utils import DATASET_CLASSES, DatasetType, get_dataloaders
from viscoin.models.classifiers import Classifier
from viscoin.testing.classifiers import test_classifier


@click.command()
@click.argument("model_name")
@batch_size
@device
@dataset
@checkpoints
def test(
    model_name: str,
    batch_size: int,
    device: str,
    dataset: DatasetType,
    checkpoints: str | None,
):
    """Test a model on a dataset"""

    _, dataloader = get_dataloaders(dataset, batch_size=batch_size)
    pretrained = checkpoints is not None

    match model_name:
        case "classifier":
            model = Classifier(output_classes=DATASET_CLASSES[dataset], pretrained=pretrained).to(
                device
            )
        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    if pretrained:
        model.load_state_dict(torch.load(checkpoints, weights_only=True))

    model = model.to(device)

    accuracy, loss = test_classifier(model, dataloader, device)

    click.echo(f"Accuracy: {100*accuracy:.2f}%")
    click.echo(f"Loss: {loss}")
