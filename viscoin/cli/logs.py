"""Inspect jsonl training logs of viscoin."""

import json

import click

from viscoin.utils.types import TestingResults, TrainingResults


@click.command()
@click.option(
    "--logs-path",
    help="The path to the logs file",
    required=True,
    type=str,
)
def logs(logs_path: str):
    """Parse a viscoin training jsonl log file and plot the losses and metrics"""

    training_results: list[TrainingResults] = []
    testing_results: list[TestingResults] = []

    # Read the log file
    with open(logs_path, "r") as f:
        for line in f:
            data = json.loads(line)

            train_kwargs = {}
            test_kwargs = {}

            for key, value in data.items():
                if key.startswith("train_"):
                    train_kwargs[key[6:]] = value
                elif key.startswith("test_"):
                    test_kwargs[key[6:]] = value
                else:
                    raise ValueError(f"Unknown key: {key}")

            training_results.append(
                TrainingResults(
                    **train_kwargs,
                )
            )
            testing_results.append(
                TestingResults(
                    **test_kwargs,
                )
            )

    # Plot the losses
    TrainingResults.plot_losses(training_results)
    TestingResults.plot_losses(testing_results)

    # Plot the metrics
    TestingResults.plot_preds_overlap(testing_results)
