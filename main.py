"""
CLI entrypoint for interacting with the viscoin library.

Example usage:
```bash
python main.py test resnet50 --batch-size 32 --dataset-path datasets/CUB_200_2011/
```

Will be subject to many changes as the project evolves.
"""

import click

from viscoin.cli.amplify import amplify
from viscoin.cli.concept_heatmaps import concept_heatmaps
from viscoin.cli.concepts import concepts
from viscoin.cli.logs import logs
from viscoin.cli.test import test
from viscoin.cli.to_pickle import to_pickle
from viscoin.cli.train import train

# Imports Trogon if installed : Terminal User Interface for Click commands
try:
    from trogon import tui
except ImportError:

    def tui():
        return lambda f: f


@tui()
@click.group(context_settings={"max_content_width": 1000})
def main():
    pass


main.add_command(train)
main.add_command(test)
main.add_command(to_pickle)
main.add_command(amplify)
main.add_command(concepts)
main.add_command(concept_heatmaps)
main.add_command(logs)


if __name__ == "__main__":
    main()
