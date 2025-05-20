"""
CLI entrypoint for interacting with the viscoin library.

Example usage:
```bash
python main.py test resnet50 --batch-size 32 --dataset-path datasets/CUB_200_2011/
```

Will be subject to many changes as the project evolves.
"""

import click

from viscoin.cli.training import train, test, to_pickle

from viscoin.cli.concept_labels import clip_concept_labels, evalutate_concept_captions

from viscoin.cli.visualization import amplify, amplify_single, concepts, concept_heatmaps, logs

from viscoin.cli.diffusion import (
    amplify_diffusion,
    get_dataset_principal_directions,
    image_to_prompt_diffusion,
)

# Imports Trogon if installed : Terminal User Interface for Click commands
try:
    from trogon import tui
except ImportError:

    def tui():
        return lambda f: f


@tui()
@click.group()
def main():
    pass


main.add_command(train)
main.add_command(test)
main.add_command(to_pickle)

main.add_command(clip_concept_labels)
main.add_command(evalutate_concept_captions)

main.add_command(amplify)
main.add_command(amplify_single)
main.add_command(concepts)
main.add_command(concept_heatmaps)
main.add_command(logs)

main.add_command(amplify_diffusion)
main.add_command(get_dataset_principal_directions)
main.add_command(image_to_prompt_diffusion)


if __name__ == "__main__":
    main()
