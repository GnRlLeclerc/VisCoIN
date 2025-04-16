"""Generated captions for the CUB dataset.

These captions were generated using CUB annotations as a base,
in the spirit of PEEB https://arxiv.org/abs/2403.05297 (captions = parts + variations)
"""

import os
from typing import Literal

parts = [
    "back",
    "beak",
    "belly",
    "breast",
    "crown",
    "forehead",
    "eye",
    "wings",
    "nape",
    "leg",
    "tail",
    "throat",
    "eyeline",
    "eyering",
    "cap",
    "crest",
    "eyebrow",
]

colors = [
    "orange",
    "pink",
    "black",
    "purple",
    "brown",
    "yellow",
    "green",
    "grey",
    "red",
    "blue",
    "white",
    "olive",
    "iridescent",
    "buff",
    "rufous",
    "multi-colored",
]

sizes = [
    "small",
    "medium",
    "large",
    "long",
    "broad",
]

patterns = [
    "striped",
    "solid",
    "plain",
    "spotted",
]

beak_shapes = [
    "hooked",
    "needle",
    "dagger",
    "cone",
    "spatulate",
    "curved",
]

tail_shapes = [
    "forked",
    "notched",
    "rounded",
    "fan-shaped",
    "pointed",
    "squared",
]

wing_shapes = [
    "rounded",
    "tapered",
    "pointed",
]


bonus = {
    "wings": wing_shapes,
    "tail": tail_shapes,
    "beak": beak_shapes,
}

size_whitelist = [
    "beak",
    "wings",
    "tail",
]


def _add_sizes(caption: str, part: str) -> list[str]:
    """Add sizes to the caption if the part is in the whitelist."""

    if part in size_whitelist:
        return [f"{caption} {size}" for size in sizes]
    else:
        return [caption]


def generate() -> list[str]:
    """Generate all possible captions."""

    captions: list[str] = []

    for part in parts:
        for color in colors:
            for pattern in patterns:
                if part in bonus:
                    for shape in bonus[part]:
                        captions.extend(_add_sizes(f"{pattern} {color} {shape} {part}", part))
                else:
                    captions.extend(_add_sizes(f"{pattern} {color} {part}", part))

    return captions


def load(prefix: Literal["a bird with"] | None = None) -> list[str]:
    """Load the generated captions from disk.

    A prefix can be inserted to the captions:
    - when analyzing concepts, use no prefix, e.g. "striped orange back"
    - when analyzing whole images, use a prefix, e.g. "a bird with striped orange back"

    Args:
        prefix: The prefix to add to the captions. Defaults to None.
    """
    filepath = os.path.join("viscoin", "captions", "cub.txt")
    with open(filepath, "r") as f:
        captions = f.readlines()

    for i, caption in enumerate(captions):
        caption = caption.strip()
        captions[i] = f"{prefix} {caption.strip()}" if prefix else caption

    return captions


if __name__ == "__main__":
    """Generate captions for the CUB dataset.

    ```bash
    python -m viscoin.captions.cub
    ```"""

    captions = generate()

    print("Generated", len(captions), "captions")

    filepath = os.path.join("viscoin", "captions", "cub.txt")

    with open(filepath, "w") as f:
        for caption in captions:
            f.write(caption + "\n")
