"""Generated captions for the CUB dataset.

These captions were generated using CUB annotations as a base,
in the spirit of PEEB https://arxiv.org/abs/2403.05297 (captions = parts + variations)
"""

import os

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


def load() -> list[str]:
    """Load the generated captions from disk."""
    filepath = os.path.join("viscoin", "captions", "cub.txt")
    with open(filepath, "r") as f:
        captions = f.readlines()
    return [c.strip() for c in captions]


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
