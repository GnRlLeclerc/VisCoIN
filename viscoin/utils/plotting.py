"""Plotting utils"""

import matplotlib.pyplot as plt
import numpy as np


def plot_grid(
    images: np.ndarray, title: str, column_titles: list[str], row_titles: list[str]
) -> None:
    """Plot a grid of images with row and column titles on the sides.

    Args:
        images: array of images, shape must be (rows, cols, 3, 256, 256)
        title: title of the plot
        column_titles: titles of the columns
        row_titles: titles of the rows
    """

    rows = len(row_titles)
    cols = len(column_titles)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle(title)

    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            ax.imshow(images[i, j])

            if i == 0:
                ax.set_title(column_titles[j])

            if j == 0:
                ax.set_ylabel(row_titles[i], rotation=0, labelpad=40)

            # Turn off axis (without removing ylabel)
            ax.xaxis.set_visible(False)
            plt.setp(ax.spines.values(), visible=False)
            ax.tick_params(left=False, labelleft=False)

    plt.show()
