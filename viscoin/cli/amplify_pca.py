import click
import numpy as np
import numpy.random as rd
import torch
from sklearn.decomposition import PCA

from viscoin.cli.utils import dataset, device, viscoin_pickle_path
from viscoin.datasets.utils import DatasetType, get_datasets
from viscoin.models.utils import load_viscoin_pickle
from viscoin.utils.images import from_torch
from viscoin.utils.plotting import plot_grid


@click.command()
@dataset
@device
@viscoin_pickle_path
@click.option(
    "--pca",
    help="Amount of PCA components to compute",
    default=10,
)
@click.option(
    "--image",
    help="Index of the test image to amplify. If not provided, a random image will be used.",
    type=int,
)
def amplify_pca(
    viscoin_pickle_path: str, dataset: DatasetType, device: str, pca: int, image: int | None
):
    """
    Amplify a random test image using PCA directions.
    """

    models = load_viscoin_pickle(viscoin_pickle_path)

    _, test_dataset = get_datasets(dataset, "test")
    train_w, test_w = models.compute_w_space(dataset, device)

    # Fit the PCA on the training set
    pca_ = PCA(n_components=pca)
    pca_.fit(train_w.view(train_w.shape[0], -1).numpy())

    # Choose a random image from the test set to amplify
    index = rd.randint(0, len(test_dataset)) if image is None else image  # type: ignore
    print("Amplifying image n°", index)

    # Compute amplified latent space
    multipliers = (0, 1, 5, 10, 15)
    factors = torch.tensor(multipliers).view(1, len(multipliers), 1, 1)
    components = torch.from_numpy(pca_.components_).view(pca, *train_w.shape[1:]).unsqueeze(1)
    latent = test_w[index].unsqueeze(0).unsqueeze(0)

    # Shape: (n_components, n_factors, *gan.w_shape)
    amplified = latent + components * factors

    # Regenerate images from the
    images = np.zeros((pca, len(multipliers), 256, 256, 3))
    gan = models.gan.to(device)

    for i, component in enumerate(amplified):
        # shape (len factors, *gan.w_shape)
        component = component.to(device)
        img = gan.gen_from_w(component).detach().cpu()
        images[i] = from_torch(img)

    for i in range(int(np.ceil(pca / 5))):
        imgs = images[i * 5 : (i + 1) * 5]
        plot_grid(
            imgs,
            "W+ amplification along PCA component",
            [f"x{factor}" for factor in multipliers],
            [f"Component {j +1}" for j in range(i * 5, (i + 1) * 5)],
        )
