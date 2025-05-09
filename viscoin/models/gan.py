# pyright:  reportUnusedVariable=none
"""Adapted StyleGAN ADA implementation from https://github.com/NVlabs/stylegan2-ada-pytorch by Jayneel Parekh"""

import os
import sys

import torch
from torch import Tensor
from torchvision import transforms


# Add stylegan2_ada submodule to the python path so that imports within this submodule work properly
def fix_path():
    pass  # Pass because the following line will be executed.
    # Use this function as a way to prevent isort from placing the stylegan import before this import


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "stylegan2_ada"))


from stylegan2_ada.torch_utils import misc
from stylegan2_ada.training.networks import (
    FullyConnectedLayer,
    SynthesisNetwork,
    normalize_2nd_moment,
)

_default_viscoin_mapping_kwargs: dict = {
    "coarse_layer": 2,
    "mid_layer": 10,
    "num_layers": 1,
}


class GeneratorAdapted(torch.nn.Module):
    """StyleGAN adapted for VisCoIN"""

    def __init__(
        self,
        z_dim=256,  # Input latent (Z) dimensionality. for z1, z2
        c_dim=0,  # Conditioning label (C) dimensionality.
        w_dim=512,  # Intermediate latent (W) dimensionality.
        img_resolution=256,  # Output resolution.
        img_channels=3,  # Number of output color channels.
        small_adjust=False,  # Small adjustment for a specific baseline model
        low_res256=False,  # Output resolution fixed to 256
        mapping_kwargs=None,  # Arguments for MappingNetwork.
        synthesis_kwargs={},  # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        if mapping_kwargs is None:
            mapping_kwargs = _default_viscoin_mapping_kwargs

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **synthesis_kwargs,
        )
        self.num_ws = self.synthesis.num_ws
        self.low_res256 = low_res256
        self.resizer = transforms.Resize((256, 256))
        self.cropper = transforms.CenterCrop(size=(384, 512))
        # self.mapping = MappingNetwork_v3(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs) # Used for analyze6, 12, 13
        # self.mapping = MappingNetwork_v4(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        # self.mapping = MappingNetworkAdapted(z1_dim=z_dim, z2_dim=8*z_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs) # Used until analyze18 for uneven (UE) representation size models
        # self.mapping = MappingNetworkAdapted(z1_dim=9*z_dim, z2_dim=8*z_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs) # Used for analyze21, 25, 26
        self.mapping = MappingNetworkAdapted(
            z1_dim=9 * z_dim, z2_dim=9 * z_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs
        )
        if small_adjust:
            print("Small adjustment to size of extra representation for analyze21 CUB model")
            self.mapping = MappingNetworkAdapted(
                z1_dim=9 * z_dim,
                z2_dim=8 * z_dim,
                w_dim=w_dim,
                num_ws=self.num_ws,
                **mapping_kwargs,
            )
        # self.mapping = MappingNetworkAdapted(z1_dim=64*z_dim, z2_dim=8*z_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs) # Used for analyze22, 23, 24

    def forward(
        self,
        z1,
        z2,
        return_latents=False,
        truncation_psi=1,
        truncation_cutoff=None,
        **synthesis_kwargs,
    ):
        ws = self.mapping(
            z1, z2, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff
        )

        # print (ws.shape)
        img = self.synthesis(ws, **synthesis_kwargs)
        if self.low_res256:
            img = self.resizer(self.cropper(img))
            # print ("Called here")
        if return_latents:
            return img, ws
        return img

    def gen_from_w(
        self,
        w: Tensor,
        **synthesis_kwargs,
    ) -> Tensor:
        """Generate images directly from w latents (useful for W+ space analysis)"""
        img = self.synthesis(w, **synthesis_kwargs)
        if self.low_res256:
            img = self.resizer(self.cropper(img))
        return img

    @classmethod
    def from_gan(
        cls,
        gan: torch.nn.Module,
        z_dim=256,  # Input latent (Z) dimensionality. for z1, z2
        w_dim=512,  # Intermediate latent (W) dimensionality.
        small_adjust=False,  # Small adjustment for a specific baseline model
        mapping_kwargs=None,  # Arguments for MappingNetwork.
    ) -> "GeneratorAdapted":
        """
        Returns an GAN model adapted for VisCoIN from a StyleGAN model.
        """

        assert gan.synthesis is not None, "The provided GAN is not a valid StyleGAN model"

        model = cls(
            z_dim=z_dim,
            w_dim=w_dim,
            small_adjust=small_adjust,
            mapping_kwargs=mapping_kwargs,
        )

        model.synthesis = gan.synthesis

        return model


class MappingNetworkAdapted(torch.nn.Module):
    """Mapping network adapted for VisCoIN"""

    def __init__(
        self,
        z1_dim,  # Input latent 1 (Z1) dimensionality, 0 = no latent.
        z2_dim,  # Input latent 2 (Z2) dimensionality, 0 = no latent.
        w_dim,  # Intermediate latent (W) dimensionality.
        num_ws,  # Number of intermediate latents to output, None = do not broadcast.
        num_layers=8,  # Number of mapping layers.
        embed_features=None,  # Label embedding dimensionality, None = same as w_dim.
        layer_features=None,  # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=0.01,  # Learning rate multiplier for the mapping layers.
        w_avg_beta=0.995,  # Decay for tracking the moving average of W during training, None = do not track.
        fixed_w_avg=None,  # If needed to supply info about average style vector for synthesis network, this variable will store avg w vector
        coarse_layer=3,  # The generator layer index below which the style vectors are predicted using support representation
        mid_layer=12,  # The generator layer index over which the style vectors are predicted using support representation
    ):
        super().__init__()
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        self.coarse_idx = coarse_layer
        # self.coarse_idx = 0 # Used for anaylze24, 25, 26
        self.mid_idx = mid_layer

        if embed_features is None:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list1 = [z1_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        features_list2 = [z2_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        if fixed_w_avg is not None:
            self.fixed_w_avg = torch.nn.Parameter(fixed_w_avg)
        else:
            # NOTE : additional fix after Jayneel's code, initializing fixed_w_avg if not provided
            self.fixed_w_avg = torch.nn.Parameter(torch.zeros([w_dim]))

        for i in range(num_ws):
            layers = []

            for idx in range(0, num_layers):
                if i < self.coarse_idx or i > self.mid_idx:
                    in_features = features_list2[idx]
                    out_features = features_list2[idx + 1]
                else:
                    in_features = features_list1[idx]
                    out_features = features_list1[idx + 1]
                layers.append(
                    FullyConnectedLayer(
                        in_features,
                        out_features,
                        activation=activation,
                        lr_multiplier=lr_multiplier,
                    )
                )

            layer = torch.nn.Sequential(*layers)
            setattr(self, f"s{i}", layer)

    def forward(self, z1, z2, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=True):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function("input"):
            if len(z1.shape) > 3:
                z1 = z1.view(-1, self.z1_dim)  # The IF statement added for 2D Phi
            if self.z1_dim > 0:
                misc.assert_shape(z1, [None, self.z1_dim])
                misc.assert_shape(z2, [None, self.z2_dim])
                x1 = normalize_2nd_moment(z1.to(torch.float32))
                x2 = normalize_2nd_moment(z2.to(torch.float32))

        styles = []
        for i in range(self.num_ws):
            layer = getattr(self, f"s{i}")
            if i < self.coarse_idx or i > self.mid_idx:
                # print (i, x2.shape, layer)
                styles.append(layer(x2))  # type: ignore
            else:
                styles.append(layer(x1))  # type: ignore
        ws = torch.stack(styles, dim=1)
        # print (ws.shape)
        ws = ws + self.fixed_w_avg.repeat([ws.shape[0], ws.shape[1], 1])
        return ws
