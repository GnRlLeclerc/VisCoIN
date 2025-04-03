"""
The goal of this model is to discover directions in the latent space of a generative model and amplify them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import sys
import os

import tqdm

from diffusers import StableDiffusionImg2ImgPipeline

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

sys.path.append(os.path.join(os.path.dirname(__file__), "./../.."))

from viscoin.models.unet import DiffusionUNetAdapted

PROMPT_EMBED_SHAPE = (77, 1024)


class LatentEditorDiffusion(nn.Module):

    def __init__(
        self,
        device: str = "cuda",
        num_inference_steps: int = 2,
        strength: float = 0.5,
        batch_size: int = 1,
    ):
        """Class to find relevant directions in the latent space of a generative model and amplify them.
        The functions were adapted from the pipeline of this model : https://huggingface.co/stabilityai/sd-turbo

        Args:
            device (str, optional): Defaults to "cuda".

            num_inference_steps (int, optional): Defaults to 2.
            strength (float, optional): Defaults to 0.5.
            --> The actual number of steps of the diffusion process is int(num_inference_steps * strength).
        """

        super(LatentEditorDiffusion, self).__init__()
        model_id = "stabilityai/sd-turbo"

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        )

        self.pipe.to(device)

        self.scheduler = self.pipe.scheduler
        self.vae = self.pipe.vae
        self.unet = DiffusionUNetAdapted(self.pipe.unet)

        # Parameters for the diffusion pipeline
        self.strength = strength
        self.num_inference_steps = num_inference_steps
        self.dtype = torch.float16
        self.batch_size = batch_size
        self.num_images_per_prompt = 1
        self.device = device

        self.setup_timestep()

    def retrieve_timesteps(
        self,
        scheduler,
        num_inference_steps,
        device: str,
        **kwargs,
    ):
        """
        Taken from diffusers: pipeline_stable_diffusion_img2img.py
        Returns timesteps values for a given number of inference steps.
        """
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        return timesteps, num_inference_steps

    def setup_timestep(self):
        """Setup the timesteps for the diffusion process."""
        timesteps, num_inference_steps = self.retrieve_timesteps(
            self.scheduler, self.num_inference_steps, self.device
        )

        self.timesteps, self.num_diffusion_steps = self.pipe.get_timesteps(
            num_inference_steps,
            self.strength,
            self.device,
        )

        self.latent_timestep = timesteps[:1].repeat(self.batch_size * self.num_images_per_prompt)

    def preprocess_image(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        return (
            self.pipe.image_processor.preprocess(image).to(dtype=self.dtype).to(device=self.device)
        )

    def postprocess_image(self, image: torch.Tensor) -> Image.Image:
        output_type = "pil"
        return self.pipe.image_processor.postprocess(image.detach(), output_type=output_type)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode an image into the latent space using the VAE encoder."""
        with torch.no_grad():
            image = self.vae.encode(image).latent_dist.sample()
            image = image * self.vae.config.scaling_factor
            return image.to(self.device)

    def compute_latent(self, encoded_image: torch.Tensor) -> torch.Tensor:
        """From an encoded image, add noise to the latent representation and return it."""
        shape = encoded_image.shape
        noise = torch.randn(
            shape, generator=None, device=self.device, dtype=self.dtype, layout=torch.strided
        ).to(self.device)

        return self.scheduler.add_noise(encoded_image, noise, self.latent_timestep)

    def generate_image(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Generate an image from the latent representation using the unet from the diffusion model and decoding with the vae.
        Args:
            latents (torch.Tensor): The latent representation from the compute_latent function.
        """
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(None, eta=0.0)

        if len(self.timesteps) > 1:
            print("Multi-step diffusion..")

        # Denoising loop
        for i, t in enumerate(self.timesteps):
            latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=torch.randn(
                        (self.batch_size, PROMPT_EMBED_SHAPE[0], PROMPT_EMBED_SHAPE[1]),
                        dtype=self.dtype,
                    ).to(self.device),
                )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=False
            )[0]

        # Decode the latents to get the image
        with torch.no_grad():
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False, generator=None
            )[0]

        # Offload all models
        self.pipe.maybe_free_model_hooks()

        return image

    def image_from_noise_residual(
        self, noise_pred: torch.Tensor, timestep: torch.Tensor, latents: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate an image from the noise residual using the unet from the diffusion model and decoding with the vae.
        Args:
            noise_pred (torch.Tensor): The noise residual from the compute_latent function.
            timestep (torch.Tensor): The timestep of the diffusion process.
            latents (torch.Tensor): The latent representation from the compute_latent function.
        """
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(None, eta=0.0)

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(
            noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False
        )[0]

        # Decode the latents to get the image
        with torch.no_grad():
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False, generator=None
            )[0]

        # Offload all models
        self.pipe.maybe_free_model_hooks()

        return image

    def find_principal_directions(
        self, latent_tensor: torch.Tensor, num_directions: int = 10
    ) -> torch.Tensor:
        """
        Finds the principal directions of the vae latent space using SVD.

        Args:
            latent_tensor (torch.Tensor): (N_images, 4, 64, 64)
            num_directions (int, optional): defaults to 10.

        Returns:
            torch.Tensor: A tensor of shape (num_directions, C, H, W).
        """
        N_images, C, H, W = latent_tensor.shape
        original_shape = (C, H, W)
        flattened_tensor = latent_tensor.view(N_images, -1)  # Shape: (N_images, C * H * W)

        # Center the data
        mean = torch.mean(flattened_tensor, dim=0)
        centered_tensor = flattened_tensor - mean

        # Perform Singular Value Decomposition (SVD)
        U, S, V = torch.linalg.svd(centered_tensor.to(torch.float32))

        principal_directions_list = []
        for i in range(num_directions):
            if i < V.shape[1]:
                principal_direction_flattened = V[:, i]
                principal_direction = principal_direction_flattened.view(original_shape)
                principal_directions_list.append(principal_direction)
            else:
                break  # Stop if we run out of principal components

        return torch.stack(principal_directions_list, dim=0)


def get_dataset_principal_directions(
    dataset_path: str = "./datasets/ffhq/",
    save_path: str = "./../Experiments/images/diffusion_amp/principal_directions.pt",
    num_directions: int = 10,
) -> None:
    """Saves the principal directions of the vae latent space of a dataset.

    Args:
        dataset_path (str, optional): Defaults to "./datasets/ffhq/".
        save_path (str, optional): Where to save the directions (.pt)
        num_directions (int, optional): Defaults to 10.
    """

    batch_size = 32

    dataset = datasets.ImageFolder(root=dataset_path, transform=transforms.ToTensor())
    dataset, _ = torch.utils.data.random_split(
        dataset, [int(len(dataset) * 0.1), len(dataset) - int(len(dataset) * 0.1)]
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    editor = LatentEditorDiffusion(
        device="cuda", num_inference_steps=3, strength=0.4, batch_size=batch_size
    )

    latents = []

    with torch.no_grad():
        for i, (img, _) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):

            if img.shape[0] != batch_size:
                continue

            preprocess = editor.preprocess_image(img)
            encoded_image = editor.encode_image(preprocess)
            latent = editor.compute_latent(encoded_image)

            latent_model_input = latent
            latent_model_input = editor.scheduler.scale_model_input(
                latent_model_input, editor.latent_timestep
            )
            t = editor.timesteps[0]

            # Unet downwards pass
            initial_sample, down_block_res_samples, emb, encoder_hidden_states = (
                editor.unet.downwards(
                    latent_model_input,
                    t,
                    torch.zeros(
                        (img.shape[0], PROMPT_EMBED_SHAPE[0], PROMPT_EMBED_SHAPE[1]),
                        dtype=editor.dtype,
                    ).to(editor.device),
                )
            )

            # Unet mid pass
            initial_sample = editor.unet.middle(
                initial_sample,
                emb,
                encoder_hidden_states,
            )

            latents.append(initial_sample)

    print(f"Latents shape: {latents[0].shape}")

    latents = torch.cat(latents, dim=0)

    with torch.no_grad():
        directions = editor.find_principal_directions(latents, num_directions=num_directions)

    print(f"Directions shape: {directions.shape}")

    torch.save(directions, save_path)


def amplify_diffusion(
    image_path: str = "./datasets/ffhq/0/00095.png",
    save_path: str = "./../Experiments/images/diffusion_amp",
) -> None:
    image = Image.open(image_path)

    image.save(f"{save_path}/original.png")

    editor = LatentEditorDiffusion(device="cuda", num_inference_steps=300, strength=0.3)

    # Preprocess the image
    image = editor.preprocess_image(image)

    with torch.no_grad():

        extra_step_kwargs = editor.pipe.prepare_extra_step_kwargs(None, eta=0.0)
        random_direction = torch.randn((1, 1280, 8, 8), dtype=editor.dtype).to(editor.device) * 50

        lambdas = [-7, -5, -3, 0, -1, 1, 3, 5, 7]
        for i, lambda_ in enumerate(lambdas):
            print(f"Lambda: {lambda_}")

            # Encode the image
            encoded_image = editor.encode_image(image)

            # Compute the vae latent
            latents = editor.compute_latent(encoded_image)

            # Get the unet input

            print(f"Number of steps: {len(editor.timesteps)}")

            for i, t in enumerate(editor.timesteps):

                latent_model_input = latents
                latent_model_input = editor.scheduler.scale_model_input(latent_model_input, t)

                # Unet downwards pass
                initial_sample, down_block_res_samples, emb, encoder_hidden_states = (
                    editor.unet.downwards(
                        latent_model_input,
                        t,
                        torch.zeros(
                            (editor.batch_size, PROMPT_EMBED_SHAPE[0], PROMPT_EMBED_SHAPE[1]),
                            dtype=editor.dtype,
                        ).to(editor.device),
                    )
                )

                # Unet mid pass
                initial_sample = editor.unet.middle(
                    initial_sample,
                    emb,
                    encoder_hidden_states,
                )

                # Reset the scheduler and timestep
                editor.scheduler._step_index = None

                initial_sample = initial_sample + lambda_ * random_direction

                # Unet upwards pass
                noise_pred = editor.unet.upwards(
                    initial_sample, down_block_res_samples, emb, encoder_hidden_states
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = editor.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

            # Decode the latents to get the image
            generated_image = editor.vae.decode(
                latents / editor.vae.config.scaling_factor, return_dict=False, generator=None
            )[0]

            generated_image = editor.postprocess_image(generated_image)[0]

            generated_image.save(f"{save_path}/amplified_{str(lambda_).replace("-", "neg")}.png")


if __name__ == "__main__":
    get_dataset_principal_directions(
        dataset_path="./datasets/ffhq/",
        save_path="./../Experiments/images/diffusion_amp/principal_directions.pt",
        num_directions=10,
    )
