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

from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

from PIL import Image

# sys.path.append(os.path.join(os.path.dirname(__file__), "./../.."))

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
        # model_id = "stabilityai/sd-turbo"
        # model_id = "ebrahim-k/Stable-Diffusion-1_5-FT-celeba_HQ_en"
        model_id = "runwayml/stable-diffusion-v1-5"
        # model_id = "alibaba-pai/pai-diffusion-food-large-zh"

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        )

        # self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)

        self.scheduler = self.pipe.scheduler
        self.vae = self.pipe.vae.to(device)
        self.unet = DiffusionUNetAdapted(self.pipe.unet).to(device)
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder.to(device)

        # Parameters for the diffusion pipeline
        self.strength = strength
        self.num_inference_steps = num_inference_steps
        self.dtype = torch.float16
        # self.dtype = torch.float32
        self.batch_size = batch_size
        self.num_images_per_prompt = 1
        self.device = device

        self.do_classifier_free_guidance = True

        # self.encode_prompt("a photo of a person")

        # self.setup_timestep()

    def encode_prompt(self, prompt: str, clip_embeddings: torch.Tensor = None):

        if clip_embeddings is None:
            self.prompt_embeds, self.negative_prompt_embeds = self.pipe.encode_prompt(
                prompt, self.device, self.num_images_per_prompt, True
            )
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.pipe.encode_prompt(
                "", self.device, self.num_images_per_prompt, True, prompt_embeds=clip_embeddings
            )

        if self.do_classifier_free_guidance:
            self.prompt_embeds = torch.cat([self.negative_prompt_embeds, self.prompt_embeds])

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

        self.scheduler._step_index = None

        timesteps, num_inference_steps = self.retrieve_timesteps(
            self.scheduler, self.num_inference_steps, self.device
        )

        # self.timesteps, self.num_diffusion_steps = self.pipe.get_timesteps(
        #     num_inference_steps,
        #     self.strength,
        #     self.device,
        # )

        self.timesteps, self.num_diffusion_steps = timesteps, num_inference_steps

        self.latent_timestep = self.timesteps[:1].repeat(
            self.batch_size * self.num_images_per_prompt
        )

    def preprocess_image(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        return (
            self.pipe.image_processor.preprocess(image).to(dtype=self.dtype).to(device=self.device)
        )

    def postprocess_image(self, image: torch.Tensor) -> Image.Image:
        output_type = "pil"
        return self.pipe.image_processor.postprocess(image.detach(), output_type=output_type)

    def encode_text(self, prompt: str) -> torch.Tensor:
        """Encode a text prompt into the latent space using the text encoder."""
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_encoding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_encoding

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
                    encoder_hidden_states=self.prompt_embeds,
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
