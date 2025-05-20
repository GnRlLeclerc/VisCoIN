import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import click

from viscoin.models.latent_editor import LatentEditorDiffusion
from viscoin.models.unet import DiffusionUNetAdapted

from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.concept2clip import Concept2CLIP
from viscoin.models.classifiers import Classifier

from transformers import AutoTokenizer, CLIPVisionModelWithProjection
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DDIMScheduler,
    AutoencoderKL,
)

from ip_adapter import IPAdapter

import open_clip

import clip

from viscoin.cli.training import (
    DEFAULT_CHECKPOINTS,
    DATASET_MODEL_PARAMS,
    DATASET_CLASSES,
    load_classifier,
)

from viscoin.datasets.transforms import RESNET_TEST_TRANSFORM

DENOISING_STEPS = 50
STRENGTH = 0.8
GUIDANCE_SCALE = 7.5


def find_principal_directions(self, latent_tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Finds the principal directions of the vae latent space using SVD.

    Args:
        latent_tensor (torch.Tensor): (N_images, 4, 64, 64)

    Returns:
        torch.Tensor: A tensor of shape (num_directions, C, H, W).
    """

    N_images, C, H, W = latent_tensors[0].shape
    original_shape = (C, H, W)

    Uts = []
    svals = []

    for i in range(len(latent_tensors)):

        flattened_tensor = (
            latent_tensors[i].view(N_images, -1).to(torch.float32)
        )  # Shape: (N_images, C * H * W)

        # Center the data
        mean = torch.mean(flattened_tensor, dim=0)
        centered_tensor = flattened_tensor - mean

        V, S, _ = torch.linalg.svd(centered_tensor @ centered_tensor.T, full_matrices=False)
        sval = S**0.5

        Ut = torch.diag(sval**-1) @ V.T @ centered_tensor

        Ut = Ut.view(-1, *original_shape)

        Uts.append(Ut)
        svals.append(sval)

    return Uts, svals


@click.command()
@click.option(
    "--dataset_path",
    default="./datasets/ffhq/",
    help="Path to the dataset folder.",
)
@click.option(
    "--save_path",
    default="./../Experiments/images/diffusion_amp/principal_directions.pt",
    help="Path to save the principal directions.",
)
def get_dataset_principal_directions(
    dataset_path: str,
    save_path: str,
) -> None:
    """Saves the principal directions of the vae latent space of a dataset.

    Args:
        dataset_path (str, optional): Defaults to "./datasets/ffhq/".
        save_path (str, optional): Where to save the directions (.pt)
        num_directions (int, optional): Defaults to 10.
    """

    batch_size = 1
    num_samples = 500

    editor = LatentEditorDiffusion(
        device="cuda", num_inference_steps=DENOISING_STEPS, strength=STRENGTH, batch_size=batch_size
    )

    dataset = datasets.ImageFolder(
        dataset_path,
        transform=transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        ),
    )

    latents = [[] for _ in range(len(editor.timesteps))]

    with torch.no_grad():
        for i in tqdm.tqdm(range(num_samples)):

            sample_image = dataset[i][0].unsqueeze(0).to(editor.dtype)

            # Preprocess the image
            sample_image = editor.preprocess_image(sample_image)

            # Encode the image
            encoded_image = editor.encode_image(sample_image)

            # Compute the vae latent
            sample_latent = editor.compute_latent(encoded_image)

            for j in range(len(editor.timesteps)):
                t = editor.timesteps[j]

                latent_model_input = (
                    torch.cat([sample_latent] * 2)
                    if editor.do_classifier_free_guidance
                    else sample_latent
                )
                latent_model_input = editor.scheduler.scale_model_input(latent_model_input, t).to(
                    editor.dtype
                )

                # Unet downwards pass
                initial_sample, down_block_res_samples, emb, encoder_hidden_states = (
                    editor.unet.downwards(
                        latent_model_input,
                        t,
                        editor.prompt_embeds,
                    )
                )

                # Unet mid pass
                initial_sample = editor.unet.middle(
                    initial_sample,
                    emb,
                    encoder_hidden_states,
                )

                # Unet upwards pass
                noise_pred = editor.unet.upwards(
                    initial_sample, down_block_res_samples, emb, encoder_hidden_states
                )

                if editor.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (
                        noise_pred_text - noise_pred_uncond
                    )

                extra_step_kwargs = editor.pipe.prepare_extra_step_kwargs(None, eta=0.0)

                # compute the previous noisy sample x_t -> x_t-1
                sample_latent = editor.scheduler.step(
                    noise_pred, t, sample_latent, **extra_step_kwargs, return_dict=False
                )[0]

                latents[j].append(initial_sample)

    editor.pipe.maybe_free_model_hooks()

    latents = [torch.cat(latent_t, dim=0) for latent_t in latents]

    directions, svals = find_principal_directions(latents)

    torch.save({"directions": directions, "svals": svals}, save_path)


@click.command()
@click.option(
    "--image_path",
    default="./datasets/ffhq/0/00113.png",
    help="Path to the image.",
)
@click.option(
    "--save_path",
    default="./../Experiments/images/diffusion_amp",
    help="Path to save the amplified images.",
)
def amplify_diffusion(
    image_path: str,
    save_path: str,
) -> None:
    image = Image.open(image_path)

    image.save(f"{save_path}/original.png")

    editor = LatentEditorDiffusion(
        device="cuda", num_inference_steps=DENOISING_STEPS, strength=STRENGTH
    )

    # Preprocess the image
    image = editor.preprocess_image(image)

    data = torch.load("./../Experiments/images/diffusion_amp/principal_directions.pt")

    directions = data["directions"]
    svals = data["svals"]

    print(len(directions))

    # print(f"Directions shape: {directions.shape}")

    lambdas = [-3, -2, -1, 0, 1, 2, 3]
    direction_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    images = {i: {} for i in direction_indices}

    # Encode the image
    encoded_image = editor.encode_image(image)

    # Compute the vae latent
    initial_latents = editor.compute_latent(encoded_image)
    # initial_latents = editor.unet.sample(num_samples=1)

    for dir in direction_indices:
        with torch.no_grad():
            print(f"Direction {dir}")

            extra_step_kwargs = editor.pipe.prepare_extra_step_kwargs(None, eta=0.0)

            for i, lambda_ in enumerate(lambdas):
                print(f"Lambda: {lambda_}")
                # Get the unet input

                print(f"Number of steps: {len(editor.timesteps)}")

                # Reset the scheduler and timestep
                editor.setup_timestep()

                latents = initial_latents.clone()

                for j, t in enumerate(editor.timesteps):

                    latent_model_input = (
                        torch.cat([latents] * 2) if editor.do_classifier_free_guidance else latents
                    )
                    latent_model_input = editor.scheduler.scale_model_input(latent_model_input, t)

                    # Unet downwards pass
                    sample, down_block_res_samples, emb, encoder_hidden_states = (
                        editor.unet.downwards(
                            latent_model_input,
                            t,
                            editor.prompt_embeds,
                        )
                    )

                    # Unet mid pass
                    sample = editor.unet.middle(
                        sample,
                        emb,
                        encoder_hidden_states,
                    )

                    direction = (
                        directions[j][dir].unsqueeze(0).to(editor.device).to(editor.dtype).clone()
                    )
                    # direction = direction / torch.linalg.norm(direction)
                    direction *= svals[j][dir].to(editor.dtype) / 5

                    # print(f"Direction norm: {torch.linalg.norm(direction)}")
                    # print(f"Sample norm: {torch.linalg.norm(sample)}")

                    sample = sample + lambda_ * direction

                    # Unet upwards pass
                    noise_pred = editor.unet.upwards(
                        sample, down_block_res_samples, emb, encoder_hidden_states
                    )

                    # perform guidance
                    if editor.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (
                            noise_pred_text - noise_pred_uncond
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

                images[dir][lambda_] = generated_image

    # Plot an image grid with all these images
    fig, axs = plt.subplots(len(direction_indices), len(lambdas), figsize=(20, 20))
    for i, dir in enumerate(direction_indices):
        for j, lambda_ in enumerate(lambdas):
            axs[i, j].imshow(images[dir][lambda_])
            axs[i, j].axis("off")
            if j == 0:
                axs[i, j].set_ylabel(f"Direction {dir}", fontsize=20)
            if i == 0:
                axs[i, j].set_title(f"Lambda {lambda_}", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{save_path}/amplified.png")


@click.command()
@click.option(
    "--image_path",
    default="./datasets/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg",
)
def image_to_prompt_diffusion(
    image_path: str,
):
    """Takes an image and gets its clip embedding using Concept2CLIP.
    Then this embedding is given as a prompt to the diffusion model.

    Args:
        image_path (str): The path of the image to be converted to a prompt.
    """

    dataset_type = "cub"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    original_image = Image.open(image_path)
    image = RESNET_TEST_TRANSFORM(original_image).unsqueeze(0).to(device)

    # Load the models

    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Loading the appropriate clip adapter model
    clip_embedding_dim = clip_model.visual.output_dim

    editor = LatentEditorDiffusion(
        device=device, num_inference_steps=DENOISING_STEPS, strength=STRENGTH
    )

    concept_extractor = ConceptExtractor(
        n_concepts=DATASET_MODEL_PARAMS[dataset_type]["n_concepts"]
    ).to(device)

    classifier = load_classifier(
        DEFAULT_CHECKPOINTS[dataset_type]["classifier"], DATASET_CLASSES[dataset_type]
    ).to(device)

    clip_adapter = torch.load("checkpoints/cub/concept2clip.pkl").to(device)

    # Get the CLIP embedding of the image from its VisCoIN concepts

    classes, hidden_states = classifier.forward(image)
    encoded_concepts, extra_info = concept_extractor.forward(hidden_states[-3:])
    clip_embeddings = clip_adapter.forward(encoded_concepts)

    base_model_path = "runwayml/stable-diffusion-v1-5"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    ip_ckpt = "checkpoints/ip-adapter_sd15.bin"

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

    # load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    )

    # load ip-adapter
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

    # generate image variations
    images = ip_model.generate(
        pil_image=original_image, num_samples=1, num_inference_steps=50, seed=42
    )

    images_concept2clip = ip_model.generate(
        clip_image_embeds=clip_embeddings,
        num_samples=1,
        num_inference_steps=50,
        seed=42,
    )

    print(images)

    # plot original and output image
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(original_image)
    axs[0].axis("off")
    axs[0].set_title("Original Image")
    axs[1].imshow(images[0])
    axs[1].axis("off")
    axs[1].set_title("Output Image")
    axs[2].imshow(images_concept2clip[0])
    axs[2].axis("off")
    axs[2].set_title("Output Image (C2C)")
    plt.tight_layout()
    plt.show()
