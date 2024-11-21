import torch
import torch.nn.functional as F
import clip
import tqdm

from loralib.utils import (
    mark_only_lora_as_trainable,
    apply_lora,
    get_lora_parameters,
    save_lora,
    load_lora,
    cls_acc,
)
from loralib import layers as lora_layers
from typing import Dict, List


def class_features(clip_model: torch.nn.Module, classnames: List[str], device: str) -> torch.Tensor:
    """
    Compute the feature embeddings for class names using the CLIP model.

    Args:
        clip_model: The CLIP model used for encoding text.
        classnames: List of class names to encode.
        device: Device to use for computation (e.g., "cuda" or "cpu").

    Returns:
        Normalized tensor of class feature embeddings.
    """

    # Define a template for the class names : "Eagle" -> "A photo of a eagle."
    template = "A photo of a {}."

    with torch.no_grad():
        texts = [template.format(classname.replace("_", " ")) for classname in classnames]
        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            texts = clip.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    return text_features


def evaluate_lora(
    args, clip_model: torch.nn.Module, loader: torch.utils.data.DataLoader, dataset
) -> float:
    """
    Evaluate the model with LoRA on the given dataset loader.

    Args:
        args: Argument namespace containing configuration.
        clip_model: The CLIP model to evaluate.
        loader: DataLoader for the evaluation dataset.
        dataset: Dataset object containing class labels.

    Returns:
        Accuracy of the model on the evaluation dataset.
    """
    clip_model.eval()
    text_features = class_features(clip_model, list(dataset.class_labels.values()), args.device)

    acc = 0.0
    tot_samples = 0

    with torch.no_grad():
        for images, target in loader:
            images, target = images.to(args.device), target.to(args.device)
            with torch.amp.autocast(device_type=args.device, dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)

    acc /= tot_samples
    return acc


def run_lora(
    args,
    clip_model: torch.nn.Module,
    dataset,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
) -> None:
    """
    Train and evaluate the LoRA-augmented CLIP model.

    Args:
        args: See main function for details.
        clip_model: The CLIP model to train and evaluate.
        dataset: Dataset object containing class labels.
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the test dataset.
    """

    list_lora_layers = apply_lora(args, clip_model)

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    # Set LoRA parameters as trainable
    mark_only_lora_as_trainable(clip_model)

    total_iters = args.n_iters * args.shots
    optimizer = torch.optim.AdamW(
        get_lora_parameters(clip_model),
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    scaler = torch.amp.GradScaler(args.device)

    clip_model = clip_model.to(args.device)

    try:

        for iters in range(total_iters):
            clip_model.train()
            acc_train = 0.0
            tot_samples = 0
            loss_epoch = 0.0

            for images, target, caption in tqdm.tqdm(train_loader):
                images, target = images.to(args.device), target.to(args.device)

                # Compute texts embeddings
                with torch.amp.autocast(device_type=args.device, dtype=torch.float16):
                    texts = clip.tokenize(caption).to(args.device)
                    texts_embeddings = clip_model.encode_text(texts)
                text_features = texts_embeddings / texts_embeddings.norm(dim=-1, keepdim=True)

                # Compute image embeddings
                with torch.amp.autocast(device_type=args.device, dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Compute logits
                logits_per_image = args.logit_scale * image_features @ text_features.t()
                logits_per_text = args.logit_scale * text_features @ image_features.t()

                # Compute loss per image and per text and average them

                # The labels ensure that the image should be paired with the corresponding text (image 0 with text 0, etc.)
                # The similarity matrix should be all zeros except for the diagonal
                labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

                image_loss = F.cross_entropy(logits_per_image, labels)
                text_loss = F.cross_entropy(logits_per_text, labels)

                loss = (image_loss + text_loss) / 2

                # Compute classifier accuracy on each batch
                test_prediction = (
                    args.logit_scale
                    * image_features
                    @ class_features(
                        clip_model, list(dataset.class_labels.values()), args.device
                    ).t()
                )

                acc_train += cls_acc(test_prediction, target) * target.shape[0]

                loss_epoch += loss.item() * target.shape[0]
                tot_samples += target.shape[0]

                # Step and update
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print("LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}".format(current_lr, acc_train, loss_epoch))

    # Save the model if interrupted by ctrl+c
    except KeyboardInterrupt:
        if args.save_path:
            print("\n**** Training interrupted. Saving model. ****\n")
            save_lora(args, list_lora_layers)
    except Exception as e:
        raise e

    acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))

    if args.save_path:
        save_lora(args, list_lora_layers)
