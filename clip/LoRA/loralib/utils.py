import os
import torch
import torch.nn as nn
import tqdm
import clip

from .layers import LoRALayer, PlainMultiheadAttentionLoRA

# Predefined attention block index mappings for text and vision encoders
INDEX_POSITIONS_TEXT = {
    "top1": [11],
    "top2": [10, 11],
    "top3": [9, 10, 11],
    "bottom": [0, 1, 2, 3],
    "mid": [4, 5, 6, 7],
    "up": [8, 9, 10, 11],
    "half-up": [6, 7, 8, 9, 10, 11],
    "half-bottom": [0, 1, 2, 3, 4, 5],
    "all": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

INDEX_POSITIONS_VISION = {
    "ViT-B/16": {
        "top": [11],
        "top3": [9, 10, 11],
        "bottom": [0, 1, 2, 3],
        "mid": [4, 5, 6, 7],
        "up": [8, 9, 10, 11],
        "half-up": [6, 7, 8, 9, 10, 11],
        "half-bottom": [0, 1, 2, 3, 4, 5],
        "all": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
    "ViT-B/32": {
        "bottom": [0, 1, 2, 3],
        "mid": [4, 5, 6, 7],
        "up": [8, 9, 10, 11],
        "half-up": [6, 7, 8, 9, 10, 11],
        "half-bottom": [0, 1, 2, 3, 4, 5],
        "all": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
    "ViT-L/14": {
        "half-up": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        "half-bottom": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "all": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
        ],
    },
}


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """
    Freeze all parameters except for LoRA modules and optionally bias parameters.

    Args:
        model (nn.Module): Model to modify.
        bias (str): Options for bias parameters ('none', 'all', 'lora_only').
    """
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = "none") -> dict[str, torch.Tensor]:
    """
    Extract the state dictionary for LoRA modules and optionally bias parameters.

    Args:
        model (nn.Module): Model containing LoRA layers.
        bias (str): Options for bias parameters ('none', 'all', 'lora_only').

    Returns:
        Dict[str, torch.Tensor]: State dictionary of selected parameters.
    """
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    elif bias == "all":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError(f"Unsupported bias option: {bias}")


def get_lora_parameters(model: nn.Module, bias: str = "none") -> list[torch.nn.Parameter]:
    """
    Retrieve trainable parameters associated with LoRA modules.

    Args:
        model (nn.Module): Model containing LoRA layers.
        bias (str): Options for bias parameters ('none', 'all', 'lora_only').

    Returns:
        List[torch.nn.Parameter]: List of parameters to train.
    """
    params = []
    for name, param in model.named_parameters():
        if bias == "none":
            if "lora_" in name:
                params.append(param)
        elif bias == "all":
            if "lora_" in name or "bias" in name:
                params.append(param)
        elif bias == "lora_only":
            if "lora_" in name:
                params.append(param)
                bias_name = name.split("lora_")[0] + "bias"
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError(f"Unsupported bias option: {bias}")
    return params


def apply_lora(args, clip_model: nn.Module) -> list[PlainMultiheadAttentionLoRA]:
    """
    Inject LoRA layers into the CLIP model based on the configuration in `args`.

    Args:
        args: Configuration arguments for LoRA.
        clip_model (nn.Module): CLIP model to modify.
    Returns:
        List[PlainMultiheadAttentionLoRA]: List of applied LoRA layers.
    """
    list_lora_layers = []
    if args.encoder in {"text", "both"}:
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule,
                            enable_lora=args.params,
                            r=args.r,
                            lora_alpha=args.alpha,
                            dropout_rate=args.dropout_rate,
                        )
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder in {"vision", "both"}:
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule,
                            enable_lora=args.params,
                            r=args.r,
                            lora_alpha=args.alpha,
                            dropout_rate=args.dropout_rate,
                        )
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers


def save_lora(args, list_lora_layers):
    """
    Save LoRA layer weights and metadata.

    Args:
        args: Argument namespace containing configuration details like paths and parameters.
        list_lora_layers: List of LoRA-modified layers from the model.
    """
    weights = {}

    # Iterate through all LoRA layers to save their weights
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}

        # Save weights for each specified parameter (q, k, v, o)
        if "q" in args.params:
            layer_weights["q_proj"] = {
                "w_lora_A": layer.q_proj.w_lora_A.data,
                "w_lora_B": layer.q_proj.w_lora_B.data,
            }
        if "k" in args.params:
            layer_weights["k_proj"] = {
                "w_lora_A": layer.k_proj.w_lora_A.data,
                "w_lora_B": layer.k_proj.w_lora_B.data,
            }
        if "v" in args.params:
            layer_weights["v_proj"] = {
                "w_lora_A": layer.v_proj.w_lora_A.data,
                "w_lora_B": layer.v_proj.w_lora_B.data,
            }
        if "o" in args.params:
            layer_weights["proj"] = {
                "w_lora_A": layer.proj.w_lora_A.data,
                "w_lora_B": layer.proj.w_lora_B.data,
            }

        weights[f"layer_{i}"] = layer_weights

    # Metadata containing training configuration
    metadata = {
        "r": args.r,
        "alpha": args.alpha,
        "encoder": args.encoder,
        "params": args.params,
        "position": args.position,
    }

    # Combine weights and metadata
    save_data = {"weights": weights, "metadata": metadata}

    # Prepare the save directory path based on arguments
    save_dir = f"{args.save_path}"
    os.makedirs(save_dir, exist_ok=True)

    # Save the data to a file
    save_path = f"{save_dir}/{args.filename}.pt"
    torch.save(save_data, save_path)
    print(f"LoRA weights saved to {save_path}")


def load_lora(args, list_lora_layers):
    """
    Load LoRA layer weights and metadata from a file.

    Args:
        args: Argument namespace containing configuration details like paths and parameters.
        list_lora_layers: List of LoRA-modified layers from the model to load weights into.
    """
    # Prepare the path to load weights
    load_path = f"{args.save_path}/{args.filename}.pt"

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"File {load_path} does not exist.")

    # Load weights and metadata from the file
    loaded_data = torch.load(load_path)
    metadata = loaded_data["metadata"]

    # Check if metadata matches the current training configuration
    if metadata["r"] != args.r:
        raise ValueError(f"r mismatch: expected {args.r}, found {metadata['r']}")
    if metadata["alpha"] != args.alpha:
        raise ValueError(f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
    if metadata["encoder"] != args.encoder:
        raise ValueError(f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
    if metadata["params"] != args.params:
        raise ValueError(f"Params mismatch: expected {args.params}, found {metadata['params']}")
    if metadata["position"] != args.position:
        raise ValueError(
            f"Position mismatch: expected {args.position}, found {metadata['position']}"
        )

    # Copy weights to the corresponding LoRA layers
    weights = loaded_data["weights"]
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f"layer_{i}"]
        if "q" in args.params and "q_proj" in layer_weights:
            layer.q_proj.w_lora_A.data.copy_(layer_weights["q_proj"]["w_lora_A"])
            layer.q_proj.w_lora_B.data.copy_(layer_weights["q_proj"]["w_lora_B"])
        if "k" in args.params and "k_proj" in layer_weights:
            layer.k_proj.w_lora_A.data.copy_(layer_weights["k_proj"]["w_lora_A"])
            layer.k_proj.w_lora_B.data.copy_(layer_weights["k_proj"]["w_lora_B"])
        if "v" in args.params and "v_proj" in layer_weights:
            layer.v_proj.w_lora_A.data.copy_(layer_weights["v_proj"]["w_lora_A"])
            layer.v_proj.w_lora_B.data.copy_(layer_weights["v_proj"]["w_lora_B"])
        if "o" in args.params and "proj" in layer_weights:
            layer.proj.w_lora_A.data.copy_(layer_weights["proj"]["w_lora_A"])
            layer.proj.w_lora_B.data.copy_(layer_weights["proj"]["w_lora_B"])

    print(f"LoRA weights loaded from {load_path}")


def cls_acc(output, target, topk=1):
    """
    Compute the classification accuracy.

    Args:
        output: Model output logits.
        target: Ground truth labels.
        topk: Consider top-k predictions for accuracy.

    Returns:
        Accuracy percentage.
    """
    # Get the top-k predictions
    pred = output.topk(topk, 1, True, True)[1].t()

    # Check if predictions match the target
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    acc = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())

    # Calculate percentage accuracy
    acc = 100 * acc / target.shape[0]
    return acc
