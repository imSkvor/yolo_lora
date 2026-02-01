from models.conv_lora import CoLoRALayer
from typing import Any
import torch
import torch.nn as nn
from utils.model_surgery import apply_colora_to_model, get_trainable_parameters

def setup_colora_training(
    base_model: nn.Module,
    config: dict[str, Any]
) -> tuple[nn.Module, torch.optim.Optimizer, dict[str, int]]:
    """
    Sets up CoLoRA model, optimizer and return parameter counts.
    Returns:
        (model, optimizer, param_stats_dict)
    """

    num_converted, converted_names = apply_colora_to_model(
        base_model,
        target_layers=config.get("target_layers", ["backbone", "neck"]),
        rank = config.get("lora_rank", 4)
    )

    print(f"Converted {num_converted} layers to CoLoRA")
    if config.get("verbose", False):
        print("Converted layers:", converted_names)

    total_params, trainable_params, trainable_pct = get_trainable_parameters(base_model)

    param_stats = {
        "total": total_params,
        "trainable": trainable_params,
        "trainable_percentage": trainable_pct,
        "converted_layers": num_converted
    }

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, base_model.parameters()),
        lr = config.get("lora_lr", 1e-3),
        weight_decay = config.get("weight_decay", 0.01)
    )

    return base_model, optimizer, param_stats

def validate_colora_integration(model: nn.Module) -> bool:
    has_colora_layers: bool = any(
        isinstance(module, CoLoRALayer)
        for module in model.modules()
    )

    all_frozen: bool = True
    for name, param in model.named_parameters():
        if "original_conv" in name and param.requires_grad:
            all_frozen = False
            print(f"Warning: original conv weight {name} is trainable!")

    return has_colora_layers and all_frozen
