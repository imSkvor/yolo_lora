from typing import Optional
import torch.nn as nn
from lora_models.conv_lora import CoLoRALayer

def apply_colora_to_model(
    model: nn.Module,
    target_layers: Optional[list[str]] = None,
    rank: int = 4,
) -> tuple[int, list[str]]:
    """
    Recursively replace Conv2d layers with CoLoRALayer
    
    Args:
        model: YOLOv9 model to modify
        target_layers: list of layer name patterns to target, e.g. ['backbone', 'neck']
                       if None, targets all Conv2s layers expect detection head
        rank: CoLoRA rank
    
    Returns:
        tuple of (number_of_layers_changed, list_of_converted_layer_names)
    """

    if target_layers is None:
        target_layers = ["backbone", "neck"]
    
    change_count: int = 0
    converted_names: list[str] = []

    def _process_module(module: nn.Module, module_name: str = "", prefix: str = "") -> None:
        nonlocal change_count, converted_names

        for name, child in module.named_children():
            full_name: str = f"{prefix}.{name}" if prefix else name

            if len(list(child.children())) > 0:
                _process_module(child, name, full_name)
            
            elif isinstance(child, nn.Conv2d):
                should_convert: bool = False
                if target_layers:
                    for target in target_layers:
                        if target in full_name.lower():
                            should_convert = True
                            break
                else:
                    should_convert = True

                # Skip detection head layers
                if 'detect' in full_name.lower() or 'head' in full_name.lower():
                    should_convert = False
                
                if should_convert:
                    col_layer = CoLoRALayer(child, rank = rank)
                    setattr(module, name, col_layer)
                    change_count += 1
                    converted_names.append(full_name)
    
    _process_module(model)
    return change_count, converted_names

def get_trainable_parameters(model: nn.Module) -> tuple[int, int, float]:
    """Return a tuple (total_params, trainable_params, trainable_percentage)"""
    total_params: int = sum(p.numel() for p in model.parameters())
    trainable_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params, (100.0 * trainable_params / total_params if total_params > 0 else 0.0)

def freeze_model_except_colora(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "lora_pointwise" in name or "lora_layerwise" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
