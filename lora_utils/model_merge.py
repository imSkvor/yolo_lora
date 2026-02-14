from typing import Optional
import copy
import torch.nn as nn
from lora_models.conv_lora import CoLoRALayer


def merge_colora_layer(layer: nn.Module) -> Optional[nn.Conv2d]:
    if isinstance(layer, CoLoRALayer):
        return layer.merge_into_conv()
    return None


def merge_colora_model(
    model: nn.Module,
    inplace: bool = False,
    verbose: bool = False
) -> nn.Module:
    """Recursively merges all CoLoRALayers in a model into standzsard Conv2d layers"""

    model_to_modify = model if inplace else copy.deepcopy(model)
    merge_count: int = 0

    def _merge_module(module: nn.Module, module_name: str = "", prefix: str = "") -> None:
        nonlocal merge_count
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if len(list(child.children())) > 0:
                _merge_module(child, name, full_name)

            if isinstance(child, CoLoRALayer):
                merged_conv = child.merge_into_conv()
                if merged_conv is not None:
                    setattr(module, name, merged_conv)
                    merge_count += 1
                    if verbose:
                        print(f"Merged CoLoRA layer: {full_name}")
                    
    _merge_module(model_to_modify)

    if verbose:
        print(f"Total CoLoRA layers merged: {merge_count}")

    return model_to_modify


def create_inference_model(
    model: nn.Module,
    merge_lora: bool = True,
    freeze_weights: bool = True
) -> nn.Module:
    """Prepares model for inference by merging CoLoRA and freezing weights"""

    inference_model = copy.deepcopy(model)

    if merge_lora:
        inference_model = merge_colora_model(inference_model, inplace=True)

    if freeze_weights:
        for param in inference_model.parameters():
            param.requires_grad = False

    inference_model.eval()
    return inference_model
