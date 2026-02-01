import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Optional

class CoLoRALayer(nn.Module):
    r"""
    CoLoRA decomposes weight update into pointwise (1 x 1 x C) and layerwise (h x w x 1) components.
    W = W_original + \delta W, where \delta W = pointwise_conv \circ layerwise_conv.
    It finetunes by learning low-rank \delta W convolution kernels.
    """

    def __init__(self, original_conv: nn.Conv2d, rank: int = 4, alpha: int = 8) -> None:
        super().__init__()
        self.original_conv = original_conv
        self.rank = rank
        self.alpha = alpha

        for param in self.original_conv.parameters():
            param.requires_grad = False # freezeing original weights
        
        in_channels = original_conv.in_channels
        out_channels = original_conv.out_channels
        kernel_size = original_conv.kernel_size
        groups = original_conv.groups

        # CoLoRA decomposition
        # pointwise 1x1
        self.lora_pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size = 1, stride = 1, padding = 0,
            groups = groups, bias = False
        )

        if isinstance(kernel_size, tuple):
            kernel_size_tuple = kernel_size
        else:
            kernel_size_tuple = (kernel_size, kernel_size)

        # layerwise
        self.lora_layerwise = nn.Conv2d(
            out_channels, out_channels,
            kernel_size = kernel_size_tuple,
            stride = original_conv.stride,
            padding = original_conv.padding,
            groups = out_channels,
            bias = False
        )

        nn.init.kaiming_uniform_(self.lora_pointwise.weight, a = math.sqrt(5)) # <- check this against more articles?
        nn.init.zeros_(self.lora_layerwise.weight)

        self.scaling: float = alpha / rank


    def forward(self, x: Tensor) -> Tensor:
        original_output: Tensor = self.original_conv(x)
        lora_output: Tensor = self.lora_layerwise(self.lora_pointwise(x))
        return original_output + self.scaling * lora_output


    def merge_into_conv(self) -> nn.Conv2d:
        """
        Merges CoLoRA parameters back into a standard Conv2d layer for inference
        Returns:
            nn.Conv2d: a new Conv2d layer
        """
        with torch.no_grad():
            pw_expanded: Tensor = self.lora_pointwise.weight
            if self.lora_layerwise.groups == self.lora_layerwise.out_channels:
                lw_expanded: Tensor = self.lora_layerwise.weight.repeat(
                    1, self.original_conv.in_channels // self.lora_layerwise.groups, 1, 1
                )
            else:
                lw_expanded = self.lora_layerwise.weight
            
            lora_weight: Tensor = lw_expanded * pw_expanded
            lora_weight *= self.scaling

            merged_weight: Tensor = self.original_conv.weight + lora_weight
            merged_bias: Optional[Tensor] = None

            if self.original_conv.bias is not None:
                merged_bias = self.original_conv.bias.clone()

            merged_conv = nn.Conv2d(
                in_channels = self.original_conv.in_channels,
                out_channels = self.original_conv.out_channels,
                kernel_size = self.original_conv.kernel_size,
                stride = self.original_conv.stride,
                padding = self.original_conv.padding,
                dilation = self.original_conv.dilation,
                groups = self.original_conv.groups,
                bias = self.original_conv.bias is not None,
                padding_mode = self.original_conv.padding_mode,
                device = self.original_conv.weight.device,
                dtype = self.original_conv.weight.dtype
            )

            merged_conv.weight.data.copy_(merged_weight)
            if merged_bias is not None:
                merged_conv.bias.data.copy_(merged_bias)

            return merged_conv


    @property
    def weight(self) -> Tensor:
        with torch.no_grad():
            pw_weight: Tensor = self.lora_pointwise.weight # [out_channels, in_channels, 1, 1]
            lw_weight: Tensor = self.lora_layerwise.weight # [out_channels, 1, kernel_h, kernel_w]

            combined: Tensor = self.scaling * lw_weight * pw_weight
            return self.original_conv.weight + combined


    def extra_repr(self) -> str:
        return f"{self.rank=}, {self.alpha=}, {self.original_conv=}"
