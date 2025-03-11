"""
Model quantization utilities
"""

import torch
import numpy as np
from typing import Any, Optional

def quantize(model: Any, bits: int = 8, scheme: str = "symmetric") -> Any:
    """
    Quantize model parameters to specified bit width
    
    Args:
        model: PyTorch model
        bits: Quantization bit width (4-8)
        scheme: Quantization scheme ("symmetric" or "asymmetric")
    
    Returns:
        Quantized model
    """
    if bits not in [4, 8]:
        raise ValueError("Only 4-bit and 8-bit quantization supported")
    
    def _quantize_tensor(x: torch.Tensor) -> torch.Tensor:
        if scheme == "symmetric":
            max_val = torch.max(torch.abs(x))
            scale = (2 ** (bits - 1) - 1) / max_val
            return torch.round(x * scale) / scale
        else:  # asymmetric
            min_val = torch.min(x)
            max_val = torch.max(x)
            scale = (2 ** bits - 1) / (max_val - min_val)
            zero_point = torch.round(-min_val * scale)
            return (torch.round(x * scale + zero_point) - zero_point) / scale
    
    # Iterate through all parameters for quantization
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only quantize trainable parameters
            with torch.no_grad():
                param.data.copy_(_quantize_tensor(param.data))
    
    return model

def optimize_memory(
    model: Any,
    max_memory: float,
    device: str = "mps"
) -> None:
    """
    Optimize model memory usage
    
    Args:
        model: PyTorch model
        max_memory: Maximum memory usage (GB)
        device: Compute device
    """
    if device == "mps":
        torch.mps.empty_cache()
        
    # Use CPU offload optimization
    if hasattr(model, "cpu_offload"):
        model.cpu_offload()
    
    # Enable gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Clear unused cache
    torch.cuda.empty_cache() if torch.cuda.is_available() else None 