"""
Model quantization utilities
"""

import torch
import numpy as np
from typing import Any, Optional

def quantize_tensor(tensor: np.ndarray, bits: int = 8, scheme: str = "symmetric") -> np.ndarray:
    """
    Quantize tensor to specified bit width
    
    Args:
        tensor: Input tensor as numpy array
        bits: Quantization bit width (4-8)
        scheme: Quantization scheme ("symmetric" or "asymmetric")
    
    Returns:
        Quantized tensor
    """
    if bits not in [4, 8]:
        raise ValueError("Only 4-bit and 8-bit quantization supported")
    
    if scheme == "symmetric":
        max_val = np.max(np.abs(tensor))
        scale = (2 ** (bits - 1) - 1) / max_val
        return np.round(tensor * scale) / scale
    else:  # asymmetric
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        scale = (2 ** bits - 1) / (max_val - min_val)
        zero_point = np.round(-min_val * scale)
        return (np.round(tensor * scale + zero_point) - zero_point) / scale

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

def quantize_tensor(tensor: np.ndarray, bits: int = 8) -> np.ndarray:
    """
    Quantize a tensor to reduced precision
    
    Args:
        tensor: Input tensor to quantize
        bits: Number of bits for quantization (1-8)
        
    Returns:
        Quantized tensor
    """
    if not isinstance(tensor, (np.ndarray, torch.Tensor)):
        raise TypeError("Input must be numpy array or torch tensor")
        
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
        
    if bits < 1 or bits > 8:
        raise ValueError("bits must be between 1 and 8")
        
    # Get tensor range
    min_val = tensor.min()
    max_val = tensor.max()
    
    # Calculate scaling factor
    scale = (max_val - min_val) / (2**bits - 1)
    
    # Quantize
    quantized = np.round((tensor - min_val) / scale)
    
    # Clip to valid range
    quantized = np.clip(quantized, 0, 2**bits - 1)
    
    # Scale back
    dequantized = quantized * scale + min_val
    
    return dequantized.astype(tensor.dtype) 