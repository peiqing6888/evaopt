"""
模型量化工具
"""

import torch
import numpy as np
from typing import Any, Optional

def quantize(model: Any, bits: int = 8, scheme: str = "symmetric") -> Any:
    """
    將模型參數量化為指定位數
    
    Args:
        model: PyTorch 模型
        bits: 量化位數 (4-8)
        scheme: 量化方案 ("symmetric" 或 "asymmetric")
    
    Returns:
        量化後的模型
    """
    if bits not in [4, 8]:
        raise ValueError("僅支持 4 位和 8 位量化")
    
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
    
    # 遍歷所有參數進行量化
    for name, param in model.named_parameters():
        if param.requires_grad:  # 只量化可訓練參數
            with torch.no_grad():
                param.data.copy_(_quantize_tensor(param.data))
    
    return model

def optimize_memory(
    model: Any,
    max_memory: float,
    device: str = "mps"
) -> None:
    """
    優化模型內存使用
    
    Args:
        model: PyTorch 模型
        max_memory: 最大內存使用量（GB）
        device: 計算設備
    """
    if device == "mps":
        torch.mps.empty_cache()
        
    # 使用 CPU 卸載優化
    if hasattr(model, "cpu_offload"):
        model.cpu_offload()
    
    # 設置梯度檢查點
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # 清理未使用的緩存
    torch.cuda.empty_cache() if torch.cuda.is_available() else None 