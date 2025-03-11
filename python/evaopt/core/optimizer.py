"""
Python implementation of the EvaOpt optimizer
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

from evaopt_core import optimize_tensors

@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str
    quantization_bits: int = 8
    use_fp16: bool = True
    max_memory_gb: float = 24.0
    device: str = "mps"

class Optimizer:
    """EvaOpt optimizer"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._setup_memory_pool()
    
    def _setup_memory_pool(self):
        """Configure memory pool"""
        if self.config.device == "mps":
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(
                self.config.max_memory_gb / 32.0
            )
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model"""
        # 1. Convert data type
        if self.config.use_fp16:
            model = model.half()
        
        # 2. Move to target device
        model = model.to(self.device)
        
        # 3. Collect all parameters
        tensors = {}
        for name, param in model.named_parameters():
            # Ensure tensor is contiguous and convert to float32
            tensor = param.data.contiguous().cpu().float()
            tensors[name] = tensor.numpy()
        
        # 4. Call Rust core for optimization
        try:
            optimized_tensors = optimize_tensors(tensors)
        except Exception as e:
            print(f"Tensor states during optimization:")
            for name, tensor in tensors.items():
                print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")
            raise e
        
        # 5. Update model parameters
        with torch.no_grad():
            for name, tensor in optimized_tensors.items():
                if name in dict(model.named_parameters()):
                    param = model.get_parameter(name)
                    tensor = torch.from_numpy(tensor)
                    if self.config.use_fp16:
                        tensor = tensor.half()
                    param.data.copy_(tensor.to(self.device))
        
        return model
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        if self.config.device == "mps":
            return {
                "allocated": torch.mps.current_allocated_memory() / 1024**3,
                "reserved": torch.mps.driver_allocated_memory() / 1024**3,
                "max_memory": self.config.max_memory_gb
            }
        return {} 