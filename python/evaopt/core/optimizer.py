"""
EvaOpt 優化器的 Python 實現
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

from evaopt_core import optimize_tensors

@dataclass
class ModelConfig:
    """模型配置"""
    model_type: str
    quantization_bits: int = 8
    use_fp16: bool = True
    max_memory_gb: float = 24.0
    device: str = "mps"

class Optimizer:
    """EvaOpt 優化器"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._setup_memory_pool()
    
    def _setup_memory_pool(self):
        """配置內存池"""
        if self.config.device == "mps":
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(
                self.config.max_memory_gb / 32.0
            )
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """優化模型"""
        # 1. 轉換數據類型
        if self.config.use_fp16:
            model = model.half()
        
        # 2. 移動到目標設備
        model = model.to(self.device)
        
        # 3. 收集所有參數
        tensors = {}
        for name, param in model.named_parameters():
            # 確保張量是連續的並轉換為 float32
            tensor = param.data.contiguous().cpu().float()
            tensors[name] = tensor.numpy()
        
        # 4. 調用 Rust 核心進行優化
        try:
            optimized_tensors = optimize_tensors(tensors)
        except Exception as e:
            print(f"優化過程中的張量狀態:")
            for name, tensor in tensors.items():
                print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")
            raise e
        
        # 5. 更新模型參數
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
        """獲取內存使用統計"""
        if self.config.device == "mps":
            return {
                "allocated": torch.mps.current_allocated_memory() / 1024**3,
                "reserved": torch.mps.driver_allocated_memory() / 1024**3,
                "max_memory": self.config.max_memory_gb
            }
        return {} 