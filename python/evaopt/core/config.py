"""
Configuration classes for EvaOpt
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from .optimizer import BlockSparseConfig

class ModelType(Enum):
    """Supported model types"""
    TRANSFORMER = "transformer"
    FEED_FORWARD = "feed_forward"
    CUSTOM = "custom"

class MatrixMethod(Enum):
    """Supported matrix optimization methods"""
    SVD = "svd"
    LOW_RANK = "low_rank"
    BLOCK_SPARSE = "block_sparse"
    SPARSE = "sparse"

@dataclass
class DeviceConfig:
    """Device configuration"""
    device_type: str = "cpu"
    device_index: int = 0
    memory_limit: Optional[int] = None
    allow_tf32: bool = True
    allow_fp16: bool = False
    
    def __post_init__(self):
        if self.device_type not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"Unsupported device type: {self.device_type}")

@dataclass
class ModelConfig:
    """Model optimization configuration"""
    model_type: ModelType
    device: DeviceConfig = field(default_factory=DeviceConfig)
    matrix_method: Optional[MatrixMethod] = None
    matrix_rank: Optional[int] = None
    matrix_tolerance: Optional[float] = None
    block_sparse: Optional[BlockSparseConfig] = None
    use_fp16: bool = False
    sparsity_threshold: float = 1e-6
    quantization_bits: int = 8
    use_parallel: bool = True
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Convert string to enum if needed
        if isinstance(self.model_type, str):
            try:
                self.model_type = ModelType(self.model_type.lower())
            except ValueError:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        if isinstance(self.matrix_method, str):
            try:
                self.matrix_method = MatrixMethod(self.matrix_method.lower())
            except ValueError:
                raise ValueError(f"Unsupported matrix method: {self.matrix_method}")
        
        # Validate quantization bits
        if self.quantization_bits not in [4, 8]:
            raise ValueError("Quantization bits must be either 4 or 8")
            
        # Validate matrix rank if specified
        if self.matrix_rank is not None and self.matrix_rank <= 0:
            raise ValueError("Matrix rank must be positive")
            
        # Validate matrix tolerance if specified
        if self.matrix_tolerance is not None:
            if not (0 < self.matrix_tolerance < 1):
                raise ValueError("Matrix tolerance must be between 0 and 1")
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format"""
        return {
            "model_type": self.model_type.value,
            "device": {
                "type": self.device.device_type,
                "index": self.device.device_index,
                "memory_limit": self.device.memory_limit,
                "allow_tf32": self.device.allow_tf32,
                "allow_fp16": self.device.allow_fp16
            },
            "matrix_method": self.matrix_method.value if self.matrix_method else None,
            "matrix_rank": self.matrix_rank,
            "matrix_tolerance": self.matrix_tolerance,
            "block_sparse": self.block_sparse.to_dict() if self.block_sparse else None,
            "use_fp16": self.use_fp16,
            "sparsity_threshold": self.sparsity_threshold,
            "quantization_bits": self.quantization_bits,
            "use_parallel": self.use_parallel,
            "custom_config": self.custom_config
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary format"""
        device_config = DeviceConfig(**config_dict.pop("device", {}))
        if "block_sparse" in config_dict and config_dict["block_sparse"]:
            config_dict["block_sparse"] = BlockSparseConfig(**config_dict["block_sparse"])
        return cls(device=device_config, **config_dict) 