"""
Configuration classes for EvaOpt
"""

from dataclasses import dataclass
from typing import Optional
from .optimizer import BlockSparseConfig

@dataclass
class ModelConfig:
    """Model optimization configuration"""
    model_type: str
    device: str = "cpu"
    matrix_method: Optional[str] = None
    matrix_rank: Optional[int] = None
    matrix_tolerance: Optional[float] = None
    block_sparse: Optional[BlockSparseConfig] = None
    use_fp16: bool = False
    sparsity_threshold: float = 1e-6
    quantization_bits: int = 8
    use_parallel: bool = True 