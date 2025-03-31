"""
Python implementation of the EvaOpt optimizer
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

from evaopt_core import optimize_tensors, get_matrix_stats as _get_matrix_stats
from ..utils.quantize import quantize_tensor

@dataclass
class BlockSparseConfig:
    """Block-sparse optimization configuration"""
    block_size: int = 32
    sparsity_threshold: float = 0.1
    min_block_norm: float = 0.01

@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str
    sparsity_threshold: float = 1e-6
    quantization_bits: int = 8
    use_parallel: bool = True
    matrix_method: Optional[str] = None
    matrix_rank: Optional[int] = None
    matrix_tolerance: Optional[float] = None
    use_fp16: bool = True
    max_memory_gb: float = 24.0
    device: str = "mps"
    block_sparse: Optional[BlockSparseConfig] = None  # Add block-sparse config

class Optimizer:
    """EvaOpt optimizer"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.device = torch.device(self.config.device)
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
            config_dict = {
                "sparsity_threshold": self.config.sparsity_threshold,
                "quantization_bits": self.config.quantization_bits,
                "use_parallel": self.config.use_parallel,
            }
            
            if self.config.matrix_method:
                config_dict.update({
                    "matrix_method": self.config.matrix_method,
                    "matrix_rank": self.config.matrix_rank or 10,
                    "matrix_tolerance": self.config.matrix_tolerance or 1e-6,
                })
                
                # Add block-sparse configuration if specified
                if self.config.matrix_method == "block_sparse" and self.config.block_sparse:
                    config_dict["block_sparse"] = {
                        "block_size": self.config.block_sparse.block_size,
                        "sparsity_threshold": self.config.block_sparse.sparsity_threshold,
                        "min_block_norm": self.config.block_sparse.min_block_norm,
                    }
            
            optimized_tensors = optimize_tensors(tensors, config_dict)
            
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

    def optimize(self, tensors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Optimize tensors using configured methods.
        
        Args:
            tensors: Dictionary of tensor names to numpy arrays
            
        Returns:
            Dictionary of optimized tensors
        """
        config_dict = {
            "sparsity_threshold": self.config.sparsity_threshold,
            "quantization_bits": self.config.quantization_bits,
            "use_parallel": self.config.use_parallel,
        }
        
        if self.config.matrix_method:
            config_dict.update({
                "matrix_method": self.config.matrix_method,
                "matrix_rank": self.config.matrix_rank or 10,
                "matrix_tolerance": self.config.matrix_tolerance or 1e-6,
            })
            
            # Add block-sparse configuration if specified
            if self.config.matrix_method == "block_sparse" and self.config.block_sparse:
                config_dict["block_sparse"] = {
                    "block_size": self.config.block_sparse.block_size,
                    "sparsity_threshold": self.config.block_sparse.sparsity_threshold,
                    "min_block_norm": self.config.block_sparse.min_block_norm,
                }
        
        return optimize_tensors(tensors, config_dict)
    
    def get_matrix_stats(self, matrix: np.ndarray, method: str, 
                        rank: Optional[int] = None,
                        tolerance: Optional[float] = None,
                        block_sparse_config: Optional[BlockSparseConfig] = None) -> Dict[str, float]:
        """Get statistics for matrix optimization.
        
        Args:
            matrix: Input matrix to analyze
            method: Optimization method ("svd", "low_rank", "sparse", "block_sparse")
            rank: Target rank for low-rank approximation
            tolerance: Error tolerance threshold
            block_sparse_config: Configuration for block-sparse optimization
            
        Returns:
            Dictionary containing optimization statistics
        """
        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
            raise ValueError("Input must be a 2D numpy array")
        
        # Convert method to lowercase for case-insensitive comparison
        method = method.lower()
        
        # Map lowercase method to correct case
        method_map = {
            "svd": "svd",
            "low_rank": "low_rank",
            "sparse": "sparse",
            "block_sparse": "block_sparse",
            "truncated_svd": "truncated_svd",
            "randomized_svd": "randomized_svd"
        }
        
        if method not in method_map:
            valid_methods = set(method_map.keys())
            raise ValueError(f"Invalid method. Must be one of: {valid_methods}")
        
        # Create config dict for block-sparse method
        config_dict = {
            "matrix_method": method_map[method],
            "matrix_rank": rank or 10,
            "matrix_tolerance": tolerance or 1e-6,
        }
        
        # Add block-sparse configuration if provided
        if method == "block_sparse" and block_sparse_config:
            config_dict["block_sparse"] = {
                "block_size": block_sparse_config.block_size,
                "sparsity_threshold": block_sparse_config.sparsity_threshold,
                "min_block_norm": block_sparse_config.min_block_norm,
            }
        
        # Call Rust core for optimization
        result = optimize_tensors({"input": matrix}, config_dict)
        compressed = result["input"]
        
        # Calculate statistics
        original_size = matrix.nbytes
        compressed_size = compressed.nbytes
        compression_ratio = 1.0 - (compressed_size / original_size)
        error = np.linalg.norm(matrix - compressed) / np.linalg.norm(matrix)
        
        return {
            "compression_ratio": compression_ratio,
            "error": error,
            "storage_size": compressed_size,
        }

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize optimization configuration."""
    default_config = {
        "method": "svd",
        "rank": 10,
        "tolerance": 1e-6,
        "use_parallel": True,
        "oversampling": 5,
        "power_iterations": 2
    }
    
    if not isinstance(config, dict):
        raise TypeError("Config must be a dictionary")
    
    # Update with user config
    final_config = default_config.copy()
    final_config.update(config)
    
    # Validate method
    valid_methods = {"svd", "truncated_svd", "randomized_svd", "low_rank", "sparse", "block_sparse"}
    if final_config["method"] not in valid_methods:
        raise ValueError(f"Invalid method. Must be one of: {valid_methods}")
    
    # Validate numeric parameters
    if not isinstance(final_config["rank"], int) or final_config["rank"] < 1:
        raise ValueError("Rank must be a positive integer")
    
    if not isinstance(final_config["tolerance"], (int, float)) or final_config["tolerance"] <= 0:
        raise ValueError("Tolerance must be a positive number")
    
    if not isinstance(final_config["oversampling"], int) or final_config["oversampling"] < 0:
        raise ValueError("Oversampling must be a non-negative integer")
    
    if not isinstance(final_config["power_iterations"], int) or final_config["power_iterations"] < 0:
        raise ValueError("Power iterations must be a non-negative integer")
    
    return final_config

def optimize_tensor(tensor: np.ndarray, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Optimize a tensor using various methods.
    
    Args:
        tensor: Input tensor as numpy array
        config: Configuration dictionary with the following options:
            - method: Optimization method ("SVD", "TruncatedSVD", "RandomizedSVD", "LowRank", "Sparse")
            - rank: Target rank for low-rank approximation
            - tolerance: Error tolerance
            - use_parallel: Whether to use parallel processing
            - oversampling: Number of extra dimensions for randomized methods
            - power_iterations: Number of power iterations for randomized methods
    
    Returns:
        Dictionary containing:
            - compressed: Compressed tensor
            - compression_ratio: Achieved compression ratio
            - error: Reconstruction error
            - storage_size: Actual storage size in bytes
    """
    if config is None:
        config = {}
    
    # Input validation
    if not isinstance(tensor, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    if tensor.dtype != np.float32:
        tensor = tensor.astype(np.float32)
    
    # Validate and normalize config
    config = validate_config(config)
    
    try:
        # Call Rust optimization function with single tensor
        tensors = {"input": tensor}
        result = optimize_tensors(tensors, config)
        
        # Extract result for single tensor
        compressed_result = {
            "compressed": result["input"],
            "compression_ratio": 0.0,  # We'll need to calculate this
            "error": 0.0,  # We'll need to calculate this
            "storage_size": result["input"].nbytes
        }
        
        # Post-process result if needed
        if config.get("quantize", False):
            compressed_result["compressed"] = quantize_tensor(
                compressed_result["compressed"],
                config.get("quantize_bits", 8)
            )
        
        return compressed_result
    
    except Exception as e:
        raise RuntimeError(f"Optimization failed: {str(e)}") 