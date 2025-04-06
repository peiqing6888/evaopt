"""
Optimization engine core
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import torch
from ..utils.quantize import quantize_tensor
from evaopt_core import optimize_matrix_in_chunks, ChunkConfig
import sys

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
    block_sparse: Optional[BlockSparseConfig] = None

class Optimizer:
    """High-performance model optimizer"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig(model_type="default")
        self._setup_memory_pool()
    
    def _setup_memory_pool(self):
        """Setup memory management"""
        self.memory_limit = int(self.config.max_memory_gb * 1024 * 1024 * 1024)  # Convert to bytes
    
    def optimize_matrix_in_chunks(self, matrix: np.ndarray, chunk_size: int = 1024,
                                method: str = "truncated_svd", rank: Optional[int] = None,
                                tolerance: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize a matrix in chunks to support larger matrices with limited memory.
        
        Args:
            matrix: Input matrix to optimize
            chunk_size: Size of chunks to process
            method: Optimization method to use
                - "svd": Full SVD (best for small matrices)
                - "truncated_svd": Truncated SVD using Krylov subspace methods (faster than full SVD)
                - "randomized_svd": Randomized SVD algorithm (very fast, slightly less accurate)
                - "low_rank": Low-rank approximation using alternating least squares
                - "sparse": Sparse matrix optimization
                - "block_sparse": Block-sparse matrix optimization
                - "adaptive_low_rank": Automatically selects optimal rank based on error/compression trade-off
                - "sparse_pattern": Optimizes based on common sparse patterns in the matrix
                - "mixed_precision": Uses different precision levels for different matrix elements
                - "tensor_core_svd": SVD optimized for tensor core hardware acceleration
            rank: Target rank for low-rank methods (default: 10 or automatically determined)
            tolerance: Error tolerance (default: 1e-6)
        
        Returns:
            Dictionary containing:
            - optimized_matrix: The optimized matrix
            - processing_time: Total processing time in seconds
            - memory_used: Peak memory usage in bytes
        """
        if matrix.ndim != 2:
            raise ValueError("Input must be a 2D matrix")
        
        # Setup chunk config
        chunk_config = ChunkConfig(
            chunk_size=chunk_size,
            use_parallel=self.config.use_parallel,
            memory_limit=int(self.config.max_memory_gb * 1e9),  # Convert GB to bytes
            prefetch_size=2,
            use_simd=True
        )
        
        # Apply optimization using core library
        result = optimize_matrix_in_chunks(
            matrix.astype(np.float32),
            chunk_config,
            method,
            rank or self.config.matrix_rank,
            tolerance or self.config.matrix_tolerance
        )
        
        return result
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize PyTorch model"""
        if self.config.use_fp16:
            model = model.half()
        
        # Move to target device
        device = torch.device(self.config.device)
        model = model.to(device)
        
        # Collect all parameters
        tensors = {}
        for name, param in model.named_parameters():
            tensor = param.data.contiguous().cpu().float()
            tensors[name] = tensor.numpy()
        
        # Optimize tensors
        try:
            optimized_tensors = {}
            for name, tensor in tensors.items():
                result = self.optimize_matrix_in_chunks(
                    tensor,
                    method=self.config.matrix_method or "truncated_svd",
                    rank=self.config.matrix_rank,
                    tolerance=self.config.matrix_tolerance
                )
                optimized_tensors[name] = result["optimized_matrix"]
        
        except Exception as e:
            print(f"Error during optimization: {e}")
            raise
        
        # Update model parameters
        with torch.no_grad():
            for name, tensor in optimized_tensors.items():
                if name in dict(model.named_parameters()):
                    param = model.get_parameter(name)
                    tensor = torch.from_numpy(tensor)
                    if self.config.use_fp16:
                        tensor = tensor.half()
                    param.data.copy_(tensor.to(device))
        
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
        """Optimize tensors"""
        optimized = {}
        for name, tensor in tensors.items():
            result = self.optimize_matrix_in_chunks(
                tensor,
                method=self.config.matrix_method or "truncated_svd",
                rank=self.config.matrix_rank,
                tolerance=self.config.matrix_tolerance
            )
            optimized[name] = result["optimized_matrix"]
        return optimized
    
    def get_matrix_stats(self, matrix: np.ndarray, method: str,
                        rank: Optional[int] = None,
                        tolerance: Optional[float] = None,
                        block_sparse_config: Optional[BlockSparseConfig] = None) -> Dict[str, float]:
        """
        Get optimization statistics for a matrix without modifying it.
        
        Args:
            matrix: Input matrix to analyze
            method: Optimization method to analyze
                - "svd": Full SVD (best for small matrices)
                - "truncated_svd": Truncated SVD using Krylov subspace methods (faster than full SVD)
                - "randomized_svd": Randomized SVD algorithm (very fast, slightly less accurate)
                - "low_rank": Low-rank approximation using alternating least squares
                - "sparse": Sparse matrix optimization
                - "block_sparse": Block-sparse matrix optimization
                - "adaptive_low_rank": Automatically selects optimal rank based on error/compression trade-off
                - "sparse_pattern": Optimizes based on common sparse patterns in the matrix
                - "mixed_precision": Uses different precision levels for different matrix elements
                - "tensor_core_svd": SVD optimized for tensor core hardware acceleration
            rank: Target rank for low-rank methods (default: 10 or from config)
            tolerance: Error tolerance (default: 1e-6 or from config)
            block_sparse_config: Configuration for block-sparse optimization (if method is "block_sparse")
        
        Returns:
            Dictionary containing:
            - compression_ratio: Compression ratio in percentage (higher is better)
            - error: Relative error (lower is better)
            - storage_size: Storage size in bytes after optimization
            - original_size: Original storage size in bytes
        """
        # Handle block_sparse method which needs special config
        if method == "block_sparse" and block_sparse_config is not None:
            # This is a workaround for now - we apply the optimization to get stats
            # In the future, we should enhance the Rust API to provide a direct method
            config = ModelConfig(
                model_type="matrix",
                matrix_method=method,
                matrix_rank=rank or self.config.matrix_rank,
                matrix_tolerance=tolerance or self.config.matrix_tolerance,
                block_sparse=block_sparse_config
            )
            optimizer = Optimizer(config)
            optimized = optimizer.optimize({"matrix": matrix})
            
            # Calculate statistics manually
            original_size = matrix.nbytes
            compressed_size = sys.getsizeof(optimized["matrix"]) 
            compression_ratio = 100 * (1 - compressed_size / original_size)
            
            error = np.linalg.norm(matrix - optimized["matrix"]) / np.linalg.norm(matrix)
            
            return {
                "compression_ratio": compression_ratio,
                "error": error,
                "storage_size": compressed_size,
                "original_size": original_size
            }
        else:
            # Use the core library function for other methods
            return get_matrix_stats(
                matrix.astype(np.float32),
                method,
                rank or self.config.matrix_rank,
                tolerance or self.config.matrix_tolerance
            )

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
        result = optimize_matrix_in_chunks(tensors, config)
        
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