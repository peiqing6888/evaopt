"""
Test script for block-sparse optimization
"""

import numpy as np
import torch
import time
from evaopt.core.optimizer import Optimizer, ModelConfig, BlockSparseConfig
import matplotlib.pyplot as plt
from evaopt.utils.visualize import EVAVisualizer

def create_structured_matrix(size):
    """Create a structured sparse matrix with highly varying block norms.
    The scale of norms decreases exponentially with distance from diagonal.
    """
    matrix = np.zeros((size, size), dtype=np.float32)
    
    # Create diagonal blocks with exponentially decreasing norms
    for i in range(0, size, 16):
        block_size = min(16, size - i)
        # Much faster decay for diagonal blocks
        scale = np.exp(-i/(size/16))  # Faster decay
        block = np.random.randn(block_size, block_size).astype(np.float32) * scale
        matrix[i:i+block_size, i:i+block_size] = block
    
    # Create off-diagonal blocks with much smaller values and faster decay
    for i in range(0, size, 16):
        for j in range(0, size, 16):
            if i != j and np.random.random() < 0.05:  # Reduce probability to 5%
                block_size = min(16, min(size - i, size - j))
                # Much smaller scale for off-diagonal blocks with faster decay
                scale = 0.0001 * np.exp(-(i+j)/(size/4))  # 10000x smaller and faster decay
                block = np.random.randn(block_size, block_size).astype(np.float32) * scale
                matrix[i:i+block_size, j:j+block_size] = block
    
    return matrix

def test_optimization():
    """Test block-sparse optimization with different block sizes."""
    print("[INFO] Starting Block-sparse Optimization Test")
    print("[INFO] Creating 1000x1000 test matrix...")
    
    # Create test matrix
    matrix = create_structured_matrix(1000)
    
    # Configure block-sparse optimization
    block_sizes = [16, 32, 64, 128]
    
    for block_size in block_sizes:
        print(f"\nTesting block size: {block_size}\n")
        
        # Configure optimization with more aggressive thresholds
        block_sparse_config = BlockSparseConfig(
            block_size=block_size,
            sparsity_threshold=0.2,  # More aggressive sparsity threshold
            min_block_norm=0.001  # Lower min_block_norm to catch more blocks
        )
        
        # Create model configuration
        model_config = ModelConfig(
            model_type="matrix",
            device="cpu",
            matrix_method="block_sparse",
            matrix_rank=10,
            matrix_tolerance=1e-6,
            block_sparse=block_sparse_config
        )
        
        # Create optimizer
        optimizer = Optimizer(model_config)
        
        # Print debug information
        print("Debug information:")
        print(f"Matrix shape: {matrix.shape}")
        print(f"Matrix method: block_sparse")
        print(f"Block size: {block_size}")
        print(f"Rank: 10")
        print(f"Tolerance: 1e-06")
        
        # Get matrix stats
        stats = optimizer.get_matrix_stats(matrix, "block_sparse", block_sparse_config=block_sparse_config)
        print("Matrix stats obtained successfully\n")
        
        # Time the optimization
        start_time = time.time()
        result = optimizer.optimize({"input": matrix})
        end_time = time.time()
        
        # Print results
        print(f"Results for block size {block_size}:")
        print(f"• Compression ratio: {stats['compression_ratio']:.1f}%")
        print(f"• Error: {stats['error']:.6f}")
        print(f"• Processing time: {end_time - start_time:.3f}s\n")
        
        # Print compression analysis
        print(f"Matrix Optimization Results for Block Size {block_size}\n")
        print("Compression Analysis:")
        original_nnz = np.count_nonzero(matrix)
        optimized_nnz = np.count_nonzero(result["input"])
        compression_ratio = (original_nnz - optimized_nnz) / original_nnz * 100
        print(f"Original Non-zero Elements: {original_nnz}")
        print(f"Optimized Non-zero Elements: {optimized_nnz}")
        print(f"Compression Ratio: {compression_ratio:.2f}%")
        print(f"Processing Time: {end_time - start_time:.3f}s\n")
        
        # Visualize matrix patterns
        print("Matrix Pattern Visualization:\n")
        print("Original Matrix Pattern:")
        visualize_matrix_pattern(matrix)
        print("\n\nOptimized Matrix Pattern:")
        visualize_matrix_pattern(result["input"])
        print("\n\n")

def visualize_matrix_pattern(matrix, threshold=1e-6):
    """Visualize matrix sparsity pattern using ASCII art."""
    pattern = np.abs(matrix) > threshold
    rows, cols = pattern.shape
    step = max(1, min(rows, cols) // 50)  # Downsample for visualization
    
    for i in range(0, rows, step):
        line = ""
        for j in range(0, cols, step):
            block = pattern[i:i+step, j:j+step]
            density = np.mean(block)
            if density > 0.6:
                line += "█"
            elif density > 0.3:
                line += "░"
            else:
                line += " "
        print(line)

if __name__ == "__main__":
    test_optimization()
    print("[SUCCESS] Test completed!") 