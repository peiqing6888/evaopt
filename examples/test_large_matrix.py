"""
Test script for large matrix optimization using chunk-based processing
"""

import numpy as np
import torch
import time
from evaopt import Optimizer, ModelConfig
from evaopt.utils.visualize import EVAVisualizer

def create_large_matrix(rows: int, cols: int, sparsity: float = 0.9) -> np.ndarray:
    """Create a large sparse matrix for testing."""
    matrix = np.random.randn(rows, cols).astype(np.float32)
    mask = np.random.random(matrix.shape) > sparsity
    return matrix * mask

def test_chunk_optimization():
    """Test chunk-based matrix optimization."""
    print("\n🚀 Testing Large Matrix Optimization")
    print("="*50)
    
    # Create test matrix (simulating a large model layer)
    rows = 10000  # Simulate part of a large model
    cols = 5000
    print(f"\nCreating test matrix ({rows}x{cols})...")
    matrix = create_large_matrix(rows, cols)
    
    # Initialize visualizer
    visualizer = EVAVisualizer()
    visualizer.print_status("Matrix created successfully", "info")
    
    # Configure optimization
    config = ModelConfig(
        model_type="matrix",
        matrix_method="truncated_svd",
        matrix_rank=100,
        matrix_tolerance=1e-6,
        use_parallel=True
    )
    
    # Create optimizer
    optimizer = Optimizer(config)
    
    # Measure baseline memory
    baseline_memory = matrix.nbytes / (1024**3)  # GB
    visualizer.print_status(f"Baseline memory usage: {baseline_memory:.2f} GB", "info")
    
    # Time the optimization
    visualizer.print_status("Starting optimization...", "info")
    start_time = time.time()
    
    try:
        # Optimize matrix
        result = optimizer.optimize_matrix_in_chunks(
            matrix,
            chunk_size=1024,
            method="truncated_svd",
            rank=100
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Get optimization statistics
        optimized_matrix = result["optimized_matrix"]
        compression_ratio = 1.0 - (np.count_nonzero(optimized_matrix) / np.count_nonzero(matrix))
        memory_usage = optimized_matrix.nbytes / (1024**3)  # GB
        error = np.linalg.norm(matrix - optimized_matrix) / np.linalg.norm(matrix)
        
        # Print results
        print("\n📊 Optimization Results:")
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Compression ratio: {compression_ratio:.2%}")
        print(f"Memory usage: {memory_usage:.2f} GB")
        print(f"Relative error: {error:.6f}")
        
        # Visualize results
        visualizer.plot_matrix_optimization(
            matrix[:50, :50],  # Show a small section
            optimized_matrix[:50, :50],
            "Large Matrix Optimization"
        )
        
    except Exception as e:
        visualizer.print_status(f"Error during optimization: {str(e)}", "error")
        raise

def main():
    try:
        test_chunk_optimization()
        print("\n✨ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")

if __name__ == "__main__":
    main() 