"""
Test matrix optimization methods with different matrix types
"""

import numpy as np
import time
from evaopt import Optimizer, ModelConfig

def create_test_matrix(n: int, m: int, rank: int = None) -> np.ndarray:
    """Create a test matrix with optional low-rank structure."""
    if rank is None:
        # Random matrix
        return np.random.randn(n, m).astype(np.float32)
    else:
        # Low-rank matrix with noise
        u = np.random.randn(n, rank).astype(np.float32)
        v = np.random.randn(m, rank).astype(np.float32)
        base = u @ v.T
        noise = np.random.randn(n, m).astype(np.float32) * 0.1
        return base + noise

def test_optimization(matrix: np.ndarray, method: str, rank: int) -> None:
    """Test matrix optimization with specified method and rank."""
    config = ModelConfig(
        model_type="test",
        matrix_method=method,
        matrix_rank=rank,
        matrix_tolerance=1e-6,
        use_parallel=True
    )
    
    optimizer = Optimizer(config)
    
    # Time the optimization
    start_time = time.time()
    result = optimizer.optimize({"test": matrix})
    end_time = time.time()
    
    compressed = result["test"]
    stats = optimizer.get_matrix_stats(matrix, method, rank)
    
    print(f"\nMethod: {method}, Rank: {rank}")
    print(f"Original shape: {matrix.shape}")
    print(f"Compressed shape: {compressed.shape}")
    print(f"Compression ratio: {stats['compression_ratio']:.2%}")
    print(f"Error: {stats['error']:.6f}")
    print(f"Storage size: {stats['storage_size'] / 1024:.2f} KB")
    print(f"Time: {end_time - start_time:.3f}s")
    
    # Verify reconstruction
    rel_error = np.linalg.norm(matrix - compressed) / np.linalg.norm(matrix)
    print(f"Relative reconstruction error: {rel_error:.6f}")

def main():
    # Test different matrix types and sizes
    test_cases = [
        # (rows, cols, true_rank, test_ranks)
        (1000, 800, None, [10, 20, 50]),  # Random matrix
        (1000, 800, 30, [10, 30, 50]),    # Low-rank matrix
        (2000, 100, 10, [5, 10, 20]),     # Tall matrix
        (100, 2000, 10, [5, 10, 20]),     # Wide matrix
    ]
    
    methods = [
        "svd",
        "truncated_svd",
        "randomized_svd",
        "low_rank",
        "sparse"
    ]
    
    for rows, cols, true_rank, test_ranks in test_cases:
        print(f"\n{'='*80}")
        print(f"Testing {rows}x{cols} matrix", end="")
        if true_rank:
            print(f" with true rank {true_rank}")
        else:
            print(" (random)")
            
        matrix = create_test_matrix(rows, cols, true_rank)
        print(f"Original size: {matrix.nbytes / 1024:.2f} KB")
        
        for method in methods:
            if method == "sparse":
                # For sparse method, only test once with default settings
                test_optimization(matrix, method, 10)
            else:
                # For other methods, test with different ranks
                for rank in test_ranks:
                    test_optimization(matrix, method, rank)

if __name__ == "__main__":
    main() 