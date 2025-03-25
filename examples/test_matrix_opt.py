"""
Demo script for matrix optimization methods
"""

import numpy as np
import time
from evaopt import Optimizer, ModelConfig
import matplotlib.pyplot as plt

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

def visualize_matrix(matrix: np.ndarray, title: str) -> None:
    """Visualize matrix values distribution."""
    plt.figure(figsize=(6, 4))
    plt.hist(matrix.flatten(), bins=50, alpha=0.7)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.show()

def test_optimization(matrix: np.ndarray, method: str, rank: int) -> None:
    """Test matrix optimization with specified method and rank."""
    print(f"\n{'='*50}")
    print(f"Testing {method.upper()} optimization (rank={rank})")
    print(f"{'='*50}")
    
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
    
    # Print results
    print("\n📊 Results:")
    print(f"• Original size: {matrix.nbytes / 1024:.1f} KB")
    print(f"• Compressed size: {compressed.nbytes / 1024:.1f} KB")
    print(f"• Compression ratio: {stats['compression_ratio']:.1%}")
    print(f"• Error: {stats['error']:.6f}")
    print(f"• Processing time: {end_time - start_time:.3f}s")
    
    # Visualize original vs compressed
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(matrix[:50, :50], cmap='viridis')
    plt.title("Original")
    plt.colorbar()
    
    plt.subplot(132)
    plt.imshow(compressed[:50, :50], cmap='viridis')
    plt.title("Compressed")
    plt.colorbar()
    
    plt.subplot(133)
    plt.imshow(np.abs(matrix[:50, :50] - compressed[:50, :50]), cmap='viridis')
    plt.title("Difference")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def main():
    print("\n🚀 EvaOpt Matrix Optimization Demo")
    print("="*50)
    
    # Test case 1: Random large matrix
    print("\n📈 Testing large random matrix...")
    matrix = create_test_matrix(1000, 800)
    visualize_matrix(matrix, "Original Matrix Distribution")
    
    # Test different methods
    methods = ["svd", "truncated_svd", "randomized_svd"]
    ranks = [10, 50]
    
    for method in methods:
        for rank in ranks:
            test_optimization(matrix, method, rank)
    
    # Test case 2: Low-rank matrix
    print("\n📉 Testing structured low-rank matrix...")
    matrix = create_test_matrix(1000, 800, rank=30)
    visualize_matrix(matrix, "Low-rank Matrix Distribution")
    
    # Test best method for low-rank case
    test_optimization(matrix, "truncated_svd", 30)
    
    print("\n✨ Demo completed!")

if __name__ == "__main__":
    main() 