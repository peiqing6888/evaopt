"""
Advanced optimization methods demonstration for EvaOpt
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from evaopt import Optimizer, ModelConfig, BlockSparseConfig
from evaopt_core import get_available_methods, get_matrix_stats

def create_test_matrix(n: int, m: int, rank: int = None, sparsity: float = 0.0) -> np.ndarray:
    """Create a test matrix with optional low-rank structure and sparsity."""
    if rank is None:
        # Random matrix
        matrix = np.random.randn(n, m).astype(np.float32)
    else:
        # Low-rank matrix with noise
        u = np.random.randn(n, rank).astype(np.float32)
        v = np.random.randn(m, rank).astype(np.float32)
        matrix = u @ v.T
        noise = np.random.randn(n, m).astype(np.float32) * 0.1
        matrix = matrix + noise
    
    # Apply sparsity
    if sparsity > 0:
        mask = np.random.random(matrix.shape) < sparsity
        matrix[mask] = 0
    
    return matrix

def print_available_methods():
    """Print information about all available optimization methods."""
    methods = get_available_methods()
    
    print("\n✨ Available Optimization Methods:")
    print("=" * 80)
    
    for name, info in methods.items():
        print(f"\n📊 {name.upper()}")
        print(f"Description: {info['description']}")
        print(f"Best for:    {info['best_for']}")
        
        print("\nParameters:")
        for param, desc in info['parameters'].items():
            print(f"  • {param}: {desc}")
        
        print("-" * 80)

def compare_methods(matrix: np.ndarray, methods: list, ranks: list = None, plot_results: bool = True):
    """Compare multiple optimization methods on the same matrix."""
    if ranks is None:
        ranks = [min(matrix.shape) // 10]  # Default rank is 10% of minimum dimension
    
    results = []
    
    print("\n📈 Optimization Method Comparison:")
    print("=" * 80)
    
    for method in methods:
        method_results = []
        
        for rank in ranks:
            # Get matrix stats for this method and rank
            stats = get_matrix_stats(matrix, method, rank=rank)
            
            # Time the optimization
            start_time = time.time()
            config = ModelConfig(
                model_type="test",
                matrix_method=method,
                matrix_rank=rank,
                matrix_tolerance=1e-6,
                use_parallel=True
            )
            optimizer = Optimizer(config)
            optimized = optimizer.optimize({"test": matrix})["test"]
            end_time = time.time()
            
            # Calculate actual error
            error = np.linalg.norm(matrix - optimized) / np.linalg.norm(matrix)
            
            # Store results
            result = {
                "method": method,
                "rank": rank,
                "compression_ratio": stats["compression_ratio"],
                "error": stats["error"],
                "measured_error": error,
                "time": end_time - start_time
            }
            method_results.append(result)
            
            print(f"\n• Method: {method}, Rank: {rank}")
            print(f"  Compression: {stats['compression_ratio']:.2f}%")
            print(f"  Error: {stats['error']:.6f} (measured: {error:.6f})")
            print(f"  Time: {end_time - start_time:.4f}s")
        
        # Add best result to overall results
        best_result = min(method_results, key=lambda x: x["error"])
        results.append(best_result)
    
    if plot_results and results:
        # Plot compression vs. error
        plt.figure(figsize=(12, 8))
        
        plt.subplot(221)
        methods = [r["method"] for r in results]
        compression = [r["compression_ratio"] for r in results]
        plt.bar(methods, compression)
        plt.title("Compression Ratio by Method")
        plt.ylabel("Compression (%)")
        plt.xticks(rotation=45)
        
        plt.subplot(222)
        errors = [r["error"] for r in results]
        plt.bar(methods, errors)
        plt.title("Error by Method")
        plt.ylabel("Error")
        plt.xticks(rotation=45)
        plt.yscale('log')
        
        plt.subplot(223)
        times = [r["time"] for r in results]
        plt.bar(methods, times)
        plt.title("Processing Time by Method")
        plt.ylabel("Time (s)")
        plt.xticks(rotation=45)
        
        plt.subplot(224)
        plt.scatter(compression, errors, s=100)
        for i, method in enumerate(methods):
            plt.annotate(method, (compression[i], errors[i]), 
                         xytext=(5, 5), textcoords='offset points')
        plt.title("Compression vs. Error")
        plt.xlabel("Compression (%)")
        plt.ylabel("Error")
        plt.yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    return results

def test_adaptive_low_rank(matrix: np.ndarray):
    """Test the adaptive low-rank optimization method."""
    print("\n🔍 Testing Adaptive Low-Rank Optimization")
    print("=" * 80)
    
    # Test with different tolerances
    tolerances = [1e-3, 1e-4, 1e-5]
    results = []
    
    for tolerance in tolerances:
        # Get matrix stats
        stats = get_matrix_stats(matrix, "adaptive_low_rank", tolerance=tolerance)
        
        # Optimize matrix
        config = ModelConfig(
            model_type="test",
            matrix_method="adaptive_low_rank",
            matrix_tolerance=tolerance
        )
        optimizer = Optimizer(config)
        start_time = time.time()
        optimized = optimizer.optimize({"test": matrix})["test"]
        end_time = time.time()
        
        # Calculate actual error
        error = np.linalg.norm(matrix - optimized) / np.linalg.norm(matrix)
        
        results.append({
            "tolerance": tolerance,
            "compression_ratio": stats["compression_ratio"],
            "error": error,
            "time": end_time - start_time
        })
        
        print(f"\n• Tolerance: {tolerance}")
        print(f"  Compression: {stats['compression_ratio']:.2f}%")
        print(f"  Error: {error:.6f}")
        print(f"  Time: {end_time - start_time:.4f}s")
    
    return results

def test_mixed_precision(matrix: np.ndarray):
    """Test the mixed precision optimization method."""
    print("\n🔢 Testing Mixed Precision Optimization")
    print("=" * 80)
    
    # Optimize matrix
    config = ModelConfig(
        model_type="test",
        matrix_method="mixed_precision"
    )
    optimizer = Optimizer(config)
    start_time = time.time()
    optimized = optimizer.optimize({"test": matrix})["test"]
    end_time = time.time()
    
    # Get matrix stats
    stats = get_matrix_stats(matrix, "mixed_precision")
    
    # Calculate error
    error = np.linalg.norm(matrix - optimized) / np.linalg.norm(matrix)
    
    # Visualize original vs optimized
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(matrix[:50, :50], cmap='viridis')
    plt.title("Original Matrix")
    plt.colorbar()
    
    plt.subplot(132)
    plt.imshow(optimized[:50, :50], cmap='viridis')
    plt.title("Mixed Precision Optimized")
    plt.colorbar()
    
    plt.subplot(133)
    plt.imshow(np.abs(matrix[:50, :50] - optimized[:50, :50]), cmap='jet')
    plt.title("Absolute Error")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n• Compression: {stats['compression_ratio']:.2f}%")
    print(f"• Error: {error:.6f}")
    print(f"• Time: {end_time - start_time:.4f}s")
    
    return {
        "compression_ratio": stats["compression_ratio"],
        "error": error,
        "time": end_time - start_time
    }

def main():
    """Main function to run the advanced optimization examples."""
    print("\n🚀 EvaOpt Advanced Optimization Demo")
    print("=" * 80)
    
    # Print available methods
    print_available_methods()
    
    # Create test matrices
    print("\n📊 Creating test matrices...")
    matrix_random = create_test_matrix(1000, 800)
    matrix_low_rank = create_test_matrix(1000, 800, rank=30)
    matrix_sparse = create_test_matrix(1000, 800, sparsity=0.9)
    
    # Compare methods on random matrix
    print("\n⚡ Testing on random matrix...")
    compare_methods(
        matrix_random, 
        ["svd", "truncated_svd", "randomized_svd", "adaptive_low_rank"], 
        [10, 50, 100]
    )
    
    # Compare methods on low-rank matrix
    print("\n⚡ Testing on low-rank matrix...")
    compare_methods(
        matrix_low_rank, 
        ["svd", "truncated_svd", "low_rank", "adaptive_low_rank"], 
        [10, 30, 50]
    )
    
    # Compare methods on sparse matrix
    print("\n⚡ Testing on sparse matrix...")
    compare_methods(
        matrix_sparse, 
        ["sparse", "sparse_pattern", "mixed_precision"], 
        [10]
    )
    
    # Test adaptive low-rank method
    test_adaptive_low_rank(matrix_low_rank)
    
    # Test mixed precision method
    test_mixed_precision(matrix_random)
    
    print("\n✅ Advanced optimization demo completed!")

if __name__ == "__main__":
    main() 