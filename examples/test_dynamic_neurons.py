"""
Test script for dynamic neuron optimization
"""

import numpy as np
import torch
import time
import logging
from evaopt import Optimizer, ModelConfig
from evaopt.core.dynamic import DynamicNeuronConfig, DynamicNeuronOptimizer
from evaopt.utils.visualize import EVAVisualizer

logging.basicConfig(level=logging.INFO)

def create_test_model():
    """Create a test model with known patterns of neuron activation"""
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )
    
    # Initialize with known patterns
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                # Create very sparse patterns
                mask = torch.rand_like(layer.weight) > 0.9  # More sparsity
                layer.weight.data.mul_(mask)
                # Set many neurons to very low activation
                num_neurons = layer.weight.size(0)
                low_neurons = int(num_neurons * 0.6)  # 60% low activation neurons
                indices = torch.randperm(num_neurons)[:low_neurons]
                layer.weight.data[indices] *= 0.000001  # Even lower activation
                
                if layer.bias is not None:
                    layer.bias.data[indices] = 0.0
    
    return model

def generate_test_data(num_samples, input_size=100, num_patterns=5):
    """Generate test data with clear patterns and very low noise"""
    patterns = []
    for _ in range(num_patterns):
        pattern = torch.zeros(input_size)
        # Create very sparse pattern
        active_indices = torch.randperm(input_size)[:input_size//8]  # More sparsity
        pattern[active_indices] = torch.rand(len(active_indices)) * 0.05 + 0.97  # More consistent values
        patterns.append(pattern)
    
    X = torch.zeros((num_samples, input_size))
    y = torch.zeros(num_samples, dtype=torch.long)
    
    for i in range(num_samples):
        pattern_idx = i % num_patterns
        pattern = patterns[pattern_idx]
        # Add extremely low noise
        noise = torch.randn_like(pattern) * 0.01
        X[i] = pattern + noise
        y[i] = pattern_idx
    
    return X, y

def evaluate_model(model, data, labels):
    """Evaluate model performance"""
    with torch.no_grad():
        outputs = model(data)
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()
    return accuracy

def test_optimization():
    """Test dynamic neuron optimization"""
    print("Starting Dynamic Neuron Optimization Test")
    
    # Create model and move to GPU if available
    model = create_test_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Generate data
    X_train, y_train = generate_test_data(10000)  # More training samples
    X_test, y_test = generate_test_data(2000)  # More test samples
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)
    
    # Configure optimizer with more aggressive settings
    config = DynamicNeuronConfig(
        activation_threshold=0.05,  # Lower threshold
        window_size=25,  # Shorter window
        min_active_ratio=0.2,  # Allow more pruning
        update_frequency=2,  # More frequent updates
        stabilization_threshold=0.0005,  # Stricter stability requirement
        warmup_steps=10,  # Shorter warmup
        relative_threshold=True,
        percentile_threshold=20.0,  # More aggressive percentile
        ema_alpha=0.3,  # Faster EMA updates
        min_neurons=4  # Allow more pruning
    )
    
    optimizer = DynamicNeuronOptimizer(config)
    
    # Initial inference time measurement
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(X_test[:32])
    original_time = (time.time() - start_time) / 100
    
    # Run optimization with more iterations
    print("Running initial inference and optimizing model...")
    with torch.no_grad():
        for i in range(500):  # Many more iterations
            batch_idx = torch.randperm(len(X_train))[:32]
            model(X_train[batch_idx])
    
    optimized_model = optimizer.optimize_model(model)
    optimized_model = optimized_model.to(device)
    
    # Measure optimized inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            optimized_model(X_test[:32])
    optimized_time = (time.time() - start_time) / 100
    
    # Calculate accuracy
    with torch.no_grad():
        original_output = model(X_test)
        original_acc = (original_output.argmax(dim=1) == y_test).float().mean().item()
        
        optimized_output = optimized_model(X_test)
        optimized_acc = (optimized_output.argmax(dim=1) == y_test).float().mean().item()
        
        test_acc = (optimized_output.argmax(dim=1) == y_test).float().mean().item()
    
    # Get optimization statistics
    stats = optimizer.get_optimization_stats()
    
    # Print results
    print("\nPerformance Results:")
    print(f"Original inference time: {original_time:.4f}s")
    print(f"Optimized inference time: {optimized_time:.4f}s")
    print(f"Speed improvement: {((original_time - optimized_time) / original_time) * 100:.1f}%")
    
    print("\nAccuracy Results:")
    print(f"Original accuracy: {original_acc:.4f}")
    print(f"Optimized accuracy: {optimized_acc:.4f}")
    print(f"Test set accuracy: {test_acc:.4f}")
    
    print("\nLayer-wise Statistics:")
    for layer_name, layer_stats in stats["layers"].items():
        print(f"\n{layer_name}:")
        print(f"Total neurons: {layer_stats['total_neurons']}")
        print(f"Active neurons: {layer_stats['active_neurons']}")
        print(f"Compression ratio: {layer_stats['compression_ratio']:.2%}")
        print(f"Mean activation: {layer_stats['mean_activation']:.6f}")
        print(f"Std activation: {layer_stats['std_activation']:.6f}")
    
    print("\nOverall Statistics:")
    print(f"Total neurons: {stats['total_neurons']}")
    print(f"Active neurons: {stats['active_neurons']}")
    print(f"Compression ratio: {stats['compression_ratio']:.2%}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_optimization() 