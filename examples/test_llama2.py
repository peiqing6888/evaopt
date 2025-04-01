"""
Test script for dynamic neuron optimization
"""

import torch
import time
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from evaopt.core.dynamic import DynamicNeuronConfig
from evaopt.core.dynamic import DynamicNeuronOptimizer

logging.basicConfig(level=logging.INFO)

@dataclass
class TestConfig:
    """Test configuration"""
    device: str = "mps"  # Apple Silicon GPU
    save_dir: str = "results/dynamic_test"
    
    # Dynamic neuron config
    activation_threshold: float = 0.05
    min_active_ratio: float = 0.3
    window_size: int = 50
    
    def __post_init__(self):
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

class SimpleTestModel(torch.nn.Module):
    """A simple test model for dynamic neuron optimization"""
    
    def __init__(self, input_dim=128, hidden_dims=[256, 512, 256, 128], output_dim=10):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims)-1):
            layers.append(torch.nn.Linear(dims[i], dims[i+1]))
            layers.append(torch.nn.ReLU())
        
        layers.append(torch.nn.Linear(dims[-1], output_dim))
        self.model = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class DynamicNeuronTest:
    """Test dynamic neuron optimization"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.model = None
        self.optimized_model = None
        self.results = {}
        
    def setup_model(self) -> None:
        """Create test model"""
        logging.info("Creating test model...")
        
        self.model = SimpleTestModel()
        self.model = self.model.to(self.config.device)
        
        # Initialize with sparse patterns
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, torch.nn.Linear):
                    # Create sparse patterns
                    mask = torch.rand_like(module.weight) > 0.7
                    module.weight.data.mul_(mask)
                    
                    # Set many neurons to very low activation
                    num_neurons = module.weight.size(0)
                    low_neurons = int(num_neurons * 0.4)
                    indices = torch.randperm(num_neurons)[:low_neurons]
                    module.weight.data[indices] *= 0.001
                    
                    if module.bias is not None:
                        module.bias.data[indices] = 0.0
        
    def measure_baseline(self) -> None:
        """Measure baseline model performance"""
        logging.info("Measuring baseline performance...")
        
        # Create input data
        batch_size = 32
        input_dim = 128
        inputs = torch.rand(batch_size, input_dim, device=self.config.device)
        
        # Measure inference time
        times = []
        for _ in range(10):
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(inputs)
            times.append(time.time() - start_time)
        
        avg_time = sum(times) / len(times)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.results["baseline"] = {
            "inference_time": avg_time,
            "total_params": total_params,
            "trainable_params": trainable_params
        }
        
        logging.info(f"Baseline inference time: {avg_time:.6f}s")
        logging.info(f"Total parameters: {total_params}")
        
    def optimize_model(self) -> None:
        """Optimize the model"""
        logging.info("Optimizing model...")
        
        # Configure dynamic neuron optimization
        dynamic_config = DynamicNeuronConfig(
            activation_threshold=self.config.activation_threshold,
            min_active_ratio=self.config.min_active_ratio,
            window_size=self.config.window_size
        )
        
        # Create dynamic optimizer
        optimizer = DynamicNeuronOptimizer(dynamic_config)
        
        # Create input data
        batch_size = 32
        input_dim = 128
        inputs = torch.rand(batch_size, input_dim, device=self.config.device)
        
        # Run model to collect activation statistics
        with torch.no_grad():
            for _ in range(dynamic_config.window_size * 2):
                self.model(inputs)
        
        # Optimize model
        self.optimized_model = optimizer.optimize_model(self.model)
        self.optimized_model = self.optimized_model.to(self.config.device)
        
        # Get optimization stats
        stats = optimizer.get_optimization_stats()
        logging.info(f"Model optimization complete. Compression ratio: {stats['compression_ratio']:.2%}")
        
        self.results["optimization"] = stats
        
    def evaluate_optimized(self) -> None:
        """Evaluate optimized model"""
        logging.info("Evaluating optimized model...")
        
        if self.optimized_model is None:
            logging.error("No optimized model available to evaluate")
            return
            
        # Create input data
        batch_size = 32
        input_dim = 128
        inputs = torch.rand(batch_size, input_dim, device=self.config.device)
        
        # Measure inference time for original model
        times_original = []
        for _ in range(10):
            # Warmup
            with torch.no_grad():
                _ = self.model(inputs)
            
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(inputs)
            times_original.append(time.time() - start_time)
        
        avg_time_original = sum(times_original) / len(times_original)
        
        # Measure inference time for optimized model
        times_optimized = []
        for _ in range(10):
            # Warmup
            with torch.no_grad():
                _ = self.optimized_model(inputs)
            
            start_time = time.time()
            with torch.no_grad():
                _ = self.optimized_model(inputs)
            times_optimized.append(time.time() - start_time)
        
        avg_time_optimized = sum(times_optimized) / len(times_optimized)
        
        # Count parameters
        original_params = sum(p.numel() for p in self.model.parameters())
        optimized_params = sum(p.numel() for p in self.optimized_model.parameters() if p.requires_grad)
        
        self.results["optimized"] = {
            "inference_time_original": avg_time_original,
            "inference_time_optimized": avg_time_optimized,
            "speedup": avg_time_original / avg_time_optimized if avg_time_optimized > 0 else 0,
            "original_params": original_params,
            "optimized_params": optimized_params,
            "param_reduction": (original_params - optimized_params) / original_params
        }
        
        # Calculate improvement
        time_improvement = (avg_time_original - avg_time_optimized) / avg_time_original * 100 if avg_time_original > 0 else 0
        
        logging.info(f"Time improvement: {time_improvement:.1f}%")
        logging.info(f"Original model inference time: {avg_time_original:.6f}s")
        logging.info(f"Optimized model inference time: {avg_time_optimized:.6f}s")
        logging.info(f"Speedup factor: {self.results['optimized']['speedup']:.2f}x")
        logging.info(f"Parameter reduction: {self.results['optimized']['param_reduction']:.2%}")
        
    def save_results(self) -> None:
        """Save test results"""
        results_file = Path(self.config.save_dir) / "results.pt"
        torch.save(self.results, results_file)
        logging.info(f"Results saved to {results_file}")
        
    def run(self) -> None:
        """Run the complete test"""
        try:
            self.setup_model()
            self.measure_baseline()
            self.optimize_model()
            self.evaluate_optimized()
            self.save_results()
        except Exception as e:
            logging.error(f"Test failed: {str(e)}")
            raise

def main():
    config = TestConfig()
    test = DynamicNeuronTest(config)
    test.run()

if __name__ == "__main__":
    main() 