"""
Test basic functionality of the optimization engine
"""

import torch
from evaopt import Optimizer, ModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # 1. Initialize configuration
    config = ModelConfig(
        model_type="llama2",
        quantization_bits=8,
        use_fp16=True,
        max_memory_gb=24.0,
        device="mps"
    )
    
    # 2. Create optimizer
    optimizer = Optimizer(config)
    
    # 3. Create a small test model
    print("Creating test model...")
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10)
    )
    
    # 4. Optimize model
    print("Starting model optimization...")
    try:
        optimized_model = optimizer.optimize_model(model)
        print("Model optimization successful!")
        
        # 5. Check memory usage
        memory_stats = optimizer.get_memory_stats()
        print("\nMemory Usage:")
        for key, value in memory_stats.items():
            print(f"{key}: {value:.2f} GB")
            
    except Exception as e:
        print(f"Error during optimization: {e}")

if __name__ == "__main__":
    main() 