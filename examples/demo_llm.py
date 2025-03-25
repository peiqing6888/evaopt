"""
Demo script for LLM optimization
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel
from evaopt import Optimizer, ModelConfig
import psutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_device():
    """Get the appropriate device for the current system."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def plot_memory_usage(original: float, optimized: float):
    """Plot memory usage comparison."""
    plt.figure(figsize=(8, 4))
    bars = plt.bar(['Original', 'Optimized'], 
                  [original, optimized],
                  color=['lightcoral', 'lightgreen'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} GB',
                ha='center', va='bottom')
    
    plt.title('Memory Usage Comparison')
    plt.ylabel('Memory (GB)')
    plt.grid(True, alpha=0.3)
    plt.show()

def measure_inference_time(model, tokenizer, device, prompt: str, n_runs: int = 3):
    """Measure inference time."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    times = []
    
    for i in tqdm(range(n_runs), desc="Running inference"):
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        times.append(time.time() - start_time)
    
    return np.mean(times), np.std(times)

def create_tiny_model():
    """Create a tiny GPT-2 model for testing."""
    config = GPT2Config(
        vocab_size=50257,
        n_positions=128,
        n_embd=256,
        n_layer=4,
        n_head=4
    )
    model = GPT2LMHeadModel(config)
    return model

def main():
    print("\n🚀 EvaOpt LLM Optimization Demo")
    print("="*50)
    
    # Get appropriate device
    device = get_device()
    print(f"\n🖥️ Using device: {device}")
    
    # Configure optimizer
    config = ModelConfig(
        model_type="gpt2",
        quantization_bits=8,
        use_fp16=True,
        max_memory_gb=24.0,
        device=device,
        matrix_method="truncated_svd",
        matrix_rank=50
    )
    
    optimizer = Optimizer(config)
    
    # Load model
    print("\n📚 Loading model...")
    
    # Measure original memory
    original_memory = psutil.Process().memory_info().rss / 1024**3
    print(f"Initial memory usage: {original_memory:.2f} GB")
    
    print("Creating tiny model for testing...")
    model = create_tiny_model()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("Model created successfully!")
    
    # Move model to device
    model = model.to(device)
    if device == "mps":
        # Enable MPS optimizations
        torch.mps.empty_cache()
    
    # Optimize model
    print("\n⚡ Optimizing model...")
    start_time = time.time()
    optimized_model = optimizer.optimize_model(model)
    optimization_time = time.time() - start_time
    print(f"Optimization completed in {optimization_time:.2f} seconds")
    
    # Measure optimized memory
    optimized_memory = psutil.Process().memory_info().rss / 1024**3
    print(f"Optimized memory usage: {optimized_memory:.2f} GB")
    
    # Visualize memory usage
    plot_memory_usage(original_memory, optimized_memory)
    
    # Test inference
    print("\n🤖 Testing inference...")
    prompt = "Artificial Intelligence is"
    
    # Measure inference time for both models
    print("\nMeasuring original model inference time...")
    original_time, original_std = measure_inference_time(
        model, tokenizer, device, prompt
    )
    
    print("\nMeasuring optimized model inference time...")
    optimized_time, optimized_std = measure_inference_time(
        optimized_model, tokenizer, device, prompt
    )
    
    print(f"\n⏱️ Inference time:")
    print(f"• Original: {original_time:.3f}s ± {original_std:.3f}s")
    print(f"• Optimized: {optimized_time:.3f}s ± {optimized_std:.3f}s")
    print(f"• Speedup: {original_time/optimized_time:.1f}x")
    
    # Generate sample output
    print("\n📝 Sample output:")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("\nGenerating with optimized model...")
    with torch.no_grad():
        outputs = optimized_model.generate(
            **inputs,
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    
    print("\n✨ Demo completed!")

if __name__ == "__main__":
    main() 