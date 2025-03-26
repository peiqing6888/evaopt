"""
Demo script for LLM optimization
"""

import torch
import time
import json
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel
from evaopt import Optimizer, ModelConfig
from evaopt.utils.metrics import MemoryMonitor, calculate_perplexity, calculate_bleu, plot_optimization_comparison
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

def evaluate_model(model, tokenizer, device, test_prompts):
    """Evaluate model performance using multiple metrics."""
    results = {
        "perplexity": [],
        "bleu_scores": [],
        "inference_times": []
    }
    
    for prompt in tqdm(test_prompts, desc="Evaluating"):
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Measure inference time
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
        inference_time = time.time() - start_time
        
        # Calculate perplexity
        perplexity = calculate_perplexity(model, inputs.input_ids)
        
        # Calculate BLEU score
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reference_text = prompt + " is a fascinating field that has revolutionized technology"  # Example reference
        bleu_score = calculate_bleu(reference_text, generated_text)
        
        # Store results
        results["perplexity"].append(perplexity)
        results["bleu_scores"].append(bleu_score)
        results["inference_times"].append(inference_time)
    
    # Average results
    return {
        "perplexity": np.mean(results["perplexity"]),
        "bleu_score": np.mean(results["bleu_scores"]),
        "inference_time": np.mean(results["inference_times"])
    }

def optimize_and_evaluate(model, config_variants, tokenizer, device, test_prompts):
    """Optimize model with different configurations and evaluate performance."""
    results = []
    
    for variant in config_variants:
        print(f"\n⚡ Optimizing model with {variant['name']}...")
        optimizer = Optimizer(variant["config"])
        
        # Setup memory monitor
        memory_monitor = MemoryMonitor()
        memory_monitor.start()
        
        # Optimize model
        optimized_model = optimizer.optimize_model(model)
        
        # Stop memory monitoring and plot
        memory_monitor.stop()
        memory_monitor.plot(title=f"Memory Usage During {variant['name']} Optimization")
        
        # Evaluate performance
        metrics = evaluate_model(optimized_model, tokenizer, device, test_prompts)
        
        # Store results
        results.append({
            "method": variant["name"],
            "perplexity": metrics["perplexity"],
            "bleu_score": metrics["bleu_score"],
            "inference_time": metrics["inference_time"],
            "memory": psutil.Process().memory_info().rss / 1024**3
        })
    
    return results

def save_results(results: list, filename: str = None):
    """Save evaluation results to a JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/optimization_results_{timestamp}.json"
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert numpy values to Python types
    processed_results = []
    for result in results:
        processed = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                processed[k] = v.tolist()
            elif isinstance(v, np.float32):
                processed[k] = float(v)
            else:
                processed[k] = v
        processed_results.append(processed)
    
    # Save results
    with open(filename, 'w') as f:
        json.dump(processed_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")

def print_results_table(results: list):
    """Print results in a formatted table."""
    print("\n📊 Optimization Results")
    print("="*80)
    
    # Print header
    headers = ["Method", "Perplexity", "BLEU Score", "Inference Time (s)", "Memory (GB)"]
    header = "| " + " | ".join(f"{h:^15}" for h in headers) + " |"
    print(header)
    print("-" * len(header))
    
    # Print results
    for result in results:
        row = [
            f"{result['method']:^15}",
            f"{result['perplexity']:^15.2f}",
            f"{result['bleu_score']:^15.3f}",
            f"{result['inference_time']:^15.3f}",
            f"{result['memory']:^15.2f}"
        ]
        print("| " + " | ".join(row) + " |")
    
    print("="*80)

def main():
    print("\n🚀 EvaOpt LLM Optimization Demo")
    print("="*50)
    
    # Get appropriate device
    device = get_device()
    print(f"\n🖥️ Using device: {device}")
    
    # Define optimization configurations to test
    config_variants = [
        {
            "name": "Base (8-bit)",
            "config": ModelConfig(
                model_type="gpt2",
                quantization_bits=8,
                use_fp16=True,
                max_memory_gb=24.0,
                device=device,
                matrix_method="truncated_svd",
                matrix_rank=50
            )
        },
        {
            "name": "4-bit Quantization",
            "config": ModelConfig(
                model_type="gpt2",
                quantization_bits=4,
                use_fp16=True,
                max_memory_gb=24.0,
                device=device,
                matrix_method="truncated_svd",
                matrix_rank=50
            )
        },
        {
            "name": "Low Rank",
            "config": ModelConfig(
                model_type="gpt2",
                quantization_bits=8,
                use_fp16=True,
                max_memory_gb=24.0,
                device=device,
                matrix_method="low_rank",
                matrix_rank=30
            )
        }
    ]
    
    # Load model
    print("\n📚 Loading model...")
    model = create_tiny_model()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = model.to(device)
    
    # Define test prompts
    test_prompts = [
        "Artificial Intelligence is",
        "Machine learning helps",
        "Deep neural networks can",
        "Natural language processing enables",
        "Computer vision technology"
    ]
    
    # Run optimization and evaluation
    results = optimize_and_evaluate(model, config_variants, tokenizer, device, test_prompts)
    
    # Print results table
    print_results_table(results)
    
    # Plot comparison
    plot_optimization_comparison(
        results=results,
        metrics=["perplexity", "bleu_score", "inference_time", "memory"],
        title="Optimization Methods Comparison"
    )
    
    # Save results
    save_results(results)
    
    print("\n✨ Demo completed!")

if __name__ == "__main__":
    main() 