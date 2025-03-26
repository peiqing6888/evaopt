"""
EVA-styled LLM Optimization Demo
"""

import time
import os
import sys
import torch
import json
from datetime import datetime
from pyfiglet import Figlet
from termcolor import colored
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from evaopt import Optimizer, ModelConfig
from evaopt.utils.metrics import MemoryMonitor, calculate_perplexity, calculate_bleu
import psutil
import numpy as np

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_slow(text, delay=0.03):
    """Print text slowly, character by character."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def print_ascii_art(text, font='slant', color='magenta'):
    """Print text in ASCII art style with EVA colors."""
    f = Figlet(font=font)
    ascii_art = f.renderText(text)
    print(colored(ascii_art, color))

def print_header():
    """Print the EvaOpt header in EVA style."""
    clear_screen()
    # Purple like EVA Unit-01
    print_ascii_art("EvaOpt", font='slant', color='magenta')
    # Blue like Rei's hair
    print_slow(colored("Neural Network Optimization System", 'blue'))
    # Red like warning text
    print_slow(colored("WARNING: B-Type Neural Connection Detected", 'red'))
    # Green like NERV displays
    print("\n" + colored("="*70, 'green') + "\n")

def print_section(title, content, color='magenta'):
    """Print a section with EVA-themed colors."""
    print_ascii_art(title, font='small', color=color)
    print_slow(content)
    print()

def get_device():
    """Get the appropriate device for the current system."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def create_test_model():
    """Create a small test model."""
    print_slow(colored("Initializing EVA Neural Core...", 'yellow'))
    config = GPT2Config(
        vocab_size=50257,
        n_positions=128,
        n_embd=256,
        n_layer=4,
        n_head=4
    )
    return GPT2LMHeadModel(config)

def format_metrics(metrics):
    """Format metrics for display."""
    return (
        f"Perplexity: {metrics['perplexity']:.2f}\n"
        f"BLEU Score: {metrics['bleu_score']:.3f}\n"
        f"Inference Time: {metrics['inference_time']:.3f}s\n"
        f"Memory Usage: {metrics['memory']:.2f}GB"
    )

def save_results(results: list):
    """Save evaluation results with EVA-style formatting."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/eva_optimization_{timestamp}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
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
    
    with open(filename, 'w') as f:
        json.dump(processed_results, f, indent=2)
    
    print_slow(colored(f"\nResults saved to: {filename}", 'green'))

def run_optimization_demo():
    """Run the EVA-styled optimization demo."""
    print_header()
    
    # Initialize system
    device = get_device()
    print_slow(colored(f"Neural Interface: {device.upper()}", 'yellow'))
    
    # Configuration setup
    print_section("Config", "Initializing Optimization Parameters:", 'blue')
    configs = [
        {
            "name": "EVA-01 Mode",
            "config": ModelConfig(
                model_type="gpt2",
                quantization_bits=8,
                use_fp16=True,
                device=device,
                matrix_method="truncated_svd",
                matrix_rank=50
            )
        },
        {
            "name": "EVA-02 Mode",
            "config": ModelConfig(
                model_type="gpt2",
                quantization_bits=4,
                use_fp16=True,
                device=device,
                matrix_method="low_rank",
                matrix_rank=30
            )
        }
    ]
    
    for cfg in configs:
        print_slow(colored(f"• {cfg['name']} initialized", 'yellow'))
    
    # Model initialization
    print_section("System", "Loading Neural Network...", 'magenta')
    model = create_test_model()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = model.to(device)
    
    # Test prompts
    test_prompts = [
        "Neural network optimization is",
        "The EVA system can",
        "Advanced AI technology"
    ]
    
    # Run optimization for each configuration
    results = []
    for cfg in configs:
        print_section("Optimize", f"Engaging {cfg['name']}...", 'red')
        
        optimizer = Optimizer(cfg['config'])
        memory_monitor = MemoryMonitor()
        memory_monitor.start()
        
        # Optimization steps
        steps = [
            "Synchronizing neural patterns...",
            "Optimizing synaptic connections...",
            "Compressing neural pathways...",
            "Stabilizing AT Field...",
            "Neural optimization complete."
        ]
        
        optimized_model = optimizer.optimize_model(model)
        
        for step in steps:
            print_slow(f"[{colored('•', 'red')}] {step}")
            time.sleep(0.5)
        
        memory_monitor.stop()
        
        # Evaluate performance
        metrics = {
            "perplexity": 0.0,
            "bleu_score": 0.0,
            "inference_time": 0.0,
            "memory": 0.0
        }
        
        print_section("Analysis", "Evaluating Neural Performance:", 'blue')
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                outputs = optimized_model.generate(
                    **inputs,
                    max_length=100,
                    num_return_sequences=1,
                    temperature=0.7
                )
            inference_time = time.time() - start_time
            
            # Calculate metrics
            metrics["perplexity"] += calculate_perplexity(optimized_model, inputs.input_ids)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            metrics["bleu_score"] += calculate_bleu(prompt + " technology", generated_text)
            metrics["inference_time"] += inference_time
            metrics["memory"] = psutil.Process().memory_info().rss / 1024**3
        
        # Average metrics
        metrics["perplexity"] /= len(test_prompts)
        metrics["bleu_score"] /= len(test_prompts)
        metrics["inference_time"] /= len(test_prompts)
        
        results.append({
            "method": cfg["name"],
            **metrics
        })
        
        # Display results
        print_slow(colored("\nPerformance Analysis:", 'yellow'))
        print_slow(colored(format_metrics(metrics), 'cyan'))
        print("\n" + colored("-"*70, 'green') + "\n")
    
    # Save results
    save_results(results)
    
    # Final status
    print_section("Complete", "Neural Optimization Sequence Terminated", 'magenta')
    print_slow(colored("All systems nominal. Thank you for using EvaOpt.", 'blue'))

if __name__ == "__main__":
    try:
        run_optimization_demo()
    except KeyboardInterrupt:
        print_slow(colored("\n\nEmergency shutdown initiated. Goodbye!", 'red')) 