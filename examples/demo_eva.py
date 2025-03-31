"""
EVA-styled LLM Optimization Demo with Real-time Visualization
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
from evaopt.utils.visualize import EVAVisualizer
import psutil
import numpy as np

# Initialize visualizer
visualizer = EVAVisualizer()

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
    visualizer.print_status("Initializing EVA Neural Core...", "info")
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
    
    visualizer.print_status(f"Results saved to: {filename}", "success")

def run_optimization_demo():
    """Run the EVA-styled optimization demo."""
    print_header()
    visualizer.start_monitoring()
    
    try:
        # Initialize system
        device = get_device()
        visualizer.print_status(f"Neural Interface: {device.upper()}", "info")
        
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
            visualizer.print_status(f"• {cfg['name']} initialized", "info")
        
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
            
            # Optimization steps with visualization
            steps = [
                "Synchronizing neural patterns...",
                "Optimizing synaptic connections...",
                "Compressing neural pathways...",
                "Stabilizing AT Field...",
                "Neural optimization complete."
            ]
            
            # Get initial matrix for visualization
            initial_weight = next(model.parameters()).detach().cpu().numpy()
            
            # Optimize model
            optimized_model = optimizer.optimize_model(model)
            
            # Get optimized matrix for visualization
            optimized_weight = next(optimized_model.parameters()).detach().cpu().numpy()
            
            # Visualize matrix optimization
            visualizer.plot_matrix_optimization(
                initial_weight[:50, :50],
                optimized_weight[:50, :50],
                cfg['name']
            )
            
            for step in steps:
                visualizer.print_status(step, "info")
                time.sleep(0.5)
            
            memory_monitor.stop()
            
            # Evaluate performance with visualization
            metrics = {
                "perplexity": 0.0,
                "bleu_score": 0.0,
                "inference_time": 0.0,
                "memory": 0.0
            }
            
            print_section("Analysis", "Evaluating Neural Performance:", 'blue')
            
            progress_data = []
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
                
                # Update progress data
                progress_data.append(metrics["perplexity"])
            
            # Average metrics
            metrics["perplexity"] /= len(test_prompts)
            metrics["bleu_score"] /= len(test_prompts)
            metrics["inference_time"] /= len(test_prompts)
            
            # Update visualization
            visualizer.update_metrics({
                "memory": {
                    "used": metrics["memory"],
                    "available": psutil.virtual_memory().available / 1024**3
                },
                "progress": {
                    "times": list(range(len(progress_data))),
                    "values": progress_data
                },
                "compression": {
                    "matrix": optimized_weight[:50, :50]
                },
                "performance": {
                    "Perplexity": metrics["perplexity"],
                    "BLEU": metrics["bleu_score"],
                    "Speed": 1.0 / metrics["inference_time"],
                    "Memory": metrics["memory"] / (psutil.virtual_memory().total / 1024**3)
                }
            })
            
            results.append({
                "method": cfg["name"],
                **metrics
            })
            
            # Display results
            visualizer.print_status("\nPerformance Analysis:", "info")
            print_slow(colored(format_metrics(metrics), 'cyan'))
            print("\n" + colored("-"*70, 'green') + "\n")
        
        # Save results
        save_results(results)
        
        # Final status
        print_section("Complete", "Neural Optimization Sequence Terminated", 'magenta')
        visualizer.print_status("All systems nominal. Thank you for using EvaOpt.", "success")
        
    finally:
        visualizer.stop_monitoring()

if __name__ == "__main__":
    try:
        run_optimization_demo()
    except KeyboardInterrupt:
        visualizer.print_status("\n\nEmergency shutdown initiated. Goodbye!", "error") 