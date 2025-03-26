"""
Evaluation metrics utilities
"""

import torch
import numpy as np
from typing import List, Dict, Any
import psutil
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from threading import Thread
import matplotlib.pyplot as plt
from collections import deque

class MemoryMonitor:
    """Real-time memory usage monitor"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.memory_usage = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)
        self.is_running = False
        self._monitor_thread = None
    
    def start(self):
        """Start monitoring memory usage"""
        self.is_running = True
        self._monitor_thread = Thread(target=self._monitor, daemon=True)
        self._monitor_thread.start()
    
    def stop(self):
        """Stop monitoring memory usage"""
        self.is_running = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor(self):
        """Monitor memory usage in background"""
        while self.is_running:
            memory = psutil.Process().memory_info().rss / 1024**3  # GB
            self.memory_usage.append(memory)
            self.timestamps.append(time.time())
            time.sleep(self.interval)
    
    def plot(self, title: str = "Memory Usage Over Time"):
        """Plot memory usage"""
        if not self.memory_usage:
            return
        
        plt.figure(figsize=(10, 5))
        times = np.array(self.timestamps) - self.timestamps[0]  # Relative time
        plt.plot(times, self.memory_usage)
        plt.title(title)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory Usage (GB)")
        plt.grid(True, alpha=0.3)
        plt.show()

def calculate_perplexity(model: torch.nn.Module, 
                        input_ids: torch.Tensor,
                        attention_mask: torch.Tensor = None) -> float:
    """
    Calculate model perplexity on given input
    
    Args:
        model: Language model
        input_ids: Input token IDs
        attention_mask: Attention mask (optional)
    
    Returns:
        Perplexity score
    """
    with torch.no_grad():
        # Move inputs to model device
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Get model outputs
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        
        # Calculate perplexity from cross entropy loss
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            return torch.exp(outputs.loss).item()
        else:
            # Fallback: calculate perplexity from logits
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                          shift_labels.view(-1))
            
            return torch.exp(loss).item()

def calculate_bleu(reference: str, 
                  hypothesis: str,
                  weights: tuple = (0.5, 0.3, 0.2)) -> float:
    """
    Calculate BLEU score between reference and hypothesis
    
    Args:
        reference: Reference text
        hypothesis: Generated text
        weights: Weights for n-grams (default: focus on lower order n-grams)
    
    Returns:
        BLEU score
    """
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    
    # Use smoothing method 7 (exponential decay) which works better for short texts
    smoothing = SmoothingFunction().method7
    
    # Calculate BLEU score with smoothing and focus on 1-3 grams
    try:
        return sentence_bleu(
            [reference_tokens], 
            hypothesis_tokens, 
            weights=weights,
            smoothing_function=smoothing
        )
    except Exception as e:
        print(f"Warning: BLEU score calculation failed: {e}")
        return 0.0

def plot_optimization_comparison(results: List[Dict[str, Any]],
                              metrics: List[str],
                              title: str = "Optimization Methods Comparison"):
    """
    Plot comparison of different optimization methods
    
    Args:
        results: List of dictionaries containing results for each method
        metrics: List of metric names to plot
        title: Plot title
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    methods = [r["method"] for r in results]
    
    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        axes[i].bar(methods, values)
        axes[i].set_title(metric)
        axes[i].set_ylabel(metric)
        axes[i].grid(True, alpha=0.3)
        plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.suptitle(title, y=1.05)
    plt.show() 