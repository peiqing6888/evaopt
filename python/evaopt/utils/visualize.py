"""EVA-styled visualization utilities for EvaOpt."""

import os
import sys
import time
from datetime import datetime
from termcolor import colored
import numpy as np

class EVAVisualizer:
    """Terminal-based visualization for EvaOpt."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.start_time = None
        self.monitoring = False
        self.metrics = {}
        
    def start_monitoring(self):
        """Start the monitoring session."""
        self.start_time = datetime.now()
        self.monitoring = True
        
    def stop_monitoring(self):
        """Stop the monitoring session."""
        self.monitoring = False
        
    def print_status(self, message, status_type="info"):
        """Print a status message with EVA-style formatting."""
        color_map = {
            "info": "blue",
            "success": "green",
            "warning": "yellow",
            "error": "red"
        }
        color = color_map.get(status_type, "white")
        prefix = colored(f"[{status_type.upper()}]", color)
        print(f"{prefix} {message}")
        
    def plot_matrix_optimization(self, original: np.ndarray, optimized: np.ndarray, title: str = "Matrix Optimization"):
        """Plot original and optimized matrix patterns."""
        self.print_status(f"\nMatrix Optimization Results for {title}", "info")
        
        # Calculate compression statistics
        orig_nonzero = np.count_nonzero(original)
        opt_nonzero = np.count_nonzero(optimized)
        
        # Avoid division by zero
        if orig_nonzero > 0:
            compression_ratio = 1 - (opt_nonzero / orig_nonzero)
        else:
            compression_ratio = 0.0
        
        self.print_status("\nCompression Analysis:", "info")
        self.print_status(f"Original Non-zero Elements: {orig_nonzero}", "info")
        self.print_status(f"Optimized Non-zero Elements: {opt_nonzero}", "info")
        self.print_status(f"Compression Ratio: {compression_ratio:.2%}", "info")
        
        # Visualization
        self.print_status("\nMatrix Pattern Visualization:\n", "info")
        
        # Convert to binary pattern for visualization
        def to_pattern(arr):
            pattern = np.zeros_like(arr, dtype=str)
            pattern[arr == 0] = ' '
            pattern[np.abs(arr) > 0] = '█'
            pattern[np.abs(arr) > np.max(np.abs(arr))*0.75] = '█'
            pattern[np.abs(arr) > np.max(np.abs(arr))*0.5] = '▓'
            pattern[np.abs(arr) > np.max(np.abs(arr))*0.25] = '▒'
            pattern[np.abs(arr) > 0] = '░'
            return pattern
        
        # Print patterns
        orig_pattern = to_pattern(original)
        opt_pattern = to_pattern(optimized)
        
        self.print_status("Original Matrix Pattern:", "info")
        for row in orig_pattern:
            print(''.join(row))
        
        print("\n")
        self.print_status("Optimized Matrix Pattern:", "info")
        for row in opt_pattern:
            print(''.join(row))
        
    def update_metrics(self, metrics):
        """Update and display the current metrics."""
        self.metrics = metrics
        
        # Display memory usage
        memory = metrics.get("memory", {})
        if memory:
            used = memory.get("used", 0)
            available = memory.get("available", 0)
            total = used + available
            usage_percent = (used / total) * 100 if total > 0 else 0
            
            print("\nMemory Usage:")
            print(f"Used: {used:.2f} GB")
            print(f"Available: {available:.2f} GB")
            self._print_progress_bar(usage_percent, "Memory")
            
        # Display performance metrics
        performance = metrics.get("performance", {})
        if performance:
            print("\nPerformance Metrics:")
            for metric, value in performance.items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")
                    
    def _print_progress_bar(self, percentage, label, width=50):
        """Print a colored progress bar."""
        filled = int(width * percentage / 100)
        bar = colored('█' * filled, 'green') + '-' * (width - filled)
        print(f"{label}: [{bar}] {percentage:.1f}%") 