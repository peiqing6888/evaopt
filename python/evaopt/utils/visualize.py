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
        
    def plot_matrix_optimization(self, original, optimized, method_name):
        """Display matrix optimization comparison in terminal."""
        self.print_status(f"\nMatrix Optimization Results for {method_name}:", "info")
        
        # Calculate compression ratio
        orig_nonzero = np.count_nonzero(original)
        opt_nonzero = np.count_nonzero(optimized)
        compression_ratio = 1 - (opt_nonzero / orig_nonzero)
        
        print("\nCompression Analysis:")
        print(f"Original Non-zero Elements: {orig_nonzero}")
        print(f"Optimized Non-zero Elements: {opt_nonzero}")
        print(f"Compression Ratio: {compression_ratio:.2%}")
        
        # Display matrix patterns using ASCII art
        print("\nMatrix Pattern Visualization:")
        self._display_matrix_pattern(original, "Original")
        print()
        self._display_matrix_pattern(optimized, "Optimized")
        
    def _display_matrix_pattern(self, matrix, title, threshold=0.1):
        """Display a simplified matrix pattern using ASCII characters."""
        print(f"\n{title} Matrix Pattern:")
        chars = " ░▒▓█"  # Different density characters
        for i in range(min(10, matrix.shape[0])):  # Show first 10 rows
            row = ""
            for j in range(min(20, matrix.shape[1])):  # Show first 20 columns
                val = abs(matrix[i, j])
                if val < threshold:
                    char = chars[0]
                else:
                    idx = min(int(val * 4), 4)
                    char = chars[idx]
                row += char
            print(row)
            
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