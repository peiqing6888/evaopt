"""
Dynamic neuron optimization interface
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from evaopt_core import DynamicOptimizer as RustDynamicOptimizer

@dataclass
class DynamicConfig:
    """Configuration for dynamic neuron optimization"""
    activation_threshold: float = 0.05  # Lower threshold for more aggressive pruning
    min_active_ratio: float = 0.5  # Keep at least 50% of neurons
    update_frequency: int = 5  # Update more frequently
    ema_alpha: float = 0.2  # Faster updates for EMA

class DynamicOptimizer:
    """High-performance dynamic neuron optimizer"""
    
    def __init__(self, config: Optional[DynamicConfig] = None):
        self.config = config or DynamicConfig()
        self.optimizer = RustDynamicOptimizer(
            activation_threshold=self.config.activation_threshold,
            min_active_ratio=self.config.min_active_ratio,
            update_frequency=self.config.update_frequency,
            ema_alpha=self.config.ema_alpha
        )
        self.hooks = []
        self.layer_names = {}
        self.step = 0
        
    def register_hooks(self, model: torch.nn.Module) -> None:
        """Register forward hooks for all linear layers"""
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                self.layer_names[module] = name
                
                def hook(mod, inp, out, layer_name=name):
                    if isinstance(out, tuple):
                        out = out[0]
                    self.optimizer.update_activations(layer_name, out.detach().cpu().numpy())
                    
                handle = module.register_forward_hook(hook)
                self.hooks.append(handle)
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def optimize_layer(self, name: str, layer: torch.nn.Linear) -> torch.nn.Linear:
        """Optimize a single layer by removing inactive neurons"""
        weights = layer.weight.detach().cpu().numpy()
        bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else None
        
        optimized_weights, optimized_bias = self.optimizer.optimize_layer(name, weights, bias)
        
        optimized_layer = torch.nn.Linear(
            in_features=layer.in_features,
            out_features=len(optimized_weights),
            bias=layer.bias is not None,
            device=layer.weight.device
        )
        
        optimized_layer.weight.data = torch.from_numpy(optimized_weights).to(layer.weight.device)
        if optimized_bias is not None:
            optimized_layer.bias.data = torch.from_numpy(optimized_bias).to(layer.bias.device)
            
        return optimized_layer
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize entire model by removing inactive neurons"""
        # First pass: collect activation statistics
        self.register_hooks(model)
        
        # Create dummy input based on model type
        device = next(model.parameters()).device
        if hasattr(model, "config"):
            # Transformer model
            dummy_input = {
                "input_ids": torch.randint(0, model.config.vocab_size, (1, 512), device=device),
                "attention_mask": torch.ones(1, 512, device=device)
            }
            # Run model to collect statistics
            with torch.no_grad():
                for _ in range(10):  # Collect sufficient statistics
                    model(**dummy_input)
        else:
            # Standard feed-forward network
            first_layer = next(m for m in model.modules() if isinstance(m, torch.nn.Linear))
            dummy_input = torch.randn(32, first_layer.in_features, device=device)
            with torch.no_grad():
                for _ in range(10):
                    model(dummy_input)
        
        self.remove_hooks()
        
        # Second pass: optimize each layer
        optimized_layers = []
        prev_output_size = None
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Optimize layer
                optimized = self.optimize_layer(name, module)
                
                # Update input size if needed
                if prev_output_size is not None:
                    optimized.in_features = prev_output_size
                    optimized.weight = torch.nn.Parameter(
                        torch.randn(optimized.out_features, prev_output_size,
                                  device=optimized.weight.device) * 0.02
                    )
                    if optimized.bias is not None:
                        optimized.bias = torch.nn.Parameter(
                            torch.zeros(optimized.out_features,
                                      device=optimized.bias.device)
                        )
                
                prev_output_size = optimized.out_features
                optimized_layers.append(optimized)
            elif isinstance(module, (torch.nn.ReLU, torch.nn.Dropout)):
                optimized_layers.append(module)
        
        # Create optimized model
        optimized_model = torch.nn.Sequential(*optimized_layers)
        
        # Log optimization results
        total_params_before = sum(p.numel() for p in model.parameters())
        total_params_after = sum(p.numel() for p in optimized_model.parameters())
        compression_ratio = 1 - (total_params_after / total_params_before)
        
        logging.info(f"Model optimization complete:")
        logging.info(f"Parameters before: {total_params_before:,}")
        logging.info(f"Parameters after: {total_params_after:,}")
        logging.info(f"Compression ratio: {compression_ratio:.2%}")
        
        return optimized_model
    
    def get_layer_stats(self, name: str) -> Tuple[int, int, float]:
        """Get statistics for a layer"""
        return self.optimizer.get_stats(name) 