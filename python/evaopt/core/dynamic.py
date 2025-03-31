"""
Dynamic neuron optimization implementation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import logging

@dataclass
class DynamicNeuronConfig:
    """Configuration for dynamic neuron optimization"""
    activation_threshold: float = 0.01  # Minimum activation threshold
    window_size: int = 100  # Number of forward passes to track
    min_active_ratio: float = 0.3  # Minimum ratio of active neurons to keep
    update_frequency: int = 10  # How often to update activation statistics
    stabilization_threshold: float = 0.001  # Threshold for activation stability
    warmup_steps: int = 50  # Number of steps before starting optimization
    relative_threshold: bool = True  # Use relative thresholds based on layer statistics
    percentile_threshold: float = 25.0  # Percentile threshold for neuron pruning
    ema_alpha: float = 0.1  # EMA weight for neuron statistics
    min_neurons: int = 4  # Minimum number of neurons to keep per layer

class DynamicNeuronOptimizer:
    """Dynamic neuron optimization implementation"""
    
    def __init__(self, config: DynamicNeuronConfig):
        self.config = config
        self.activation_history: Dict[str, Dict[int, List[float]]] = {}
        self.frozen_neurons: Dict[str, Set[int]] = {}
        self.step_counter = 0
        self.layer_stats: Dict[str, Dict] = {}
        self.hooks = []
        self.input_shape = None
        self.layer_names = {}  # Map module to layer name
        self.linear_layers = []  # Track linear layers in order
        self.layer_counter = 0  # Counter for generating unique layer names
        self.module_order = []  # Track module order
        self.layer_dims = {}  # Track input/output dimensions for each layer
    
    def _get_unique_layer_name(self, module: torch.nn.Module) -> str:
        """Generate a unique name for a layer"""
        name = f"linear_{self.layer_counter}"
        self.layer_counter += 1
        return name
    
    def register_activation_hook(self, module: torch.nn.Module, name: Optional[str] = None) -> None:
        """Register forward hook to monitor activations"""
        if name is None:
            name = self._get_unique_layer_name(module)
        
        self.layer_names[module] = name  # Store layer name
        if isinstance(module, torch.nn.Linear):
            self.linear_layers.append((name, module))
            
            # Initialize layer statistics
            if name not in self.layer_stats:
                self.layer_stats[name] = {
                    "max_activation": float('-inf'),
                    "min_activation": float('inf'),
                    "total_neurons": module.out_features,
                    "active_neurons": module.out_features,
                    "mean_activation": 0.0,
                    "std_activation": 0.0,
                    "percentile_threshold": 0.0,
                    "neuron_means": np.zeros(module.out_features),
                    "neuron_peaks": np.zeros(module.out_features)
                }
            
            # Initialize activation history
            if name not in self.activation_history:
                self.activation_history[name] = {}
                self.frozen_neurons[name] = set()
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            
            # Store layer dimensions
            if isinstance(module, torch.nn.Linear):
                self.layer_dims[name] = {
                    "in_features": input[0].shape[-1],
                    "out_features": output.shape[-1]
                }
            
            if self.step_counter >= self.config.warmup_steps and \
               self.step_counter % self.config.update_frequency == 0:
                self.update_activations(self.layer_names[module], output)
            self.step_counter += 1
        
        hook_handle = module.register_forward_hook(hook)
        self.hooks.append(hook_handle)
        return hook_handle
    
    def update_activations(self, layer_name: str, activations: torch.Tensor) -> None:
        """Update activation history for a layer"""
        # Convert to numpy for efficient processing
        act_np = activations.detach().cpu().numpy()
        
        # Calculate mean activation per neuron
        neuron_activations = np.mean(np.abs(act_np), axis=0)
        
        stats = self.layer_stats[layer_name]
        current_max = float(np.max(neuron_activations))
        current_min = float(np.min(neuron_activations))
        current_mean = float(np.mean(neuron_activations))
        current_std = float(np.std(neuron_activations))
        
        # Update running statistics with EMA
        alpha = self.config.ema_alpha
        stats["max_activation"] = max(stats["max_activation"], current_max)
        stats["min_activation"] = min(stats["min_activation"], current_min)
        stats["mean_activation"] = (1 - alpha) * stats["mean_activation"] + alpha * current_mean
        stats["std_activation"] = (1 - alpha) * stats["std_activation"] + alpha * current_std
        
        # Update neuron-specific statistics
        stats["neuron_means"] = (1 - alpha) * stats["neuron_means"] + alpha * neuron_activations
        stats["neuron_peaks"] = np.maximum(stats["neuron_peaks"], neuron_activations)
        
        # Calculate percentile threshold based on neuron means
        stats["percentile_threshold"] = float(np.percentile(stats["neuron_means"], self.config.percentile_threshold))
        
        # Calculate adaptive threshold based on layer statistics
        mean_threshold = stats["mean_activation"] * self.config.activation_threshold
        peak_threshold = np.mean(stats["neuron_peaks"]) * self.config.activation_threshold
        percentile_threshold = stats["percentile_threshold"]
        
        # Use the most aggressive threshold
        threshold = max(mean_threshold, percentile_threshold)
        if not self.config.relative_threshold:
            threshold = self.config.activation_threshold
        
        # Update neuron histories and check for freezing
        for neuron_id, activation in enumerate(neuron_activations):
            if neuron_id not in self.activation_history[layer_name]:
                self.activation_history[layer_name][neuron_id] = []
            
            history = self.activation_history[layer_name][neuron_id]
            history.append(float(activation))
            
            # Keep fixed window size
            if len(history) > self.config.window_size:
                history.pop(0)
            
            # Check if neuron should be frozen
            if len(history) == self.config.window_size and neuron_id not in self.frozen_neurons[layer_name]:
                mean_activation = np.mean(history)
                std_activation = np.std(history)
                peak_activation = stats["neuron_peaks"][neuron_id]
                
                # Freeze if consistently low activation and stable
                should_freeze = (
                    mean_activation < threshold or
                    stats["neuron_means"][neuron_id] < percentile_threshold or
                    peak_activation < peak_threshold
                ) and std_activation < self.config.stabilization_threshold * stats["mean_activation"]
                
                # Check if we can freeze more neurons
                min_neurons = max(self.config.min_neurons, 
                                int(stats["total_neurons"] * self.config.min_active_ratio))
                can_freeze = stats["active_neurons"] > min_neurons
                
                if should_freeze and can_freeze:
                    self.frozen_neurons[layer_name].add(neuron_id)
                    stats["active_neurons"] -= 1
    
    def optimize_layer(self, layer: torch.nn.Module, name: str, next_in_features: Optional[int] = None) -> torch.nn.Module:
        """Optimize a single layer by freezing inactive neurons"""
        if not isinstance(layer, torch.nn.Linear):
            return layer
            
        # Get frozen neurons for this layer
        frozen_neurons = self.frozen_neurons.get(name, set())
        stats = self.layer_stats[name]
        
        total_neurons = stats["total_neurons"]
        active_neurons = stats["active_neurons"]
        
        # Ensure minimum number of neurons
        min_neurons = max(self.config.min_neurons, 
                         int(total_neurons * self.config.min_active_ratio))
        
        if active_neurons < min_neurons:
            logging.warning(f"Layer {name} has too few active neurons ({active_neurons}/{total_neurons})")
            # Keep the top neurons by activation
            neuron_means = stats["neuron_means"]
            top_indices = np.argsort(neuron_means)[-min_neurons:]
            frozen_neurons = set(i for i in range(total_neurons) if i not in top_indices)
            active_neurons = min_neurons
            stats["active_neurons"] = active_neurons
        
        # Create optimized layer
        optimized = torch.nn.Linear(
            layer.in_features,
            active_neurons,
            bias=layer.bias is not None,
            device=layer.weight.device
        )
        
        # Copy active weights and bias
        active_indices = [i for i in range(total_neurons) if i not in frozen_neurons]
        optimized.weight.data = layer.weight.data[active_indices]
        
        if layer.bias is not None:
            optimized.bias.data = layer.bias.data[active_indices]
        
        # Update layer dimensions for next layer
        if next_in_features is not None:
            self.layer_dims[name]["out_features"] = active_neurons
            
        return optimized
    
    def get_optimization_stats(self) -> Dict:
        """Get optimization statistics"""
        total_neurons = 0
        total_active = 0
        layer_info = {}
        
        for layer_name, stats in self.layer_stats.items():
            total_neurons += stats["total_neurons"]
            total_active += stats["active_neurons"]
            
            layer_info[layer_name] = {
                "total_neurons": stats["total_neurons"],
                "active_neurons": stats["active_neurons"],
                "compression_ratio": 1 - (stats["active_neurons"] / stats["total_neurons"]),
                "max_activation": stats["max_activation"],
                "min_activation": stats["min_activation"],
                "mean_activation": stats.get("mean_activation", 0.0),
                "std_activation": stats.get("std_activation", 0.0),
                "percentile_threshold": stats.get("percentile_threshold", 0.0)
            }
        
        return {
            "total_neurons": total_neurons,
            "active_neurons": total_active,
            "compression_ratio": 1 - (total_active / total_neurons) if total_neurons > 0 else 0,
            "layers": layer_info
        }
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize entire model by removing inactive neurons"""
        # Reset state
        self.linear_layers = []
        self.layer_names = {}
        self.activation_history.clear()
        self.frozen_neurons.clear()
        self.layer_stats.clear()
        self.step_counter = 0
        self.layer_counter = 0
        self.module_order = []
        self.layer_dims = {}
        
        # First pass: collect module order and register hooks
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.ReLU)):
                self.module_order.append((name, module))
                if isinstance(module, torch.nn.Linear):
                    layer_name = self._get_unique_layer_name(module)
                    self.register_activation_hook(module, layer_name)
        
        # Run model to collect activation statistics
        device = next(model.parameters()).device
        dummy_input = torch.randn(32, self.module_order[0][1].in_features, device=device)
        
        with torch.no_grad():
            for _ in range(self.config.window_size * 2):
                model(dummy_input)
        
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        # Build optimized model preserving layer order and dimensions
        layers = []
        prev_out_features = None
        
        for i, (name, module) in enumerate(self.module_order):
            if isinstance(module, torch.nn.Linear):
                # Get next layer's input dimension requirement
                next_in_features = None
                for next_name, next_module in self.module_order[i+1:]:
                    if isinstance(next_module, torch.nn.Linear):
                        next_in_features = next_module.in_features
                        break
                
                optimized = self.optimize_layer(module, self.layer_names[module], next_in_features)
                
                # Update input features if needed
                if prev_out_features is not None:
                    optimized.in_features = prev_out_features
                    optimized.weight = torch.nn.Parameter(
                        torch.randn(optimized.out_features, prev_out_features, 
                                  device=optimized.weight.device) * 0.02
                    )
                    if optimized.bias is not None:
                        optimized.bias = torch.nn.Parameter(
                            torch.zeros(optimized.out_features, 
                                      device=optimized.bias.device)
                        )
                
                prev_out_features = optimized.out_features
                layers.append(optimized)
            else:
                layers.append(module)
        
        # Create optimized model
        optimized_model = torch.nn.Sequential(*layers)
        
        # Log optimization results
        stats = self.get_optimization_stats()
        logging.info(f"Model optimization complete. Compression ratio: {stats['compression_ratio']:.2%}")
        
        return optimized_model 