use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use rayon::prelude::*;
use crate::OptError;
use crate::Result;

// Cache line size in bytes (typical for most modern CPUs)
const CACHE_LINE_SIZE: usize = 64;
// L1 cache size (conservative estimate)
const L1_CACHE_SIZE: usize = 32 * 1024;

/// Calculate optimal chunk size based on data dimensions and CPU cache
#[inline]
fn determine_optimal_chunk_size(rows: usize, cols: usize) -> usize {
    let elements_per_cache_line = CACHE_LINE_SIZE / std::mem::size_of::<f32>();
    let rows_per_chunk = L1_CACHE_SIZE / (cols * std::mem::size_of::<f32>());
    rows_per_chunk.max(elements_per_cache_line).min(rows)
}

#[derive(Clone, Debug)]
pub struct NeuronStats {
    mean_activation: f32,
    peak_activation: f32,
    active_ratio: f32,
    last_update: u64,
    frozen: bool,
}

impl NeuronStats {
    pub fn new() -> Self {
        Self {
            mean_activation: 0.0,
            peak_activation: 0.0,
            active_ratio: 1.0,
            last_update: 0,
            frozen: false,
        }
    }
}

pub struct DynamicOptimizer {
    activation_threshold: f32,
    min_active_ratio: f32,
    update_frequency: u64,
    ema_alpha: f32,
    step_counter: Arc<Mutex<u64>>,
    neuron_stats: Arc<Mutex<HashMap<String, Vec<NeuronStats>>>>,
    cache_enabled: bool,
}

impl DynamicOptimizer {
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            activation_threshold: config.activation_threshold,
            min_active_ratio: config.min_active_ratio,
            update_frequency: config.update_frequency,
            ema_alpha: config.ema_alpha,
            step_counter: Arc::new(Mutex::new(0)),
            neuron_stats: Arc::new(Mutex::new(HashMap::new())),
            cache_enabled: config.cache_enabled,
        }
    }

    pub fn update_activations(&self, layer_name: &str, activations: ArrayView2<f32>) -> Result<()> {
        let mut step = self.step_counter.lock()
            .map_err(|e| OptError::Internal(e.to_string()))?;
        
        let n_neurons = activations.shape()[1];
        let batch_size = activations.shape()[0] as f32;

        // 使用向量存储计算结果
        let mut means = vec![0.0f32; n_neurons];
        let mut peaks = vec![0.0f32; n_neurons];

        // 使用自适应块大小进行批处理
        let chunk_size = determine_optimal_chunk_size(activations.shape()[0], n_neurons);
        
        // 并行处理数据块
        activations.axis_chunks_iter(Axis(0), chunk_size)
            .into_par_iter()
            .for_each(|chunk| {
                let mut local_means = vec![0.0f32; n_neurons];
                let mut local_peaks = vec![0.0f32; n_neurons];
                
                chunk.axis_iter(Axis(0)).for_each(|row| {
                    for (i, &val) in row.iter().enumerate() {
                        let abs_val = val.abs();
                        local_means[i] += abs_val;
                        local_peaks[i] = local_peaks[i].max(abs_val);
                    }
                });
                
                // 合并结果到全局数组
                means.par_iter_mut().zip(local_means.par_iter())
                    .for_each(|(mean, &local)| {
                        *mean += local;
                    });
                    
                peaks.par_iter_mut().zip(local_peaks.par_iter())
                    .for_each(|(peak, &local)| {
                        *peak = peak.max(local);
                    });
            });

        // 计算最终的平均值
        means.par_iter_mut().for_each(|mean| {
            *mean /= batch_size;
        });

        let mut stats = self.neuron_stats.lock()
            .map_err(|e| OptError::Internal(e.to_string()))?;
        let layer_stats = stats
            .entry(layer_name.to_string())
            .or_insert_with(|| vec![NeuronStats::new(); n_neurons]);

        // 只在更新频率匹配时更新统计信息
        if *step % self.update_frequency == 0 {
            layer_stats.par_iter_mut().enumerate().for_each(|(i, stat)| {
                if !stat.frozen {
                    stat.mean_activation = stat.mean_activation * (1.0 - self.ema_alpha)
                        + means[i] * self.ema_alpha;
                    stat.peak_activation = stat.peak_activation.max(peaks[i]);
                    
                    let is_active = stat.mean_activation >= self.activation_threshold
                        || stat.peak_activation >= self.activation_threshold * 2.0;
                    stat.active_ratio = stat.active_ratio * (1.0 - self.ema_alpha)
                        + if is_active { 1.0 } else { 0.0 } * self.ema_alpha;
                    stat.last_update = *step;
                }
            });
        }

        *step += 1;
        Ok(())
    }

    pub fn get_active_neurons(&self, layer_name: &str) -> Result<Vec<usize>> {
        let stats = self.neuron_stats.lock()
            .map_err(|e| OptError::Internal(e.to_string()))?;
        let layer_stats = stats.get(layer_name)
            .ok_or_else(|| OptError::LayerNotFound(layer_name.to_string()))?;

        let mut active_neurons: Vec<usize> = layer_stats.par_iter()
            .enumerate()
            .filter(|(_, stat)| !stat.frozen && stat.active_ratio >= self.min_active_ratio)
            .map(|(i, _)| i)
            .collect();

        let total_neurons = layer_stats.len();
        let min_neurons = (total_neurons as f32 * self.min_active_ratio).ceil() as usize;

        if active_neurons.len() < min_neurons {
            let mut neuron_scores: Vec<(usize, f32)> = layer_stats
                .par_iter()
                .enumerate()
                .filter(|(_, stat)| !stat.frozen)
                .map(|(i, s)| (i, s.mean_activation))
                .collect();
            
            neuron_scores.par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (i, _) in neuron_scores.iter().take(min_neurons) {
                if !active_neurons.contains(i) {
                    active_neurons.push(*i);
                }
            }
        }

        Ok(active_neurons)
    }

    pub fn optimize_layer(
        &self,
        layer_name: &str,
        weights: ArrayView2<f32>,
        bias: Option<ArrayView1<f32>>,
    ) -> Result<(Array2<f32>, Option<Array1<f32>>)> {
        let active_neurons = self.get_active_neurons(layer_name)?;
        let in_features = weights.shape()[1];
        
        let optimized_weights = Array2::from_shape_fn(
            (active_neurons.len(), in_features),
            |(i, j)| weights[[active_neurons[i], j]]
        );

        let optimized_bias = bias.map(|b| {
            Array1::from_iter(active_neurons.iter().map(|&i| b[i]))
        });

        Ok((optimized_weights, optimized_bias))
    }

    pub fn freeze_neurons(&self, layer_name: &str, neuron_indices: &[usize]) -> Result<()> {
        let mut stats = self.neuron_stats.lock()
            .map_err(|e| OptError::Internal(e.to_string()))?;
        let layer_stats = stats.get_mut(layer_name)
            .ok_or_else(|| OptError::LayerNotFound(layer_name.to_string()))?;

        for &idx in neuron_indices {
            if idx < layer_stats.len() {
                layer_stats[idx].frozen = true;
            }
        }
        Ok(())
    }

    pub fn unfreeze_neurons(&self, layer_name: &str, neuron_indices: &[usize]) -> Result<()> {
        let mut stats = self.neuron_stats.lock()
            .map_err(|e| OptError::Internal(e.to_string()))?;
        let layer_stats = stats.get_mut(layer_name)
            .ok_or_else(|| OptError::LayerNotFound(layer_name.to_string()))?;

        for &idx in neuron_indices {
            if idx < layer_stats.len() {
                layer_stats[idx].frozen = false;
            }
        }
        Ok(())
    }

    pub fn get_stats(&self, layer_name: &str) -> Result<LayerStats> {
        let stats = self.neuron_stats.lock()
            .map_err(|e| OptError::Internal(e.to_string()))?;
        let layer_stats = stats.get(layer_name)
            .ok_or_else(|| OptError::LayerNotFound(layer_name.to_string()))?;

        let total_neurons = layer_stats.len();
        let active_neurons = layer_stats.iter()
            .filter(|stat| !stat.frozen && stat.active_ratio >= self.min_active_ratio)
            .count();
        let frozen_neurons = layer_stats.iter()
            .filter(|stat| stat.frozen)
            .count();
        let compression_ratio = 1.0 - (active_neurons as f32 / total_neurons as f32);

        Ok(LayerStats {
            total_neurons,
            active_neurons,
            frozen_neurons,
            compression_ratio,
        })
    }
}

#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub activation_threshold: f32,
    pub min_active_ratio: f32,
    pub update_frequency: u64,
    pub ema_alpha: f32,
    pub cache_enabled: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            activation_threshold: 0.1,
            min_active_ratio: 0.3,
            update_frequency: 10,
            ema_alpha: 0.1,
            cache_enabled: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerStats {
    pub total_neurons: usize,
    pub active_neurons: usize,
    pub frozen_neurons: usize,
    pub compression_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    fn create_test_optimizer() -> DynamicOptimizer {
        DynamicOptimizer::new(OptimizerConfig {
            activation_threshold: 0.1,
            min_active_ratio: 0.5,
            update_frequency: 1,
            ema_alpha: 0.1,
            cache_enabled: true,
        })
    }

    #[test]
    fn test_layer_not_found() {
        let optimizer = create_test_optimizer();
        
        // Test get_active_neurons
        match optimizer.get_active_neurons("nonexistent_layer") {
            Err(OptError::LayerNotFound(msg)) => {
                assert!(msg.contains("nonexistent_layer"));
            }
            other => panic!("Expected LayerNotFound error, got {:?}", other),
        }

        // Test get_stats
        match optimizer.get_stats("nonexistent_layer") {
            Err(OptError::LayerNotFound(msg)) => {
                assert!(msg.contains("nonexistent_layer"));
            }
            other => panic!("Expected LayerNotFound error, got {:?}", other),
        }

        // Test freeze_neurons
        match optimizer.freeze_neurons("nonexistent_layer", &[0]) {
            Err(OptError::LayerNotFound(msg)) => {
                assert!(msg.contains("nonexistent_layer"));
            }
            other => panic!("Expected LayerNotFound error, got {:?}", other),
        }
    }

    #[test]
    fn test_neuron_activation_tracking() {
        let optimizer = create_test_optimizer();
        let layer_name = "test_layer";
        
        // Create test activations
        let activations = Array2::from_shape_fn((10, 4), |(i, j)| {
            if j == 0 {
                1.0 // Always active
            } else if j == 1 {
                0.05 // Below threshold
            } else if j == 2 {
                if i < 5 { 0.2 } else { 0.0 } // Sometimes active
            } else {
                0.0 // Never active
            }
        });

        // Update activations
        optimizer.update_activations(layer_name, activations.view())
            .expect("Failed to update activations");

        // Get stats
        let stats = optimizer.get_stats(layer_name).expect("Failed to get stats");
        
        assert_eq!(stats.total_neurons, 4);
        assert!(stats.active_neurons <= 2, "Expected at most 2 active neurons");
        assert_eq!(stats.frozen_neurons, 0);
    }

    #[test]
    fn test_neuron_freezing() {
        let optimizer = create_test_optimizer();
        let layer_name = "test_layer";
        
        // Create test activations and update them
        let activations = Array2::from_shape_fn((10, 4), |_| 1.0);
        optimizer.update_activations(layer_name, activations.view())
            .expect("Failed to update activations");

        // Freeze some neurons
        optimizer.freeze_neurons(layer_name, &[0, 2])
            .expect("Failed to freeze neurons");

        // Get stats
        let stats = optimizer.get_stats(layer_name).expect("Failed to get stats");
        assert_eq!(stats.frozen_neurons, 2);

        // Unfreeze one neuron
        optimizer.unfreeze_neurons(layer_name, &[0])
            .expect("Failed to unfreeze neurons");

        // Check stats again
        let stats = optimizer.get_stats(layer_name).expect("Failed to get stats");
        assert_eq!(stats.frozen_neurons, 1);
    }

    #[test]
    fn test_layer_optimization() {
        let optimizer = create_test_optimizer();
        let layer_name = "test_layer";
        
        // Create test activations with some neurons below threshold
        let activations = Array2::from_shape_fn((10, 4), |(_, j)| {
            if j < 2 { 1.0 } else { 0.0 }
        });

        // Update activations
        optimizer.update_activations(layer_name, activations.view())
            .expect("Failed to update activations");

        // Create test weights and bias
        let weights = Array2::from_shape_fn((4, 8), |_| 1.0);
        let bias = Array1::from_vec(vec![1.0; 4]);

        // Optimize layer
        let (optimized_weights, optimized_bias) = optimizer.optimize_layer(
            layer_name,
            weights.view(),
            Some(bias.view())
        ).expect("Failed to optimize layer");

        // Check dimensions
        assert_eq!(optimized_weights.shape()[1], 8, "Input features should be preserved");
        assert!(optimized_weights.shape()[0] <= 4, "Should have fewer or equal neurons");
        assert_eq!(
            optimized_weights.shape()[0],
            optimized_bias.as_ref().map_or(0, |b| b.len()),
            "Bias size should match number of neurons"
        );
    }

    #[test]
    fn test_minimum_active_neurons() {
        let optimizer = create_test_optimizer();
        let layer_name = "test_layer";
        
        // Create test activations with all neurons below threshold
        let activations = Array2::from_shape_fn((10, 4), |_| 0.05);

        // Update activations
        optimizer.update_activations(layer_name, activations.view())
            .expect("Failed to update activations");

        // Get active neurons
        let active_neurons = optimizer.get_active_neurons(layer_name)
            .expect("Failed to get active neurons");

        // Should have minimum number of neurons active
        let min_neurons = (4.0 * optimizer.min_active_ratio).ceil() as usize;
        assert_eq!(active_neurons.len(), min_neurons);
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;
        
        let optimizer = Arc::new(create_test_optimizer());
        let layer_name = "test_layer";
        
        // Create test activations
        let activations = Arc::new(Array2::from_shape_fn((10, 4), |_| 1.0));

        // Spawn multiple threads to update activations
        let mut handles = vec![];
        for _ in 0..4 {
            let opt = optimizer.clone();
            let acts = activations.clone();
            let name = layer_name.to_string();
            
            handles.push(thread::spawn(move || {
                for _ in 0..10 {
                    opt.update_activations(&name, acts.view())
                        .expect("Failed to update activations");
                }
            }));
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify final state
        let stats = optimizer.get_stats(layer_name).expect("Failed to get stats");
        assert_eq!(stats.total_neurons, 4);
        assert!(stats.active_neurons > 0);
    }
} 