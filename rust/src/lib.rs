use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use std::sync::Arc;
use std::error::Error;

pub mod tensor;
pub mod memory;
pub mod matrix;
pub mod chunk;
pub mod dynamic;

pub use matrix::{MatrixConfig, DecompositionMethod, optimize_matrix, BlockSparseConfig};
pub use chunk::{ChunkConfig, ChunkManager};
pub use memory::MemoryPool;
pub use dynamic::{DynamicOptimizer, NeuronStats, OptimizerConfig, LayerStats};

/// Tensor optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub sparsity_threshold: f32,
    pub quantization_bits: u8,
    pub use_parallel: bool,
    pub matrix_config: Option<MatrixConfig>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            sparsity_threshold: 1e-6,
            quantization_bits: 8,
            use_parallel: true,
            matrix_config: None,
        }
    }
}

/// Main function for tensor optimization
pub fn optimize_tensors(
    tensors: HashMap<String, Array2<f32>>,
    config: Option<OptimizationConfig>,
) -> Result<HashMap<String, Array2<f32>>> {
    let config = config.unwrap_or_default();
    let mut optimized = HashMap::new();
    
    for (name, mut matrix) in tensors {
        // Apply matrix optimization if configured
        if let Some(matrix_config) = &config.matrix_config {
            match optimize_matrix(&matrix, matrix_config) {
                Ok(result) => {
                    optimized.insert(name, result.compressed);
                },
                Err(e) => {
                    return Err(OptError::ComputationError(
                        format!("Matrix optimization failed: {}", e)
                    ));
                }
            }
            continue;
        }
        
        // Otherwise apply basic optimization
        optimize_array_2d(&mut matrix, &config);
        optimized.insert(name, matrix);
    }
    
    Ok(optimized)
}

/// Optimize 2D array
fn optimize_array_2d(array: &mut Array2<f32>, config: &OptimizationConfig) {
    if config.use_parallel {
        array.axis_iter_mut(Axis(0))
            .for_each(|mut row| {
                row.mapv_inplace(|x| if x.abs() < config.sparsity_threshold { 0.0 } else { x });
            });
    } else {
        array.mapv_inplace(|x| if x.abs() < config.sparsity_threshold { 0.0 } else { x });
    }
}

/// Get optimization statistics for matrix
pub fn get_matrix_stats(
    matrix: &Array2<f32>,
    method: &str,
    rank: Option<usize>,
    tolerance: Option<f32>,
) -> Result<HashMap<String, f64>> {
    // Create matrix configuration
    let matrix_config = MatrixConfig {
        method: match method {
            "svd" => DecompositionMethod::SVD,
            "low_rank" => DecompositionMethod::LowRank,
            "sparse" => DecompositionMethod::Sparse,
            "truncated_svd" => DecompositionMethod::TruncatedSVD,
            "randomized_svd" => DecompositionMethod::RandomizedSVD,
            "block_sparse" => DecompositionMethod::BlockSparse,
            "adaptive_low_rank" => DecompositionMethod::AdaptiveLowRank,
            "sparse_pattern" => DecompositionMethod::SparsePattern,
            "mixed_precision" => DecompositionMethod::MixedPrecision,
            "tensor_core_svd" => DecompositionMethod::TensorCoreSVD,
            _ => return Err(OptError::InvalidDimensions(
                format!("Unsupported method: {}", method)
            )),
        },
        rank: rank.unwrap_or(10),
        tolerance: tolerance.unwrap_or(1e-6),
        use_parallel: true,
        oversampling: 5,
        power_iterations: 2,
        block_sparse: None,
    };
    
    // Optimize matrix to calculate stats
    let result = optimize_matrix(matrix, &matrix_config)
        .map_err(|e| OptError::ComputationError(format!("Matrix optimization failed: {}", e)))?;
    
    // Create result dictionary
    let mut stats = HashMap::new();
    stats.insert("compression_ratio".to_string(), result.compression_ratio as f64 * 100.0);
    stats.insert("error".to_string(), result.error as f64);
    stats.insert("storage_size".to_string(), result.storage_size as f64);
    stats.insert("original_size".to_string(), (matrix.len() * std::mem::size_of::<f32>()) as f64);
    
    Ok(stats)
}

/// Optimize matrix in chunks
pub fn optimize_matrix_in_chunks(
    matrix: &mut Array2<f32>,
    config: ChunkConfig,
    method: &str,
    rank: Option<usize>,
    tolerance: Option<f32>
) -> Result<HashMap<String, f64>> {
    use parking_lot::RwLock;
    
    // Create memory pool
    let memory_pool = Arc::new(RwLock::new(MemoryPool::new(config.memory_limit)));
    
    // Create chunk manager
    let manager = ChunkManager::new(config.clone(), memory_pool);
    
    // Create matrix config
    let matrix_config = MatrixConfig {
        method: match method {
            "svd" => DecompositionMethod::SVD,
            "low_rank" => DecompositionMethod::LowRank,
            "sparse" => DecompositionMethod::Sparse,
            "truncated_svd" => DecompositionMethod::TruncatedSVD,
            "randomized_svd" => DecompositionMethod::RandomizedSVD,
            "block_sparse" => DecompositionMethod::BlockSparse,
            "adaptive_low_rank" => DecompositionMethod::AdaptiveLowRank,
            "sparse_pattern" => DecompositionMethod::SparsePattern,
            "mixed_precision" => DecompositionMethod::MixedPrecision,
            "tensor_core_svd" => DecompositionMethod::TensorCoreSVD,
            _ => return Err(OptError::InvalidDimensions(
                format!("Unsupported method: {}", method)
            )),
        },
        rank: rank.unwrap_or(10),
        tolerance: tolerance.unwrap_or(1e-6),
        use_parallel: config.use_parallel,
        oversampling: 5,
        power_iterations: 2,
        block_sparse: None,
    };
    
    // Process matrix in chunks
    let stats = manager.process_matrix(matrix, |chunk| {
        match optimize_matrix(chunk, &matrix_config) {
            Ok(result) => {
                chunk.assign(&result.compressed);
                Ok(())
            },
            Err(e) => Err(e.to_string()),
        }
    }).map_err(|e| OptError::ComputationError(e))?;
    
    // Calculate total stats
    let total_time: f64 = stats.iter().map(|s| s.processing_time).sum();
    let total_memory: usize = stats.iter().map(|s| s.memory_used).max().unwrap_or(0);
    
    // Create result dictionary
    let mut result = HashMap::new();
    result.insert("processing_time".to_string(), total_time);
    result.insert("memory_used".to_string(), total_memory as f64);
    
    Ok(result)
}

#[derive(Debug, thiserror::Error)]
pub enum OptError {
    #[error("Layer not found: {0}")]
    LayerNotFound(String),
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),
    #[error("Computation error: {0}")]
    ComputationError(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, OptError>;

/// Core neural network layer representation
#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub weights: Arc<Array2<f32>>,
    pub bias: Option<Arc<Array1<f32>>>,
    pub input_size: usize,
    pub output_size: usize,
}

impl Layer {
    pub fn new(
        name: String,
        weights: Array2<f32>,
        bias: Option<Array1<f32>>,
    ) -> Result<Self> {
        let (output_size, input_size) = weights.dim();
        
        if let Some(b) = &bias {
            if b.len() != output_size {
                return Err(OptError::InvalidDimensions(
                    format!("Bias size {} does not match output size {}", b.len(), output_size)
                ));
            }
        }

        Ok(Self {
            name,
            weights: Arc::new(weights),
            bias: bias.map(Arc::new),
            input_size,
            output_size,
        })
    }

    pub fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        if input.ncols() != self.input_size {
            return Err(OptError::InvalidDimensions(
                format!("Input size {} does not match layer input size {}", 
                    input.ncols(), self.input_size)
            ));
        }

        let mut output = input.dot(&self.weights.t());
        
        if let Some(bias) = &self.bias {
            for mut row in output.rows_mut() {
                row += &bias.view();
            }
        }

        Ok(output)
    }
}

/// Neural network model with dynamic optimization support
pub struct Model {
    layers: Vec<Layer>,
    optimizer: Option<DynamicOptimizer>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            optimizer: None,
        }
    }

    pub fn with_optimizer(config: OptimizerConfig) -> Self {
        Self {
            layers: Vec::new(),
            optimizer: Some(DynamicOptimizer::new(config)),
        }
    }

    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn forward(&self, input: Array2<f32>) -> Result<Array2<f32>> {
        let mut current = input;
        
        for layer in &self.layers {
            current = layer.forward(&current)?;
            
            if let Some(opt) = &self.optimizer {
                opt.update_activations(&layer.name, current.view())
                    .map_err(|e| OptError::Internal(e.to_string()))?;
            }
        }

        Ok(current)
    }

    pub fn optimize(&self) -> Result<Model> {
        let optimizer = self.optimizer.as_ref()
            .ok_or_else(|| OptError::Internal("No optimizer configured".into()))?;
        
        let mut optimized = Model::new();
        
        for layer in &self.layers {
            let (weights, bias) = optimizer.optimize_layer(
                &layer.name,
                layer.weights.view(),
                layer.bias.as_ref().map(|b| b.view()),
            ).map_err(|e| OptError::Internal(e.to_string()))?;
            
            optimized.add_layer(Layer::new(
                layer.name.clone(),
                weights,
                bias,
            )?);
        }
        
        Ok(optimized)
    }

    pub fn get_layer_stats(&self, layer_name: &str) -> Result<LayerStats> {
        let optimizer = self.optimizer.as_ref()
            .ok_or_else(|| OptError::Internal("No optimizer configured".into()))?;
        
        optimizer.get_stats(layer_name)
            .map_err(|e| OptError::Internal(e.to_string()))
    }
}