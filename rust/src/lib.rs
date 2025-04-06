use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::wrap_pyfunction;
use pyo3::Python;
use numpy::{PyArray1, PyArray2, IntoPyArray, PyReadonlyArray2};
use ndarray::{Array1, Array2, Axis, ShapeBuilder};
use std::collections::HashMap;

mod tensor;
mod memory;
mod matrix;
mod chunk;
mod dynamic;

use matrix::{MatrixConfig, DecompositionMethod, optimize_matrix, BlockSparseConfig};
use chunk::{ChunkConfig, ChunkManager};
use memory::MemoryPool;
use std::sync::Arc;
use parking_lot::RwLock;
use dynamic::{DynamicOptimizer, NeuronStats};

/// Tensor optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    sparsity_threshold: f32,
    quantization_bits: u8,
    use_parallel: bool,
    matrix_config: Option<MatrixConfig>,
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
#[pyfunction]
fn optimize_tensors(
    py: Python,
    tensors: HashMap<String, PyObject>,
    config: Option<HashMap<String, PyObject>>,
) -> PyResult<HashMap<String, PyObject>> {
    let config = parse_config(py, config)?;
    let mut optimized: HashMap<String, PyObject> = HashMap::new();
    
    for (name, tensor) in tensors {
        // Try to extract as 2D array first
        if let Ok(array) = tensor.extract::<&PyArray2<f32>>(py) {
            let matrix_view = array.readonly();
            let shape = (matrix_view.shape()[0], matrix_view.shape()[1]);
            let mut matrix = Array2::from_shape_vec(shape, matrix_view.as_slice()?.to_vec())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            
            // Apply matrix optimization if configured
            if let Some(matrix_config) = &config.matrix_config {
                match optimize_matrix(&matrix, matrix_config) {
                    Ok(result) => {
                        let py_array = result.compressed.into_pyarray(py);
                        optimized.insert(name, py_array.to_object(py));
                    },
                    Err(e) => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Matrix optimization failed: {}", e)
                        ));
                    }
                }
                continue;
            }
            
            // Otherwise apply basic optimization
            optimize_array_2d(&mut matrix, &config);
            let py_array = matrix.into_pyarray(py);
            optimized.insert(name, py_array.to_object(py));
            continue;
        }
        
        // Try to extract as 1D array
        if let Ok(array) = tensor.extract::<&PyArray1<f32>>(py) {
            let view = array.readonly();
            let mut vector = Array1::from_vec(view.as_slice()?.to_vec());
            
            // Apply basic optimization for 1D arrays
            optimize_array_1d(&mut vector, &config);
            let py_array = vector.into_pyarray(py);
            optimized.insert(name, py_array.to_object(py));
            continue;
        }
        
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unsupported tensor type for {}", name)
        ));
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

/// Optimize 1D array
fn optimize_array_1d(array: &mut Array1<f32>, config: &OptimizationConfig) {
    array.mapv_inplace(|x| if x.abs() < config.sparsity_threshold { 0.0 } else { x });
}

/// Parse optimization configuration from Python dict
fn parse_config(py: Python, config: Option<HashMap<String, PyObject>>) -> PyResult<OptimizationConfig> {
    if let Some(config_map) = config {
        let mut opt_config = OptimizationConfig::default();
        
        if let Some(threshold) = config_map.get("sparsity_threshold") {
            if let Ok(val) = threshold.extract::<f32>(py) {
                opt_config.sparsity_threshold = val;
            }
        }
        
        if let Some(bits) = config_map.get("quantization_bits") {
            if let Ok(val) = bits.extract::<u8>(py) {
                opt_config.quantization_bits = val;
            }
        }
        
        if let Some(parallel) = config_map.get("use_parallel") {
            if let Ok(val) = parallel.extract::<bool>(py) {
                opt_config.use_parallel = val;
            }
        }
        
        // Parse matrix optimization config
        if let Some(matrix_method) = config_map.get("matrix_method") {
            if let Ok(method_str) = matrix_method.extract::<String>(py) {
                let method = match method_str.to_lowercase().as_str() {
                    "svd" => DecompositionMethod::SVD,
                    "low_rank" => DecompositionMethod::LowRank,
                    "sparse" => DecompositionMethod::Sparse,
                    "truncated_svd" => DecompositionMethod::TruncatedSVD,
                    "randomized_svd" => DecompositionMethod::RandomizedSVD,
                    "block_sparse" => DecompositionMethod::BlockSparse,
                    _ => return Ok(opt_config),
                };
                
                let rank = config_map.get("matrix_rank")
                    .and_then(|x| x.extract::<usize>(py).ok())
                    .unwrap_or(10);
                
                let tolerance = config_map.get("matrix_tolerance")
                    .and_then(|x| x.extract::<f32>(py).ok())
                    .unwrap_or(1e-6);
                
                let oversampling = config_map.get("oversampling")
                    .and_then(|x| x.extract::<usize>(py).ok())
                    .unwrap_or(5);
                
                let power_iterations = config_map.get("power_iterations")
                    .and_then(|x| x.extract::<usize>(py).ok())
                    .unwrap_or(2);
                
                // Parse block-sparse config if method is block_sparse
                let block_sparse = if method_str.to_lowercase() == "block_sparse" {
                    let block_config = config_map.get("block_sparse")
                        .and_then(|x| x.extract::<HashMap<String, PyObject>>(py).ok());
                    
                    if let Some(block_map) = block_config {
                        Some(BlockSparseConfig {
                            block_size: block_map.get("block_size")
                                .and_then(|x| x.extract::<usize>(py).ok())
                                .unwrap_or(32),
                            sparsity_threshold: block_map.get("sparsity_threshold")
                                .and_then(|x| x.extract::<f32>(py).ok())
                                .unwrap_or(0.3),
                            min_block_norm: block_map.get("min_block_norm")
                                .and_then(|x| x.extract::<f32>(py).ok())
                                .unwrap_or(1e-6),
                        })
                    } else {
                        Some(BlockSparseConfig::default())
                    }
                } else {
                    None
                };
                
                opt_config.matrix_config = Some(MatrixConfig {
                    method,
                    rank,
                    tolerance,
                    use_parallel: opt_config.use_parallel,
                    oversampling,
                    power_iterations,
                    block_sparse,
                });
            }
        }
        
        Ok(opt_config)
    } else {
        Ok(OptimizationConfig::default())
    }
}

/// Get optimization statistics for matrix
#[pyfunction]
#[pyo3(name = "get_matrix_stats")]
fn get_matrix_stats(
    py: Python,
    matrix: &PyArray2<f32>,
    method: &str,
    rank: Option<usize>,
    tolerance: Option<f32>,
) -> PyResult<HashMap<String, PyObject>> {
    // Convert input to ndarray
    let array = matrix.readonly();
    let shape = (array.shape()[0], array.shape()[1]);
    let matrix_data = Array2::from_shape_vec(shape, array.as_slice()?.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
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
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
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
    let result = optimize_matrix(&matrix_data, &matrix_config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Matrix optimization failed: {}", e)
        ))?;
    
    // Create result dictionary
    let mut stats = HashMap::new();
    stats.insert("compression_ratio".to_string(), (result.compression_ratio * 100.0).to_object(py));
    stats.insert("error".to_string(), result.error.to_object(py));
    stats.insert("storage_size".to_string(), result.storage_size.to_object(py));
    stats.insert("original_size".to_string(), (matrix_data.len() * std::mem::size_of::<f32>()).to_object(py));
    
    Ok(stats)
}

/// Optimize matrix in chunks
#[pyfunction]
#[pyo3(name = "optimize_matrix_in_chunks")]
pub fn optimize_matrix_in_chunks(
    py: Python,
    array: PyReadonlyArray2<f32>,
    config: ChunkConfig,
    method: &str,
    rank: Option<usize>,
    tolerance: Option<f32>
) -> PyResult<HashMap<String, PyObject>> {
    // Create memory pool
    let memory_pool = Arc::new(RwLock::new(MemoryPool::new(config.memory_limit)));
    
    // Create chunk manager
    let manager = ChunkManager::new(config.clone(), memory_pool);
    
    // Convert input array to ndarray
    let shape = (array.shape()[0], array.shape()[1]);
    let mut matrix_data = Array2::from_shape_vec(shape, array.as_slice()?.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
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
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
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
    let stats = manager.process_matrix(&mut matrix_data, |chunk| {
        match optimize_matrix(chunk, &matrix_config) {
            Ok(result) => {
                chunk.assign(&result.compressed);
                Ok(())
            },
            Err(e) => Err(e.to_string()),
        }
    }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    
    // Calculate total stats
    let total_time: f64 = stats.iter().map(|s| s.processing_time).sum();
    let total_memory: usize = stats.iter().map(|s| s.memory_used).max().unwrap_or(0);
    
    // Create result dictionary
    let mut result = HashMap::new();
    result.insert("optimized_matrix".to_string(), matrix_data.into_pyarray(py).to_object(py));
    result.insert("processing_time".to_string(), total_time.to_object(py));
    result.insert("memory_used".to_string(), total_memory.to_object(py));
    
    Ok(result)
}

/// Get information about available optimization methods
#[pyfunction]
fn get_available_methods(py: Python) -> PyResult<HashMap<String, PyObject>> {
    let mut methods = HashMap::new();
    
    // Add method info
    let add_method = |methods: &mut HashMap<String, PyObject>, name: &str, 
                     description: &str, best_for: &str, params: Vec<(&str, &str)>| {
        let method_info = PyDict::new(py);
        method_info.set_item("description", description)?;
        method_info.set_item("best_for", best_for)?;
        
        let params_dict = PyDict::new(py);
        for (param_name, param_desc) in params {
            params_dict.set_item(param_name, param_desc)?;
        }
        method_info.set_item("parameters", params_dict)?;
        
        methods.insert(name.to_string(), method_info.to_object(py));
        Ok::<(), PyErr>(())
    };
    
    // Add each method
    add_method(&mut methods, "svd", 
        "Full Singular Value Decomposition", 
        "Small to medium matrices with high accuracy requirements",
        vec![
            ("rank", "Number of singular values to keep"),
            ("tolerance", "Error tolerance"),
        ])?;
        
    add_method(&mut methods, "truncated_svd", 
        "Truncated SVD using Krylov subspace methods", 
        "Large matrices where full SVD is too expensive",
        vec![
            ("rank", "Number of singular values to keep"),
            ("tolerance", "Error tolerance"),
        ])?;
    
    add_method(&mut methods, "randomized_svd", 
        "Randomized SVD algorithm", 
        "Very large matrices, faster but less accurate than truncated SVD",
        vec![
            ("rank", "Number of singular values to keep"),
            ("tolerance", "Error tolerance"),
            ("oversampling", "Oversampling parameter for randomized algorithm"),
            ("power_iterations", "Number of power iterations for accuracy"),
        ])?;
    
    add_method(&mut methods, "low_rank", 
        "Low rank approximation using alternating least squares", 
        "Matrices with known low-rank structure",
        vec![
            ("rank", "Target rank for approximation"),
            ("tolerance", "Error tolerance"),
        ])?;
    
    add_method(&mut methods, "sparse", 
        "Sparse matrix optimization", 
        "Matrices with many near-zero elements",
        vec![
            ("tolerance", "Sparsity threshold (elements below are set to zero)"),
        ])?;
    
    add_method(&mut methods, "block_sparse", 
        "Block-sparse matrix optimization", 
        "Matrices with block structure",
        vec![
            ("block_size", "Size of blocks (e.g., 16, 32, 64)"),
            ("sparsity_threshold", "Threshold for block elimination"),
            ("min_block_norm", "Minimum block norm to preserve"),
        ])?;
    
    add_method(&mut methods, "adaptive_low_rank", 
        "Adaptive low-rank approximation that automatically selects optimal rank", 
        "When optimal rank is unknown",
        vec![
            ("max_rank", "Maximum rank to consider"),
            ("error_threshold", "Target error threshold"),
            ("min_compression_ratio", "Minimum acceptable compression ratio"),
        ])?;
    
    add_method(&mut methods, "sparse_pattern", 
        "Optimization based on identifying common sparse patterns", 
        "Structured sparsity in neural networks",
        vec![
            ("global_threshold", "Global sparsity threshold"),
            ("pattern_threshold", "Pattern similarity threshold"),
            ("block_size", "Analysis block size"),
        ])?;
    
    add_method(&mut methods, "mixed_precision", 
        "Mixed precision optimization using different bit widths", 
        "Trading precision for memory savings",
        vec![
            ("high_precision_threshold", "Threshold for high precision elements"),
            ("mid_precision_threshold", "Threshold for medium precision elements"),
            ("high_precision_bits", "Bits for high precision (default: 32)"),
            ("mid_precision_bits", "Bits for medium precision (default: 16)"),
            ("low_precision_bits", "Bits for low precision (default: 8)"),
        ])?;
    
    add_method(&mut methods, "tensor_core_svd", 
        "SVD optimized for tensor core hardware acceleration", 
        "Hardware with tensor cores (Apple Silicon, NVIDIA GPUs)",
        vec![
            ("tile_size", "Tile size for tensor core processing"),
            ("precision", "Working precision (16 or 32)"),
            ("max_iter", "Maximum iterations for power method"),
            ("use_fp16", "Whether to use FP16 acceleration"),
        ])?;
    
    Ok(methods)
}

/// Initialize Python module
#[pymodule]
#[pyo3(name = "evaopt_core")]
fn init_module(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register ChunkConfig class
    m.add_class::<ChunkConfig>()?;
    m.add_class::<DynamicOptimizer>()?;
    m.add_class::<NeuronStats>()?;

    // Register functions
    m.add_function(wrap_pyfunction!(optimize_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(get_matrix_stats, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_matrix_in_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(get_available_methods, m)?)?;

    // Add module docstring
    let dict = m.dict();
    dict.set_item("__doc__", "High-performance optimization engine core")?;

    // Add version
    dict.set_item("__version__", "0.1.0")?;

    // Add all exported items
    dict.set_item(
        "__all__",
        vec![
            "optimize_matrix_in_chunks",
            "ChunkConfig",
            "DynamicOptimizer",
            "NeuronStats",
            "optimize_tensors",
            "get_matrix_stats",
            "get_available_methods",
        ],
    )?;

    Ok(())
}