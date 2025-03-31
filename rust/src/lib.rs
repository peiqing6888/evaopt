use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, IntoPyArray};
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

mod tensor;
mod memory;
mod matrix;

use matrix::{MatrixConfig, DecompositionMethod, optimize_matrix, BlockSparseConfig};

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
fn get_matrix_stats(
    py: Python,
    matrix: &PyArray2<f32>,
    method: &str,
    rank: Option<usize>,
    tolerance: Option<f32>,
) -> PyResult<HashMap<String, PyObject>> {
    let array = matrix.readonly();
    let shape = (array.shape()[0], array.shape()[1]);
    let matrix = Array2::from_shape_vec(shape, array.as_slice()?.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    let config = match method.to_lowercase().as_str() {
        "svd" => MatrixConfig {
            method: DecompositionMethod::SVD,
            rank: rank.unwrap_or(10),
            tolerance: tolerance.unwrap_or(1e-6),
            use_parallel: true,
            oversampling: 5,
            power_iterations: 2,
            block_sparse: None,
        },
        "low_rank" => MatrixConfig {
            method: DecompositionMethod::LowRank,
            rank: rank.unwrap_or(10),
            tolerance: tolerance.unwrap_or(1e-6),
            use_parallel: true,
            oversampling: 5,
            power_iterations: 2,
            block_sparse: None,
        },
        "sparse" => MatrixConfig {
            method: DecompositionMethod::Sparse,
            rank: rank.unwrap_or(10),
            tolerance: tolerance.unwrap_or(1e-6),
            use_parallel: true,
            oversampling: 5,
            power_iterations: 2,
            block_sparse: None,
        },
        "block_sparse" => MatrixConfig {
            method: DecompositionMethod::BlockSparse,
            rank: rank.unwrap_or(10),
            tolerance: tolerance.unwrap_or(1e-6),
            use_parallel: true,
            oversampling: 5,
            power_iterations: 2,
            block_sparse: Some(BlockSparseConfig::default()),
        },
        "truncated_svd" => MatrixConfig {
            method: DecompositionMethod::TruncatedSVD,
            rank: rank.unwrap_or(10),
            tolerance: tolerance.unwrap_or(1e-6),
            use_parallel: true,
            oversampling: 5,
            power_iterations: 2,
            block_sparse: None,
        },
        "randomized_svd" => MatrixConfig {
            method: DecompositionMethod::RandomizedSVD,
            rank: rank.unwrap_or(10),
            tolerance: tolerance.unwrap_or(1e-6),
            use_parallel: true,
            oversampling: 5,
            power_iterations: 2,
            block_sparse: None,
        },
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid method")),
    };
    
    match optimize_matrix(&matrix, &config) {
        Ok(result) => {
            let mut stats = HashMap::new();
            stats.insert("compression_ratio".to_string(), result.compression_ratio.to_object(py));
            stats.insert("error".to_string(), result.error.to_object(py));
            stats.insert("storage_size".to_string(), result.storage_size.to_object(py));
            Ok(stats)
        },
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Matrix optimization failed: {}", e)
        ))
    }
}

/// Initialize Python module
#[pymodule]
fn evaopt_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(get_matrix_stats, m)?)?;
    Ok(())
} 