use ndarray::{Array2, Array1, Axis, s};
use ndarray_linalg::{SVD, Solve, Inverse};
use rand::Rng;
use thiserror::Error;

/// Matrix optimization errors
#[derive(Error, Debug)]
pub enum MatrixError {
    #[error("SVD decomposition failed")]
    SVDError,
    #[error("Matrix inversion failed")]
    InversionError,
    #[error("Invalid rank {rank} for matrix of shape ({rows}, {cols})")]
    InvalidRank { rank: usize, rows: usize, cols: usize },
    #[error("Invalid tolerance value: {0}")]
    InvalidTolerance(f32),
}

/// Matrix decomposition methods
#[derive(Debug, Clone, Copy)]
pub enum DecompositionMethod {
    SVD,
    LowRank,
    Sparse,
    TruncatedSVD,  // New method
    RandomizedSVD,  // New method
}

/// Matrix optimization configuration
#[derive(Debug, Clone)]
pub struct MatrixConfig {
    pub method: DecompositionMethod,
    pub rank: usize,
    pub tolerance: f32,
    pub use_parallel: bool,
    pub oversampling: usize,  // For randomized methods
    pub power_iterations: usize,  // For randomized methods
}

impl Default for MatrixConfig {
    fn default() -> Self {
        Self {
            method: DecompositionMethod::SVD,
            rank: 10,
            tolerance: 1e-6,
            use_parallel: true,
            oversampling: 5,
            power_iterations: 2,
        }
    }
}

/// Matrix optimization result
#[derive(Debug)]
pub struct OptimizationResult {
    pub compressed: Array2<f32>,
    pub compression_ratio: f32,
    pub error: f32,
    pub storage_size: usize,  // Actual storage size in bytes
}

/// Calculate compression statistics
fn get_compression_stats(
    original: &Array2<f32>,
    compressed_components: &[Array2<f32>],
    error: f32
) -> (f32, usize) {
    let original_size = original.len() * std::mem::size_of::<f32>();
    let compressed_size: usize = compressed_components
        .iter()
        .map(|x| x.len() * std::mem::size_of::<f32>())
        .sum();
    
    let ratio = 1.0 - (compressed_size as f32 / original_size as f32);
    (ratio, compressed_size)
}

/// Optimize matrix using SVD decomposition
pub fn optimize_svd(matrix: &Array2<f32>, config: &MatrixConfig) -> Result<OptimizationResult, MatrixError> {
    let (nrows, ncols) = matrix.dim();
    let rank = config.rank.min(nrows.min(ncols));
    
    if rank == 0 || rank > nrows.min(ncols) {
        return Err(MatrixError::InvalidRank {
            rank,
            rows: nrows,
            cols: ncols,
        });
    }
    
    let svd = matrix.svd(true, true)
        .map_err(|_| MatrixError::SVDError)?;
    
    let u = svd.0.unwrap();
    let s = svd.1;
    let vt = svd.2.unwrap();
    
    let s_trunc = Array2::from_diag(&s.slice(s![..rank]));
    let u_trunc = u.slice(s![.., ..rank]).to_owned();
    let vt_trunc = vt.slice(s![..rank, ..]).to_owned();
    
    let compressed = u_trunc.dot(&s_trunc).dot(&vt_trunc);
    let error = frobenius_norm(&(matrix - &compressed)) / frobenius_norm(matrix);
    
    let (compression_ratio, storage_size) = get_compression_stats(
        matrix,
        &[u_trunc, s_trunc, vt_trunc],
        error
    );
    
    Ok(OptimizationResult {
        compressed,
        compression_ratio,
        error,
        storage_size,
    })
}

/// Optimize matrix using truncated SVD (faster than full SVD for large matrices)
pub fn optimize_truncated_svd(matrix: &Array2<f32>, config: &MatrixConfig) -> Result<OptimizationResult, MatrixError> {
    let (nrows, ncols) = matrix.dim();
    let rank = config.rank.min(nrows.min(ncols));
    
    if rank == 0 || rank > nrows.min(ncols) {
        return Err(MatrixError::InvalidRank {
            rank,
            rows: nrows,
            cols: ncols,
        });
    }
    
    // Use Krylov subspace method for truncated SVD
    let mut q = Array2::<f32>::zeros((nrows, rank));
    let mut h = Array2::<f32>::zeros((rank, rank));
    
    // Initialize with random vector
    let mut v = Array2::<f32>::zeros((nrows, 1));
    v.mapv_inplace(|_| rand::random::<f32>());
    v /= frobenius_norm(&v);
    
    // Lanczos iteration
    for j in 0..rank {
        let w = if j == 0 {
            // For the first iteration, use v directly
            let temp = matrix.t().dot(&v);  // (ncols, 1)
            matrix.dot(&temp)  // (nrows, 1)
        } else {
            // For subsequent iterations, use previous q column
            let qj = q.slice(s![.., j-1..j]);  // (nrows, 1)
            let temp = matrix.t().dot(&qj);  // (ncols, 1)
            matrix.dot(&temp)  // (nrows, 1)
        };
        
        let mut w = w.to_owned();
        
        // Gram-Schmidt orthogonalization
        for i in 0..j {
            let qi = q.slice(s![.., i..i+1]);  // (nrows, 1)
            h[[i, j]] = qi.t().dot(&w)[[0, 0]];
            w = w - &(qi.mapv(|x| x * h[[i, j]]));
        }
        
        h[[j, j]] = frobenius_norm(&w);
        if h[[j, j]] > 1e-10 {
            q.slice_mut(s![.., j..j+1]).assign(&(w.mapv(|x| x / h[[j, j]])));
        }
    }
    
    // Get truncated SVD from Krylov decomposition
    let h_svd = h.svd(true, true)
        .map_err(|_| MatrixError::SVDError)?;
    
    let u_trunc = q.dot(&h_svd.0.unwrap());  // (nrows, rank)
    let s_trunc = Array2::from_diag(&h_svd.1);  // (rank, rank)
    
    // Compute V using the relationship V = (M^T * U * S^{-1})
    let s_inv = Array2::from_diag(&h_svd.1.mapv(|x| if x > 1e-10 { 1.0 / x } else { 0.0 }));
    let v_trunc = matrix.t().dot(&u_trunc).dot(&s_inv);  // (ncols, rank)
    
    let compressed = u_trunc.dot(&s_trunc).dot(&v_trunc.t());
    let error = frobenius_norm(&(matrix - &compressed)) / frobenius_norm(matrix);
    
    let (compression_ratio, storage_size) = get_compression_stats(
        matrix,
        &[u_trunc, s_trunc, v_trunc],
        error
    );
    
    Ok(OptimizationResult {
        compressed,
        compression_ratio,
        error,
        storage_size,
    })
}

/// Optimize matrix using randomized SVD
pub fn optimize_randomized_svd(matrix: &Array2<f32>, config: &MatrixConfig) -> Result<OptimizationResult, MatrixError> {
    let (nrows, ncols) = matrix.dim();
    let rank = config.rank.min(nrows.min(ncols));
    
    if rank == 0 || rank > nrows.min(ncols) {
        return Err(MatrixError::InvalidRank {
            rank,
            rows: nrows,
            cols: ncols,
        });
    }
    
    let rank_oversample = rank + config.oversampling;
    
    // Stage 1: Random projection
    let mut omega = Array2::<f32>::zeros((ncols, rank_oversample));
    omega.mapv_inplace(|_| rand::random::<f32>());
    
    let mut q = matrix.dot(&omega);
    for _ in 0..config.power_iterations {
        q = matrix.t().dot(&q);
        q = matrix.dot(&q);
    }
    
    // QR decomposition via Gram-Schmidt
    let mut q_basis = Array2::<f32>::zeros((nrows, rank_oversample));
    for i in 0..rank_oversample {
        let mut v = q.slice(s![.., i..i+1]).to_owned();
        for j in 0..i {
            let qj = q_basis.slice(s![.., j..j+1]).to_owned();
            let proj = qj.t().dot(&v)[[0, 0]];
            v = v - &(qj.mapv(|x| x * proj));
        }
        let norm = frobenius_norm(&v);
        if norm > 1e-10 {
            q_basis.slice_mut(s![.., i..i+1]).assign(&(v.mapv(|x| x / norm)));
        }
    }
    
    // Stage 2: SVD on the small matrix
    let b = q_basis.t().dot(matrix);
    let b_svd = b.svd(true, true)
        .map_err(|_| MatrixError::SVDError)?;
    
    let u_trunc = q_basis.dot(&b_svd.0.unwrap().slice(s![.., ..rank]));
    let s_trunc = Array2::from_diag(&b_svd.1.slice(s![..rank]));
    let vt_trunc = b_svd.2.unwrap().slice(s![..rank, ..]).to_owned();
    
    let compressed = u_trunc.dot(&s_trunc).dot(&vt_trunc);
    let error = frobenius_norm(&(matrix - &compressed)) / frobenius_norm(matrix);
    
    let (compression_ratio, storage_size) = get_compression_stats(
        matrix,
        &[u_trunc, s_trunc, vt_trunc],
        error
    );
    
    Ok(OptimizationResult {
        compressed,
        compression_ratio,
        error,
        storage_size,
    })
}

/// Optimize matrix using low-rank approximation
pub fn optimize_low_rank(matrix: &Array2<f32>, config: &MatrixConfig) -> Result<OptimizationResult, MatrixError> {
    let (nrows, ncols) = matrix.dim();
    let rank = config.rank.min(nrows.min(ncols));
    
    if rank == 0 || rank > nrows.min(ncols) {
        return Err(MatrixError::InvalidRank {
            rank,
            rows: nrows,
            cols: ncols,
        });
    }
    
    // Initialize with SVD for better initial guess
    let svd = matrix.svd(true, true)
        .map_err(|_| MatrixError::SVDError)?;
        
    let mut u = svd.0.unwrap().slice(s![.., ..rank]).to_owned();
    let mut v = svd.2.unwrap().slice(s![..rank, ..]).t().to_owned();
    
    // Get initial singular values
    let s = svd.1.slice(s![..rank]).to_owned();
    
    // Scale initial matrices
    for i in 0..rank {
        let s_sqrt = s[i].sqrt();
        u.column_mut(i).mapv_inplace(|x| x * s_sqrt);
        v.column_mut(i).mapv_inplace(|x| x * s_sqrt);
    }
    
    // Alternating least squares with regularization
    let lambda = config.tolerance;
    for _ in 0..50 {
        // Fix V, solve for U
        let vt_v = v.t().dot(&v) + lambda * Array2::eye(rank);
        let vt_v_inv = vt_v.inv()
            .map_err(|_| MatrixError::InversionError)?;
        let mut u_new = matrix.dot(&v).dot(&vt_v_inv);
        
        // Fix U, solve for V
        let ut_u = u_new.t().dot(&u_new) + lambda * Array2::eye(rank);
        let ut_u_inv = ut_u.inv()
            .map_err(|_| MatrixError::InversionError)?;
        let mut v_new = matrix.t().dot(&u_new).dot(&ut_u_inv);
        
        // Normalize columns
        for k in 0..rank {
            let u_norm = u_new.column(k).mapv(|x| x * x).sum().sqrt();
            let v_norm = v_new.column(k).mapv(|x| x * x).sum().sqrt();
            let scale = (u_norm * v_norm).sqrt();
            
            if scale > 1e-10 {
                u_new.column_mut(k).mapv_inplace(|x| x * (scale / u_norm));
                v_new.column_mut(k).mapv_inplace(|x| x * (scale / v_norm));
            }
        }
        
        u = u_new;
        v = v_new;
    }
    
    let compressed = u.dot(&v.t());
    let error = frobenius_norm(&(matrix - &compressed)) / frobenius_norm(matrix);
    
    let (compression_ratio, storage_size) = get_compression_stats(
        matrix,
        &[u, v],
        error
    );
    
    Ok(OptimizationResult {
        compressed,
        compression_ratio,
        error,
        storage_size,
    })
}

/// Optimize sparse matrix
pub fn optimize_sparse(matrix: &Array2<f32>, config: &MatrixConfig) -> Result<OptimizationResult, MatrixError> {
    if config.tolerance <= 0.0 {
        return Err(MatrixError::InvalidTolerance(config.tolerance));
    }
    
    let mut compressed = matrix.clone();
    let threshold = config.tolerance;
    
    compressed.mapv_inplace(|x| if x.abs() < threshold { 0.0 } else { x });
    
    let nnz = compressed.iter().filter(|&&x| x != 0.0).count();
    let total = compressed.len();
    
    Ok(OptimizationResult {
        compressed,
        compression_ratio: 1.0 - (nnz as f32 / total as f32),
        error: threshold,
        storage_size: total * std::mem::size_of::<f32>(),
    })
}

/// Calculate Frobenius norm of a matrix
fn frobenius_norm(matrix: &Array2<f32>) -> f32 {
    matrix.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Optimize matrix using specified method
pub fn optimize_matrix(matrix: &Array2<f32>, config: &MatrixConfig) -> OptimizationResult {
    let result = match config.method {
        DecompositionMethod::SVD => optimize_svd(matrix, config),
        DecompositionMethod::TruncatedSVD => optimize_truncated_svd(matrix, config),
        DecompositionMethod::RandomizedSVD => optimize_randomized_svd(matrix, config),
        DecompositionMethod::LowRank => optimize_low_rank(matrix, config),
        DecompositionMethod::Sparse => optimize_sparse(matrix, config),
    };
    
    match result {
        Ok(opt_result) => opt_result,
        Err(e) => {
            eprintln!("Matrix optimization error: {}", e);
            // Fall back to sparse optimization with default config
            let fallback_config = MatrixConfig {
                method: DecompositionMethod::Sparse,
                rank: config.rank,
                tolerance: 1e-6,
                use_parallel: config.use_parallel,
                oversampling: config.oversampling,
                power_iterations: config.power_iterations,
            };
            optimize_sparse(matrix, &fallback_config).unwrap_or_else(|_| OptimizationResult {
                compressed: matrix.clone(),
                compression_ratio: 0.0,
                error: 0.0,
                storage_size: matrix.len() * std::mem::size_of::<f32>(),
            })
        }
    }
} 