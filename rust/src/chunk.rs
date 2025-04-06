use ndarray::{Array2, Axis, s};
use std::sync::Arc;
use parking_lot::RwLock;
use rayon::prelude::*;
use crate::memory::MemoryPool;

#[derive(Debug, Clone)]
pub struct ChunkStats {
    pub processing_time: f64,
    pub memory_used: usize,
    pub chunk_size: usize,
}

/// Configuration for chunk-based processing
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    pub chunk_size: usize,
    pub use_parallel: bool,
    pub memory_limit: usize,
    pub prefetch_size: usize,
    pub use_simd: bool,
}

impl ChunkConfig {
    pub fn new(
        chunk_size: Option<usize>,
        use_parallel: Option<bool>,
        memory_limit: Option<usize>,
        prefetch_size: Option<usize>,
        use_simd: Option<bool>
    ) -> Self {
        let default = Self::default();
        Self {
            chunk_size: chunk_size.unwrap_or(default.chunk_size),
            use_parallel: use_parallel.unwrap_or(default.use_parallel),
            memory_limit: memory_limit.unwrap_or(default.memory_limit),
            prefetch_size: prefetch_size.unwrap_or(default.prefetch_size),
            use_simd: use_simd.unwrap_or(default.use_simd),
        }
    }
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024,
            use_parallel: true,
            memory_limit: 1 << 30, // 1GB
            prefetch_size: 2,
            use_simd: true,
        }
    }
}

/// A chunk of a large matrix with optimized memory layout
#[derive(Debug)]
pub struct MatrixChunk {
    /// The actual data with aligned memory
    pub data: Array2<f32>,
    /// Starting row index in the original matrix
    pub start_row: usize,
    /// Whether this chunk has been modified
    pub modified: bool,
    /// Statistics for this chunk
    pub stats: Option<ChunkStats>,
}

/// Manager for chunk-based matrix operations with optimizations
pub struct ChunkManager {
    config: ChunkConfig,
    memory_pool: Arc<RwLock<MemoryPool>>,
    active_chunks: RwLock<Vec<MatrixChunk>>,
    stats: RwLock<Vec<ChunkStats>>,
}

impl ChunkManager {
    /// Create a new optimized chunk manager
    pub fn new(config: ChunkConfig, memory_pool: Arc<RwLock<MemoryPool>>) -> Self {
        Self {
            config,
            memory_pool,
            active_chunks: RwLock::new(Vec::new()),
            stats: RwLock::new(Vec::new()),
        }
    }
    
    /// Process matrix in chunks with optimizations
    pub fn process_matrix<F>(&self, matrix: &mut Array2<f32>, f: F) -> Result<Vec<ChunkStats>, String>
    where
        F: Fn(&mut Array2<f32>) -> Result<(), String> + Send + Sync + Clone,
    {
        let total_rows = matrix.shape()[0];
        let chunks = (total_rows + self.config.chunk_size - 1) / self.config.chunk_size;
        let mut stats = Vec::new();
        
        if self.config.use_parallel {
            // Create aligned chunks for better memory access
            let chunk_data: Vec<_> = (0..chunks)
                .into_par_iter()
                .map(|i| {
                    let start = i * self.config.chunk_size;
                    let end = (start + self.config.chunk_size).min(total_rows);
                    let mut chunk = matrix.slice(s![start..end, ..]).to_owned();
                    
                    let start_time = std::time::Instant::now();
                    let result = f(&mut chunk);
                    let elapsed = start_time.elapsed().as_secs_f64();
                    
                    result.map(|_| {
                        let chunk_size = chunk.len() * std::mem::size_of::<f32>();
                        (chunk, ChunkStats {
                            processing_time: elapsed,
                            memory_used: chunk_size,
                            chunk_size: end - start,
                        })
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            
            // Write back processed chunks and collect stats
            for (i, (chunk, chunk_stats)) in chunk_data.into_iter().enumerate() {
                let start = i * self.config.chunk_size;
                let end = (start + chunk.shape()[0]).min(total_rows);
                matrix.slice_mut(s![start..end, ..]).assign(&chunk);
                stats.push(chunk_stats);
            }
        } else {
            // Process chunks sequentially with prefetching
            let mut prefetch_buffer = Vec::with_capacity(self.config.prefetch_size);
            
            for i in 0..chunks {
                let start = i * self.config.chunk_size;
                let end = (start + self.config.chunk_size).min(total_rows);
                
                // Prefetch next chunks
                if prefetch_buffer.len() < self.config.prefetch_size && i + 1 < chunks {
                    let next_start = (i + 1) * self.config.chunk_size;
                    let next_end = (next_start + self.config.chunk_size).min(total_rows);
                    prefetch_buffer.push(matrix.slice(s![next_start..next_end, ..]).to_owned());
                }
                
                let mut chunk = matrix.slice(s![start..end, ..]).to_owned();
                let chunk_size = chunk.len() * std::mem::size_of::<f32>();
                let start_time = std::time::Instant::now();
                f(&mut chunk)?;
                let elapsed = start_time.elapsed().as_secs_f64();
                
                matrix.slice_mut(s![start..end, ..]).assign(&chunk);
                stats.push(ChunkStats {
                    processing_time: elapsed,
                    memory_used: chunk_size,
                    chunk_size: end - start,
                });
            }
        }
        
        Ok(stats)
    }
    
    /// Get optimization statistics
    pub fn get_stats(&self) -> Vec<ChunkStats> {
        self.stats.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_chunk_processing() {
        let config = ChunkConfig::default();
        let memory_pool = Arc::new(RwLock::new(MemoryPool::new(1 << 30)));
        let manager = ChunkManager::new(config.clone(), memory_pool);
        
        // Create test matrix
        let mut matrix = Array2::zeros((2048, 1024));
        
        // Process matrix in chunks
        let stats = manager.process_matrix(&mut matrix, |chunk| {
            // Simple test operation: set all elements to 1
            chunk.fill(1.0);
            Ok(())
        }).unwrap();
        
        assert!(matrix.iter().all(|&x| x == 1.0));
        assert!(!stats.is_empty());
        assert!(stats.iter().all(|s| s.processing_time >= 0.0));
    }
}