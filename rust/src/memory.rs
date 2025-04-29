use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;
use std::time::{Instant, Duration};
use metrics::{counter, gauge};

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub total_allocations: usize,
    pub peak_memory: usize,
    pub cache_efficiency: f32,
}

/// Memory pool for tensor operations
#[derive(Debug)]
pub struct MemoryPool {
    allocated: AtomicUsize,
    max_size: usize,
    blocks: RwLock<HashMap<usize, Vec<Vec<u8>>>>,
    last_access: RwLock<HashMap<usize, Instant>>,
    cache_ttl: u64, // Time-to-live in seconds
    stats: RwLock<PoolStats>,
    peak_memory: AtomicUsize,
}

impl MemoryPool {
    /// Create a new memory pool with specified maximum size in bytes
    pub fn new(max_size: usize) -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            max_size,
            blocks: RwLock::new(HashMap::new()),
            last_access: RwLock::new(HashMap::new()),
            cache_ttl: 300, // 5 minutes default TTL
            stats: RwLock::new(PoolStats {
                hits: 0,
                misses: 0,
                evictions: 0,
                total_allocations: 0,
                peak_memory: 0,
                cache_efficiency: 0.0,
            }),
            peak_memory: AtomicUsize::new(0),
        }
    }
    
    /// Create a new memory pool with pre-warmed cache
    pub fn with_prewarm(max_size: usize, common_sizes: &[usize]) -> Self {
        let mut pool = Self::new(max_size);
        for &size in common_sizes {
            if let Some(block) = pool.allocate(size) {
                pool.free(size, block);
            }
        }
        pool
    }
    
    /// Set cache time-to-live in seconds
    pub fn set_cache_ttl(&mut self, ttl: u64) {
        self.cache_ttl = ttl;
    }
    
    /// Allocate memory block with optional recycling
    pub fn allocate(&self, size: usize) -> Option<Vec<u8>> {
        if self.allocated.load(Ordering::Relaxed) + size > self.max_size {
            self.cleanup_old_blocks();
        }
        
        let mut blocks = self.blocks.write();
        let mut last_access = self.last_access.write();
        let mut stats = self.stats.write();
        
        // Try to reuse existing block
        if let Some(block_list) = blocks.get_mut(&size) {
            if let Some(block) = block_list.pop() {
                self.allocated.fetch_sub(size, Ordering::Relaxed);
                last_access.remove(&size);
                stats.hits += 1;
                gauge!("memory_pool.cache_hits", stats.hits as f64);
                return Some(block);
            }
        }
        
        stats.misses += 1;
        gauge!("memory_pool.cache_misses", stats.misses as f64);
        
        // Update cache efficiency
        let total = stats.hits + stats.misses;
        if total > 0 {
            stats.cache_efficiency = stats.hits as f32 / total as f32;
            gauge!("memory_pool.cache_efficiency", stats.cache_efficiency as f64);
        }
        
        // Allocate new block if within limits
        if self.allocated.fetch_add(size, Ordering::Relaxed) + size <= self.max_size {
            stats.total_allocations += 1;
            counter!("memory_pool.allocations", 1);
            
            // Update peak memory usage
            let current = self.allocated.load(Ordering::Relaxed);
            let mut peak = self.peak_memory.load(Ordering::Relaxed);
            while current > peak {
                match self.peak_memory.compare_exchange_weak(
                    peak,
                    current,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        peak = current;
                        gauge!("memory_pool.peak_memory", peak as f64);
                        break;
                    }
                    Err(actual) => peak = actual,
                }
            }
            
            Some(vec![0; size])
        } else {
            self.allocated.fetch_sub(size, Ordering::Relaxed);
            None
        }
    }
    
    /// Free memory block with caching
    pub fn free(&self, size: usize, block: Vec<u8>) {
        let mut blocks = self.blocks.write();
        let mut last_access = self.last_access.write();
        
        blocks.entry(size)
            .or_insert_with(Vec::new)
            .push(block);
            
        last_access.insert(size, Instant::now());
        self.allocated.fetch_add(size, Ordering::Relaxed);
    }
    
    /// Cleanup old memory blocks
    fn cleanup_old_blocks(&self) {
        let now = Instant::now();
        let mut blocks = self.blocks.write();
        let mut last_access = self.last_access.write();
        let mut stats = self.stats.write();
        
        let old_blocks: Vec<_> = last_access
            .iter()
            .filter(|&(_, &time)| {
                now.duration_since(time).as_secs() > self.cache_ttl
            })
            .map(|(&size, _)| size)
            .collect();
            
        for size in old_blocks {
            if let Some(block_list) = blocks.get_mut(&size) {
                let freed_size = size * block_list.len();
                self.allocated.fetch_sub(freed_size, Ordering::Relaxed);
                stats.evictions += block_list.len();
                counter!("memory_pool.evictions", block_list.len() as u64);
                block_list.clear();
            }
            last_access.remove(&size);
        }
    }
    
    /// Get current memory usage
    pub fn get_usage(&self) -> MemoryUsage {
        MemoryUsage {
            allocated: self.allocated.load(Ordering::Relaxed),
            max_size: self.max_size,
            block_count: self.blocks.read().values().map(|v| v.len()).sum(),
        }
    }
    
    /// Clear all allocated memory
    pub fn clear(&self) {
        let mut blocks = self.blocks.write();
        let mut last_access = self.last_access.write();
        blocks.clear();
        last_access.clear();
        self.allocated.store(0, Ordering::Relaxed);
    }
    
    /// Get pool statistics
    pub fn get_stats(&self) -> PoolStats {
        let stats = self.stats.read().clone();
        let peak = self.peak_memory.load(Ordering::Relaxed);
        PoolStats {
            peak_memory: peak,
            ..stats
        }
    }
    
    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = PoolStats {
            hits: 0,
            misses: 0,
            evictions: 0,
            total_allocations: 0,
            peak_memory: self.peak_memory.load(Ordering::Relaxed),
            cache_efficiency: 1.0,
        };
        
        // Reset metrics
        gauge!("memory_pool.cache_hits", 0.0);
        gauge!("memory_pool.cache_misses", 0.0);
        gauge!("memory_pool.cache_efficiency", 1.0);
        gauge!("memory_pool.evictions", 0.0);
    }
    
    /// Pre-warm the cache with common block sizes
    pub fn prewarm_cache(&self, sizes: &[usize]) {
        for &size in sizes {
            // Pre-allocate blocks based on estimated usage
            let block_count = (self.max_size / size / 10).min(5);
            for _ in 0..block_count {
                if let Some(block) = self.allocate(size) {
                    self.free(size, block);
                }
            }
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Copy)]
pub struct MemoryUsage {
    pub allocated: usize,
    pub max_size: usize,
    pub block_count: usize,
}

impl MemoryUsage {
    /// Get usage ratio (0.0 - 1.0)
    pub fn usage_ratio(&self) -> f32 {
        self.allocated as f32 / self.max_size as f32
    }
    
    /// Check if memory is nearly full (>90% usage)
    pub fn is_nearly_full(&self) -> bool {
        self.usage_ratio() > 0.9
    }
    
    /// Get available memory in bytes
    pub fn available(&self) -> usize {
        self.max_size - self.allocated
    }
}

/// Memory optimization strategies
#[derive(Debug, Clone, Copy)]
pub enum OptimizationStrategy {
    /// Aggressive optimization when memory is nearly full
    Aggressive,
    /// Balance between memory usage and performance
    Balanced,
    /// Optimize for performance, use more memory
    Performance,
}

impl OptimizationStrategy {
    /// Get compression ratio based on current memory usage
    pub fn get_compression_ratio(&self, usage: &MemoryUsage) -> f32 {
        match self {
            OptimizationStrategy::Aggressive => {
                if usage.is_nearly_full() {
                    0.5  // Aggressive compression
                } else {
                    0.8  // Moderate compression
                }
            }
            OptimizationStrategy::Balanced => {
                if usage.is_nearly_full() {
                    0.7  // Moderate compression
                } else {
                    0.9  // Light compression
                }
            }
            OptimizationStrategy::Performance => {
                if usage.is_nearly_full() {
                    0.8  // Light compression
                } else {
                    1.0  // No compression
                }
            }
        }
    }
    
    /// Get quantization bits based on current memory usage
    pub fn get_quantization_bits(&self, usage: &MemoryUsage) -> u8 {
        match self {
            OptimizationStrategy::Aggressive => {
                if usage.is_nearly_full() {
                    4  // Use 4-bit quantization
                } else {
                    8  // Use 8-bit quantization
                }
            }
            OptimizationStrategy::Balanced => {
                if usage.is_nearly_full() {
                    8  // Use 8-bit quantization
                } else {
                    16  // Use 16-bit quantization
                }
            }
            OptimizationStrategy::Performance => {
                if usage.is_nearly_full() {
                    16  // Use 16-bit quantization
                } else {
                    32  // Use full precision
                }
            }
        }
    }
} 