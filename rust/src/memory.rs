use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;

/// Memory pool for tensor operations
#[derive(Debug)]
pub struct MemoryPool {
    allocated: usize,
    max_size: usize,
    blocks: RwLock<HashMap<usize, Vec<u8>>>,
}

impl MemoryPool {
    /// Create a new memory pool with specified maximum size in bytes
    pub fn new(max_size: usize) -> Self {
        Self {
            allocated: 0,
            max_size,
            blocks: RwLock::new(HashMap::new()),
        }
    }
    
    /// Allocate memory block
    pub fn allocate(&mut self, size: usize) -> Option<Vec<u8>> {
        if self.allocated + size > self.max_size {
            return None;
        }
        
        let mut blocks = self.blocks.write();
        if let Some(block) = blocks.remove(&size) {
            return Some(block);
        }
        
        self.allocated += size;
        Some(vec![0; size])
    }
    
    /// Free memory block
    pub fn free(&mut self, size: usize, block: Vec<u8>) {
        let mut blocks = self.blocks.write();
        blocks.insert(size, block);
        self.allocated -= size;
    }
    
    /// Get current memory usage
    pub fn get_usage(&self) -> MemoryUsage {
        let blocks = self.blocks.read();
        MemoryUsage {
            allocated: self.allocated,
            max_size: self.max_size,
            block_count: blocks.len(),
        }
    }
    
    /// Clear all allocated memory
    pub fn clear(&mut self) {
        let mut blocks = self.blocks.write();
        blocks.clear();
        self.allocated = 0;
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